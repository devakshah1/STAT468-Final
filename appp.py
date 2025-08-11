import logging
from shiny import App, ui, render, reactive
import duckdb
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import shinyswatch
from lets_plot import *
LetsPlot.setup_html()
from matplotlib import pyplot as plt

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logging.info("App Started")

EXPLORER_S3_URL = "s3://devakshah-stat468-models/regression_input_data/20250810T214442Z-5c9e6/regression_input_data.csv"
PROSPECTS_S3_URL = "s3://devakshah-stat468-models/prospects_2020_to_2025_data/20250810T234638Z-0844c/prospects_2020_to_2025_data.csv"
DUCKDB_CONN = duckdb.connect()

load_dotenv()

if "S3_REGION" in os.environ and "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
    DUCKDB_CONN.sql(f"SET s3_region='{os.environ['S3_REGION']}';")
    DUCKDB_CONN.sql(f"SET s3_access_key_id='{os.environ['AWS_ACCESS_KEY_ID']}';")
    DUCKDB_CONN.sql(f"SET s3_secret_access_key='{os.environ['AWS_SECRET_ACCESS_KEY']}';")
    logging.info("DuckDB S3 credentials set successfully")
else:
    missing = [var for var in ["S3_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] if var not in os.environ]
    logging.critical(f"Environment variable(s) missing: {missing}")

MODEL_API_URL = "http://3.137.208.59:8000/predict"
MODEL_SUMMARY_URL = "http://3.137.208.59:8000/model_summary"

def predict_via_api(df):
    logging.debug("Sending prediction request to API")
    response = requests.post(MODEL_API_URL, headers={"Content-Type": "application/json"},
                             json={"data": df.to_dict(orient="records")})

    if response.status_code != 200:
        logging.warning(f"Prediction failed with status code: {response.status_code}")
        return pd.DataFrame({"error": [f"Prediction failed: {response.status_code}"]})

    json_resp = response.json()
    if not json_resp:
        logging.error("Empty response from prediction API")
        return pd.DataFrame({"error": ["Empty response from prediction API"]})

    result = pd.DataFrame(json_resp)
    logging.debug(f"Prediction API returned data: {result.head()}")
    return result

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Home",
        ui.h2("Welcome to the NHL Prospect Explorer"),
        ui.p("This app is designed to let you load data used to train the model, and predict the probability of NHL prospects becoming full-time NHLers. Explore the tabs for different features.")
    ),
    ui.nav_panel(
        "Training Data Explorer",
        ui.h2("Explore the Training Data"),
        ui.p("This is the data that was used to train the model. You can filter the data using the controls below. " \
        "The data contains most OHL forwards drafted to the NHL since 2000, along with their draft year stats and whether they became full-time NHL players (played over 200 NHL games)."),
        ui.output_text_verbatim("summary"),
        ui.output_ui("filter_selection"),
        ui.input_action_button("apply_btn", "Apply Filters"),
        ui.output_table("filtered_table"),
    ),
    ui.nav_panel(
    "Training Data Plots",
    ui.p("Let's explore some graphs to see how different variables relate to the number of NHL games played by prospects. This is a good way to visualize impact the model's variables have."),
    ui.output_plot("nhl_gp_vs_apg_plot"),
    ui.output_plot("nhl_gp_vs_gpg_plot"),
    ui.output_plot("nhl_gp_vs_height_plot"),
    ui.output_plot("nhl_gp_vs_weight_plot"),
    ),
    ui.nav_panel(
        "Model",
        ui.h2("Predictive Model"),
        ui.p("In the dropdown below, we have all OHL forwards drafted to the NHL since 2020. Select a player below to generate my model's prediction of them becoming a full-time NHL player. The prediction is based on the player's build and draft year stats. " \
        "The prediction will output as the last column in the output, and will be a probability between 0 and 1. The probability is the chance that the player will become a full-time NHLer (play over 200 NHL games)."),
        ui.output_ui("player_dropdown"),
        ui.input_action_button("predict_btn", "Predict on Selected Player"),
        ui.output_table("prediction_output"),
        ui.h4("Inputs to be sent to Model through API:"),
        ui.output_text_verbatim("sent_row")
    ),
    ui.nav_panel(
        "Model Coefficients",
        ui.h2("Model Coefficients"),
        ui.p("After running logistic regression on the data in the Data Explorer tab, here are the coefficients for each variable in the model. These coefficients represent the impact of each variable on the log-odds of a prospect becoming a full-time NHL player."),
        ui.output_text_verbatim("model_coefficients_output")
    ),
    title="OHL Prospect Evaluation App",
    id="main_page",
    theme=shinyswatch.theme.superhero()
)

def server(input, output, session):
    @reactive.Calc
    def explorer_df():
        return DUCKDB_CONN.execute(f"SELECT * FROM read_csv_auto('{EXPLORER_S3_URL}')").df()

    @reactive.Calc
    def model_data_df():
        return DUCKDB_CONN.execute(f"SELECT * FROM read_csv_auto('{PROSPECTS_S3_URL}')").df()

    @output
    @render.text
    def summary():
        return f"{len(explorer_df())} rows loaded."

    #filter on data explorer
    @output
    @render.ui
    def filter_selection():
        numeric = explorer_df().select_dtypes("number").columns.tolist()
        return [
            ui.input_select("filter_col", "Numeric column", choices=numeric),
            ui.input_numeric("threshold", "Minimum value", value=0)
        ]
    
    @reactive.Calc
    def filtered_df():
        df = explorer_df()
        if input.apply_btn() == 0: #want the data to be loaded on app start, once user inputs a filter, it will apply
            return df
        col = input.filter_col()
        thr = input.threshold()
        logging.info(f"Filter applied: {col} >= {thr}")
        return df[df[col] >= thr]

    @output
    @render.table
    def filtered_table():
        return filtered_df()

    #dropdown for predictions
    @output
    @render.ui
    def player_dropdown():
        choices = sorted(model_data_df()["player"].dropna().unique().tolist())
        return ui.input_select("selected_player", "Select Player", choices=choices)

    #get prediction from api
    @reactive.Calc
    @reactive.event(input.predict_btn)
    def prediction():
        selected = input.selected_player()
        df = model_data_df()

        cols_order = ['gp', 'height_cm', 'weight_kg', 'age_days', 'gpg', 'apg']

        #for debugging purposes
        if not all(col in df.columns for col in cols_order):
            missing_cols = [col for col in cols_order if col not in df.columns]
            logging.error(f"Missing required columns for prediction: {missing_cols}")
            return pd.DataFrame({"error": ["Missing required columns for prediction."]})
        
        logging.info(f"Preparing data for prediction for player: {selected}")
        send_df = df[df["player"] == selected][cols_order].copy()
        send_df.insert(0, 'const', 1.0)
        return predict_via_api(send_df)

    @output
    @render.table
    def prediction_output():
        return prediction()

    #show row sent to api
    @output
    @render.text
    def sent_row():
        selected = input.selected_player()
        df = model_data_df()
        cols = ['gp', 'height_cm', 'weight_kg', 'age_days', 'gpg', 'apg']
        send_df = df[df["player"] == selected][cols].copy()
        send_df.insert(0, 'const', 1.0)
        logging.info(f"Data sent to API for player {selected}:\n{send_df}")
        return send_df.to_string(index=False)

    #show model coefficients
    @output
    @render.text
    def model_coefficients_output():
        response = requests.get(MODEL_SUMMARY_URL)
        logging.info(f"Model Coefficients Requests Response: {response.status_code}")
        if response.status_code == 200:
            return str(response.json().get("coeficcients", "No summary returned."))
        return f"Error: Status {response.status_code}"
    
    @output
    @render.plot
    def nhl_gp_vs_apg_plot():
        df = explorer_df()
        plt.scatter(df['apg'], df['nhl_gp'], alpha=0.5)
        plt.xlabel("OHL Draft Year Assists Per Game")
        plt.ylabel("NHL Games Played")
        plt.title("NHL Games Played vs OHL DY Assists Per Game")
        plt.tight_layout()

    @output
    @render.plot
    def nhl_gp_vs_gpg_plot():
        df = explorer_df()
        plt.scatter(df['gpg'], df['nhl_gp'], alpha=0.5)
        plt.xlabel("OHL Draft Year Goals Per Game")
        plt.ylabel("NHL Games Played")
        plt.title("NHL Games Played vs OHL DY Goals Per Game")
        plt.tight_layout()

    @output
    @render.plot
    def nhl_gp_vs_height_plot():
        df = explorer_df()
        plt.scatter(df['height_cm'], df['nhl_gp'], alpha=0.5)
        plt.xlabel("OHL Draft Year Height (cm)")
        plt.ylabel("NHL Games Played")
        plt.title("NHL Games Played vs OHL DY Height")
        plt.tight_layout()

    @output
    @render.plot
    def nhl_gp_vs_weight_plot():
        df = explorer_df()
        plt.scatter(df['weight_kg'], df['nhl_gp'], alpha=0.5)
        plt.xlabel("OHL Draft Year Weight (kg)")
        plt.ylabel("NHL Games Played")
        plt.title("NHL Games Played vs OHL DY Weight")
        plt.tight_layout()

app = App(app_ui, server)