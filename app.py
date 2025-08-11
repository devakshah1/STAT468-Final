import logging
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

# Configure the log object with debug level to capture all messages
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logging.info("App Started")
import shinyswatch
from lets_plot import *
LetsPlot.setup_html()

# Configure the log object with debug level to capture all messages
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logging.info("App Started")

# S3 and DuckDB setup
EXPLORER_S3_URL = "s3://devakshah-stat468-models/regression_input_data/20250810T214442Z-5c9e6/regression_input_data.csv"
PROSPECTS_S3_URL = "s3://devakshah-stat468-models/prospects_2020_to_2025_data/20250810T234638Z-0844c/prospects_2020_to_2025_data.csv"
EXPLORER_S3_URL = "s3://devakshah-stat468-models/regression_input_data/20250810T214442Z-5c9e6/regression_input_data.csv"
PROSPECTS_S3_URL = "s3://devakshah-stat468-models/prospects_2020_to_2025_data/20250810T234638Z-0844c/prospects_2020_to_2025_data.csv"
DUCKDB_CONN = duckdb.connect()

load_dotenv()  # Load environment variables from .env file

if "S3_REGION" in os.environ and "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
    DUCKDB_CONN.sql(f"SET s3_region='{os.environ['S3_REGION']}';")
    DUCKDB_CONN.sql(f"SET s3_access_key_id='{os.environ['AWS_ACCESS_KEY_ID']}';")
    DUCKDB_CONN.sql(f"SET s3_secret_access_key='{os.environ['AWS_SECRET_ACCESS_KEY']}';")
    logging.info("DuckDB S3 credentials set successfully")
else:
    missing = [var for var in ["S3_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] if var not in os.environ]
    logging.critical(f"Environment variable(s) missing: {missing}")

if "S3_REGION" in os.environ and "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
    DUCKDB_CONN.sql(f"SET s3_region='{os.environ['S3_REGION']}';")
    DUCKDB_CONN.sql(f"SET s3_access_key_id='{os.environ['AWS_ACCESS_KEY_ID']}';")
    DUCKDB_CONN.sql(f"SET s3_secret_access_key='{os.environ['AWS_SECRET_ACCESS_KEY']}';")
    logging.info("DuckDB S3 credentials set successfully")
else:
    missing = [var for var in ["S3_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] if var not in os.environ]
    logging.critical(f"Environment variable(s) missing: {missing}")

# Vetiver API endpoints 
MODEL_API_URL = "http://3.137.208.59:8000/predict"
MODEL_SUMMARY_URL = "http://3.137.208.59:8000/model_summary"

data_explorer_df = reactive.Value(pd.DataFrame())
filtered = reactive.Value(pd.DataFrame())
model_df = reactive.Value(pd.DataFrame())
predictions = reactive.Value(pd.DataFrame())
sent_row_text = reactive.Value("")
model_summary_text = reactive.Value("")
model_summary_text = reactive.Value("")

def predict_via_api(df):
    logging.debug("Sending prediction request to API")

    response = requests.post(MODEL_API_URL, headers={"Content-Type": "application/json"},
                             json={"data": df.to_dict(orient="records")})

def predict_via_api(df):
    logging.debug("Sending prediction request to API")

    response = requests.post(MODEL_API_URL, headers={"Content-Type": "application/json"},
                             json={"data": df.to_dict(orient="records")})

    if response.status_code != 200:
        logging.warning(f"Prediction failed with status code: {response.status_code}")
        return pd.DataFrame(f"Prediction failed with status code: {response.status_code}")

    json_resp = response.json()
    if not json_resp:
        logging.error("Empty response from prediction API")
        return pd.DataFrame("Empty response from prediction API")

    result = pd.DataFrame(json_resp)
    logging.debug(f"Prediction API returned data: {result.head()}")
    return result

        logging.warning(f"Prediction failed with status code: {response.status_code}")
        return pd.DataFrame(f"Prediction failed with status code: {response.status_code}")

    json_resp = response.json()
    if not json_resp:
        logging.error("Empty response from prediction API")
        return pd.DataFrame("Empty response from prediction API")

    result = pd.DataFrame(json_resp)
    logging.debug(f"Prediction API returned data: {result.head()}")
    return result

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Home",
        ui.h2("Welcome to the NHL Prospect Explorer"),
        ui.p("This app is designed to let you load data used to train the model, and predict the probability of NHL prospects becoming full-time NHLers. Explore the tabs for different features")
        ui.p("This app is designed to let you load data used to train the model, and predict the probability of NHL prospects becoming full-time NHLers. Explore the tabs for different features")
    ),
    ui.nav_panel(
        "Data Explorer",
        ui.h2("Explore the Training Data"),
        ui.h2("Explore the Training Data"),
        ui.output_text_verbatim("summary"),
        ui.output_ui("dynamic_ui"),
        ui.input_action_button("apply_btn", "Apply Filters"),
        ui.output_table("filtered_table"),
    ),
    ui.nav_panel(
        "Model",
        ui.h2("Predictive Model"),
        ui.p("Select a player below to generate prediction. The prediction will output as the last column in the output, and will be a probability between 0 and 1. The probability is the chance that the player will become a full-time NHLer (play over 200 NHL games)."),
        ui.p("Select a player below to generate prediction. The prediction will output as the last column in the output, and will be a probability between 0 and 1. The probability is the chance that the player will become a full-time NHLer (play over 200 NHL games)."),
        ui.output_ui("player_dropdown"),
        ui.input_action_button("predict_btn", "Predict on Selected Player"),
        ui.output_table("prediction_output"),
        ui.h4("Inputs sent to Model through API:"),
        ui.h4("Inputs sent to Model through API:"),
        ui.output_text_verbatim("sent_row")
    ),
    ui.nav_panel(
        "Model Coefficients",
        ui.h2("Model Coefficients"),
        "Model Coefficients",
        ui.h2("Model Coefficients"),
        ui.output_text_verbatim("model_summary_output")
    ),
    title="NHL Prospect App",
    id="main_page",
    theme=shinyswatch.theme.superhero()
    id="main_page",
    theme=shinyswatch.theme.superhero()
)

def server(input, output, session):
    @reactive.Effect
    def fetch_model_summary_on_start():
        logging.info("Fetching model coefficients from API")

        logging.info("Fetching model coefficients from API")

        response = requests.get(MODEL_SUMMARY_URL)
        if response.status_code == 200:
            summary = response.json().get("coeficcients", "No summary returned.")
            logging.debug(f"Model coefficients received: {summary}")
            summary = response.json().get("coeficcients", "No summary returned.")
            logging.debug(f"Model coefficients received: {summary}")
        else:
            summary = f"Error: Status {response.status_code}"
            logging.warning(f"Failed to get model coefficients: status {response.status_code}")

            summary = f"Error: Status {response.status_code}"
            logging.warning(f"Failed to get model coefficients: status {response.status_code}")

        model_summary_text.set(summary)

    @reactive.Effect
    def load_data_once():
        reactive.invalidate_later(100, session=session)  #delay to make sure app is loaded before loading data
        reactive.invalidate_later(100, session=session)  #delay to make sure app is loaded before loading data
        if data_explorer_df().empty:
            logging.info("Loading explorer and prospects data from S3 via DuckDB")
            logging.info("Loading explorer and prospects data from S3 via DuckDB")
            explorer_df = DUCKDB_CONN.execute(f"SELECT * FROM read_csv_auto('{EXPLORER_S3_URL}')").df()
            model_data = DUCKDB_CONN.execute(f"SELECT * FROM read_csv_auto('{PROSPECTS_S3_URL}')").df()
            logging.info(f"Data explorer data loaded")
            logging.info(f"Prospects data loaded for model inputs")
            model_data = DUCKDB_CONN.execute(f"SELECT * FROM read_csv_auto('{PROSPECTS_S3_URL}')").df()
            logging.info(f"Data explorer data loaded")
            logging.info(f"Prospects data loaded for model inputs")
            data_explorer_df.set(explorer_df)
            filtered.set(explorer_df)
            model_df.set(model_data)

    @output
    @render.text
    def model_summary_output():
        return model_summary_text()

    @output
    @render.text
    def summary():
        df = data_explorer_df()
        text = f"{len(df)} rows loaded." if not df.empty else "No data loaded yet."
        return text
        text = f"{len(df)} rows loaded." if not df.empty else "No data loaded yet."
        return text

    @output
    @render.ui
    def dynamic_ui():
        df = data_explorer_df()
        if df.empty:
            logging.debug("Data is empty, no filters to show")
            return ui.p("No data available to filter.")
            logging.debug("Data is empty, no filters to show")
            return ui.p("No data available to filter.")
        numeric = df.select_dtypes("number").columns.tolist()
        return [
            ui.input_select("filter_col", "Numeric column", choices=numeric),
            ui.input_numeric("threshold", "Minimum value", value=0)
        ]

    @reactive.effect
    @reactive.event(input.apply_btn)
    def _():
        logging.info("Apply Filters button clicked")
        logging.info("Apply Filters button clicked")
        df = data_explorer_df()
        col = input.filter_col()
        thr = input.threshold()
        logging.debug(f"Filtering data on column '{col}' with threshold {thr}")
        filtered_df = df[df[col] >= thr] if col in df.columns else df
        filtered.set(filtered_df)
        logging.info(f"Filtered data has {len(filtered_df)} rows")
        col = input.filter_col()
        thr = input.threshold()
        logging.debug(f"Filtering data on column '{col}' with threshold {thr}")
        filtered_df = df[df[col] >= thr] if col in df.columns else df
        filtered.set(filtered_df)
        logging.info(f"Filtered data has {len(filtered_df)} rows")

    @output
    @render.table
    def filtered_table():
        return filtered()

    @output
    @render.ui
    def player_dropdown():
        df = model_df()
        if df.empty or "player" not in df.columns:
            logging.warning("Model dataframe empty or 'player' column missing for dropdown")
            logging.warning("Model dataframe empty or 'player' column missing for dropdown")
            return ui.input_select("selected_player", "Select Player", choices=[], selected=None)
        choices = sorted(df["player"].dropna().unique().tolist())
        logging.debug(f"Player dropdown choices updated: {choices[:5]}{'...' if len(choices) > 5 else ''}")
        logging.debug(f"Player dropdown choices updated: {choices[:5]}{'...' if len(choices) > 5 else ''}")
        return ui.input_select("selected_player", "Select Player", choices=choices, selected=choices[0])

    @reactive.effect
    @reactive.event(input.predict_btn)
    def _():
        logging.info("Predict button clicked")
        logging.info("Predict button clicked")
        df = model_df()
        selected = input.selected_player()
        if df.empty or not selected:
            logging.warning("No player selected or model dataframe empty")
            logging.warning("No player selected or model dataframe empty")
            predictions.set(pd.DataFrame({"message": ["No player selected for prediction."]}))
            sent_row_text.set("No player selected.")
        else:
            logging.info(f"Prediction requested for player: {selected}")
            logging.info(f"Prediction requested for player: {selected}")
            selected_df = df[df["player"] == selected]
            if selected_df.empty:
                logging.warning(f"Player '{selected}' not found in data")
                logging.warning(f"Player '{selected}' not found in data")
                predictions.set(pd.DataFrame({"message": ["Selected player not found in data."]}))
                sent_row_text.set("Player not found.")
            else:
                cols_order = ['gp', 'height_cm', 'weight_kg', 'age_days', 'gpg', 'apg']
                missing_cols = [col for col in cols_order if col not in selected_df.columns]
                if missing_cols:
                    logging.error(f"Missing required columns for prediction: {missing_cols}")
                cols_order = ['gp', 'height_cm', 'weight_kg', 'age_days', 'gpg', 'apg']
                missing_cols = [col for col in cols_order if col not in selected_df.columns]
                if missing_cols:
                    logging.error(f"Missing required columns for prediction: {missing_cols}")
                    predictions.set(pd.DataFrame({"error": ["Missing required columns for prediction."]}))
                    sent_row_text.set("Missing required columns for prediction.")
                    return

                send_df = selected_df[cols_order].copy()
                send_df.insert(0, 'const', 1.0)
                logging.debug(f"Sending data for prediction:\n{send_df}")
                logging.debug(f"Sending data for prediction:\n{send_df}")
                sent_row_text.set(send_df.to_string(index=False))
                preds = predict_via_api(send_df)
                predictions.set(preds)

    @output
    @render.table
    def prediction_output():
        return predictions()

    @output
    @render.text
    def sent_row():
        return sent_row_text()

app = App(app_ui, server)