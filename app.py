from shiny import App, ui, render, reactive
import duckdb
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# ========================
# 🔧 CONFIGURATION
# ======================== 

# S3 and DuckDB setup
EXPLORER_S3_URL = "s3://devakshah-stat468-models/regression_input_data/20250805T110702Z-466c5/regression_input_data.csv"
MODEL_S3_URL = "s3://devakshah-stat468-models/prospects_2020_to_2025_data/20250806T133431Z-50ed5/prospects_2020_to_2025_data.csv"
DUCKDB_CONN = duckdb.connect()

load_dotenv()  # Load environment variables from .env file
DUCKDB_CONN.sql(f"SET s3_region='{os.environ['S3_REGION']}';")
DUCKDB_CONN.sql(f"SET s3_access_key_id='{os.environ['AWS_ACCESS_KEY_ID']}';")
DUCKDB_CONN.sql(f"SET s3_secret_access_key='{os.environ['AWS_SECRET_ACCESS_KEY']}';")

# Vetiver API endpoints
MODEL_API_URL = "http://3.137.208.59:8000/predict"
MODEL_SUMMARY_URL = "http://3.137.208.59:8000/model_summary"

# ========================
# 🔁 REACTIVE STATE
# ========================
data_explorer_df = reactive.Value(pd.DataFrame())
filtered = reactive.Value(pd.DataFrame())
model_df = reactive.Value(pd.DataFrame())
predictions = reactive.Value(pd.DataFrame())
sent_row_text = reactive.Value("")
model_summary_text = reactive.Value("Loading model summary...")

# ========================
# 📤 Prediction helper
# ========================
def predict_via_api(df: pd.DataFrame) -> pd.DataFrame:
    response = requests.post(
        MODEL_API_URL,
        headers={"Content-Type": "application/json"},
        json={"data": df.to_dict(orient="records")}
    )
    if response.status_code != 200:
        return pd.DataFrame({"error": [f"Prediction failed: {response.status_code}"]})
    return pd.DataFrame(response.json())

# ========================
# 🖥️ USER INTERFACE
# ========================
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Home",
        ui.h2("Welcome to the NHL Prospect Explorer"),
        ui.p("This app is designed to let you load data, explore it, and eventually run predictive models.")
    ),
    ui.nav_panel(
        "Data Explorer",
        ui.h2("Explore Your Data"),
        ui.output_text_verbatim("summary"),
        ui.output_ui("dynamic_ui"),
        ui.input_action_button("apply_btn", "Apply Filters"),
        ui.output_table("filtered_table"),
        ui.download_button("download_btn", "Download filtered data")
    ),
    ui.nav_panel(
        "Model",
        ui.h2("Predictive Model"),
        ui.p("Select a player below to generate predictions."),
        ui.output_ui("player_dropdown"),
        ui.input_action_button("predict_btn", "Predict on Selected Player"),
        ui.output_table("prediction_output"),
        ui.h4("Row sent to API:"),
        ui.output_text_verbatim("sent_row")
    ),
    ui.nav_panel(
        "Model Summary",
        ui.h2("Model Summary"),
        ui.output_text_verbatim("model_summary_output")
    ),
    title="NHL Prospect App",
    id="main_page"
)

# ========================
# ⚙️ SERVER LOGIC
# ========================
def server(input, output, session):
    # Load model summary once on app start
    @reactive.Effect
    def fetch_model_summary_on_start():
        response = requests.get(MODEL_SUMMARY_URL)
        if response.status_code == 200:
            summary = response.json().get("model_summary", "No summary returned.")
        else:
            summary = f"Error fetching summary: Status {response.status_code}"
        model_summary_text.set(summary)

    # Load data once on startup using invalidate_later
    @reactive.Effect
    def load_data_once():
        reactive.invalidate_later(100, session=session)  # small delay to ensure app is ready
        if data_explorer_df().empty:
            explorer_df = DUCKDB_CONN.execute(f"SELECT * FROM read_csv_auto('{EXPLORER_S3_URL}')").df()
            model_data = DUCKDB_CONN.execute(f"SELECT * FROM read_csv_auto('{MODEL_S3_URL}')").df()
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
        return f"{len(df)} rows loaded." if not df.empty else "No data loaded yet."

    @output
    @render.ui
    def dynamic_ui():
        df = data_explorer_df()
        if df.empty:
            return ui.p("Load data first to see filters.")
        numeric = df.select_dtypes("number").columns.tolist()
        return [
            ui.input_select("filter_col", "Numeric column", choices=numeric),
            ui.input_numeric("threshold", "Minimum value", value=0)
        ]

    @reactive.effect
    @reactive.event(input.apply_btn)
    def _():
        df = data_explorer_df()
        if not df.empty:
            col = input.filter_col()
            thr = input.threshold()
            filtered_df = df[df[col] >= thr] if col in df.columns else df
            filtered.set(filtered_df)

    @output
    @render.table
    def filtered_table():
        return filtered()

    @output
    @render.download()
    def download_btn():
        return {"filename": "filtered_data.csv", "content": filtered().to_csv(index=False)}

    @output
    @render.ui
    def player_dropdown():
        df = model_df()
        if df.empty or "player" not in df.columns:
            return ui.input_select("selected_player", "Select Player", choices=[], selected=None)
        choices = sorted(df["player"].dropna().unique().tolist())
        return ui.input_select("selected_player", "Select Player", choices=choices, selected=choices[0])

    @reactive.effect
    @reactive.event(input.predict_btn)
    def _():
        df = model_df()
        selected = input.selected_player()
        if df.empty or not selected:
            predictions.set(pd.DataFrame({"message": ["No player selected for prediction."]}))
            sent_row_text.set("No player selected.")
        else:
            selected_df = df[df["player"] == selected]
            if selected_df.empty:
                predictions.set(pd.DataFrame({"message": ["Selected player not found in data."]}))
                sent_row_text.set("Player not found.")
            else:
                cols_order = ['tp', 'height_cm', 'weight_kg', 'age_days', 'gpg', 'ppg']
                if not all(col in selected_df.columns for col in cols_order):
                    predictions.set(pd.DataFrame({"error": ["Missing required columns for prediction."]}))
                    sent_row_text.set("Missing required columns for prediction.")
                    return

                send_df = selected_df[cols_order].copy()
                send_df.insert(0, 'const', 1.0)
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

# ========================
# 🚀 LAUNCH APP
# ========================
app = App(app_ui, server)



