"""Streamlit GUI for Metro Interstate highway traffic-volume estimation."""

from __future__ import annotations

import hashlib
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import celsius_to_kelvin, make_prediction_features, train_and_evaluate

st.set_page_config(
    page_title="Traffic Volume Forecasting",
    page_icon="🚦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5eaf2;
        padding: 14px 16px; border-radius: 12px;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #000000 !important;
    }
    .intro {color: #536174; font-size: 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚦 Traffic Volume Forecasting")
st.markdown(
    '<p class="intro">Estimate hourly highway traffic volume from date, time, '
    "and weather conditions using a Random Forest trained on the Metro Interstate dataset.</p>",
    unsafe_allow_html=True,
)
st.info(
    "This dataset represents a highway in Minnesota, USA. This application is a "
    "historical-data-based estimator, not a live traffic monitoring system."
)

st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload a different Metro Interstate CSV (optional)",
    type=["csv"],
    help="The official UCI dataset included with this project is loaded by default.",
)
bundled_dataset_path = (
    Path(__file__).parent / "data" / "Metro_Interstate_Traffic_Volume.csv"
)
if uploaded_file is None and bundled_dataset_path.is_file():
    dataset_label = "Bundled UCI dataset"
    try:
        csv_content = bundled_dataset_path.read_bytes()
    except OSError as error:
        st.error(f"Unable to read the bundled dataset: {error}")
        st.stop()
elif uploaded_file is None:
    st.subheader("Get started")
    st.write(
        "Download the Metro Interstate Traffic Volume Dataset from UCI, extract the "
        "CSV file, and upload it using the sidebar. The model is trained from the "
        "uploaded data before the prediction form becomes available."
    )
    st.markdown(
        """
        **Expected file:** the original Metro Interstate CSV with `date_time`,
        `traffic_volume`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`,
        `weather_main`, and `holiday` columns.

        **Data science lifecycle:** understand the data → clean and engineer features →
        train a chronological baseline → evaluate on held-out records → estimate a
        user-specified hour.
        """
    )
    st.stop()
else:
    dataset_label = uploaded_file.name
    csv_content = uploaded_file.getvalue()

content_hash = hashlib.sha256(csv_content).hexdigest()
try:
    dataset = pd.read_csv(
        BytesIO(csv_content),
        keep_default_na=False,
        na_values=[""],
    )
except (UnicodeDecodeError, pd.errors.ParserError, OSError) as error:
    st.error(f"Unable to read the uploaded CSV. Check that it is a valid CSV file. Details: {error}")
    st.stop()

try:
    @st.cache_resource(show_spinner="Cleaning data, training the model, and evaluating it...")
    def get_training_result(_content_hash: str, csv_bytes: bytes):
        frame = pd.read_csv(
            BytesIO(csv_bytes),
            keep_default_na=False,
            na_values=[""],
        )
        return train_and_evaluate(frame)

    training = get_training_result(content_hash, csv_content)
except (ValueError, KeyError, TypeError) as error:
    st.error(str(error))
    st.stop()

st.subheader("Dataset overview")
st.caption(f"Dataset source: {dataset_label}")
overview = st.columns(4)
overview[0].metric("Uploaded records", f"{len(dataset):,}")
overview[1].metric("Records used", f"{training.rows_used:,}")
overview[2].metric("Duplicate rows removed", f"{training.duplicate_rows:,}")
overview[3].metric("Invalid records removed", f"{training.discarded_rows:,}")
st.caption(
    f"Observed period: {training.start_time:%Y-%m-%d %H:%M} to "
    f"{training.end_time:%Y-%m-%d %H:%M} · Target: traffic volume (vehicles/hour)"
)
with st.expander("Preview uploaded data"):
    st.dataframe(dataset.head(10), use_container_width=True)

st.subheader("Predict traffic volume")
st.write("Enter conditions for a single hour. The supplied defaults are examples.")

weather_values = sorted(
    value
    for value in dataset["weather_main"].dropna().astype(str).unique().tolist()
    if value.strip()
)
holiday_values = sorted(
    value
    for value in dataset["holiday"].dropna().astype(str).unique().tolist()
    if value.strip()
)
if not weather_values or not holiday_values:
    st.error("The uploaded dataset has no usable weather or holiday categories.")
    st.stop()

with st.form("prediction_inputs"):
    date_col, hour_col = st.columns(2)
    with date_col:
        prediction_date = st.date_input("Date", value=date(2018, 9, 30))
    with hour_col:
        hour = st.slider("Hour (0–23)", min_value=0, max_value=23, value=17)

    temp_col, rain_col, snow_col = st.columns(3)
    with temp_col:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-100.0,
            max_value=100.0,
            value=14.85,
            step=0.1,
        )
    with rain_col:
        rainfall = st.number_input(
            "Rainfall (mm/h)", min_value=0.0, value=0.0, step=0.1
        )
    with snow_col:
        snowfall = st.number_input(
            "Snowfall (mm/h)", min_value=0.0, value=0.0, step=0.1
        )

    cloud_col, weather_col, holiday_col = st.columns(3)
    with cloud_col:
        cloud_cover = st.slider("Cloud cover (%)", min_value=0, max_value=100, value=40)
    with weather_col:
        weather_default = weather_values.index("Rain") if "Rain" in weather_values else 0
        weather = st.selectbox(
            "Weather condition", weather_values, index=weather_default
        )
    with holiday_col:
        holiday_default = holiday_values.index("None") if "None" in holiday_values else 0
        holiday = st.selectbox(
            "Holiday status", holiday_values, index=holiday_default
        )
    predict_clicked = st.form_submit_button(
        "Predict Traffic Volume", type="primary", use_container_width=True
    )

if predict_clicked:
    try:
        prediction_features = make_prediction_features(
            prediction_date=pd.Timestamp(prediction_date),
            hour=hour,
            temperature=celsius_to_kelvin(temperature),
            rainfall=rainfall,
            snowfall=snowfall,
            cloud_cover=cloud_cover,
            weather=weather,
            holiday=holiday,
        )
        prediction = float(training.model.predict(prediction_features)[0])
    except ValueError as error:
        st.error(str(error))
    else:
        st.success("Prediction generated by the model trained on your uploaded dataset.")
        st.metric("Estimated traffic volume", f"{prediction:,.0f} vehicles per hour")
        st.caption(
            f"For {prediction_date:%Y-%m-%d} at {hour:02d}:00 · "
            f"{weather} · Holiday: {holiday}"
        )

st.subheader("Model evaluation")
st.write(
    "The model is evaluated on the final 20% of the observations in chronological "
    "order. That later period is excluded from evaluation-model training to reduce "
    "future-data leakage. The prediction model is then refit on all valid uploaded "
    "records."
)
metric_columns = st.columns(4)
for column, metric_name in zip(metric_columns, ["MAE", "MSE", "RMSE", "R²"]):
    column.metric(metric_name, f"{training.metrics[metric_name]:,.3f}")

chart_period = st.radio(
    "Chart time range",
    options=["Last 7 days", "Last 30 days", "Last 90 days", "Entire test set"],
    horizontal=True,
    index=0,
)
evaluation_end = training.evaluation.index.max()
days_by_period = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
}
if chart_period == "Entire test set":
    chart_data = training.evaluation
else:
    chart_start = evaluation_end - pd.Timedelta(days=days_by_period[chart_period])
    chart_data = training.evaluation.loc[training.evaluation.index >= chart_start]

evaluation_chart = go.Figure()
evaluation_chart.add_trace(
    go.Scatter(
        x=chart_data.index,
        y=chart_data["Actual"],
        mode="lines",
        name="Observed",
        line={"color": "#2563EB", "width": 2.6},
        hovertemplate="%{x|%b %d, %Y · %H:%M}<br>Observed: %{y:,.0f} vehicles/h<extra></extra>",
    )
)
evaluation_chart.add_trace(
    go.Scatter(
        x=chart_data.index,
        y=chart_data["Predicted"],
        mode="lines",
        name="Model prediction",
        line={"color": "#F97316", "width": 2, "dash": "dot"},
        hovertemplate="%{x|%b %d, %Y · %H:%M}<br>Predicted: %{y:,.0f} vehicles/h<extra></extra>",
    )
)
evaluation_chart.update_layout(
    template="plotly_white",
    title={
        "text": (
            "Observed vs. predicted traffic"
            "<br><sup>Chronological holdout · "
            f"{len(chart_data):,} hourly observations · {chart_period}</sup>"
        ),
        "x": 0.02,
        "xanchor": "left",
        "font": {"size": 21, "color": "#172033"},
    },
    xaxis={
        "title": {"text": "Date and time", "font": {"color": "#475569"}},
        "tickfont": {"color": "#475569"},
        "showgrid": False,
        "linecolor": "#D9E2EF",
    },
    yaxis={
        "title": {
            "text": "Traffic volume (vehicles/hour)",
            "font": {"color": "#475569"},
        },
        "tickformat": ",",
        "tickfont": {"color": "#475569"},
        "gridcolor": "#E9EEF5",
        "zeroline": False,
    },
    hovermode="x",
    font={"family": "Arial, sans-serif", "color": "#334155"},
    hoverlabel={"bgcolor": "#172033", "font": {"color": "white", "size": 13}},
    height=540,
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "x": 1,
        "xanchor": "right",
        "font": {"color": "#334155", "size": 13},
    },
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin={"l": 75, "r": 24, "t": 105, "b": 32},
)
st.plotly_chart(evaluation_chart, use_container_width=True, config={"displaylogo": False})

with st.expander("How the model works and limitations"):
    st.markdown(
        """
        - The uploaded CSV is cleaned by dropping duplicate records and rows without
          a valid timestamp or target value. Numeric feature gaps are imputed using
          training-set medians; missing weather/holiday values use the most common
          training category.
        - Calendar features (hour, day, month, weekday, year, day of year, and cyclic
          time encodings) are extracted from `date_time`.
        - Weather and holiday are one-hot encoded by the same Scikit-learn pipeline
          used for model training and GUI predictions. Unseen categories are handled
          safely by the encoder.
        - Evaluation reports MAE, MSE, RMSE, and R² on an ordered holdout period.
          The plotted values are actual predictions made before refitting on all
          records for interactive use.
        - This first version estimates one hourly value from supplied conditions. It
          is not a sequence-based future forecast, and it does not use live traffic,
          holiday calendars, incidents, or external weather services.
        """
    )
