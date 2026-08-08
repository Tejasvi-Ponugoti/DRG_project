
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from xgboost import DMatrix, XGBRegressor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Dynamic Resilient Grid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# VISUAL THEME
# =========================================================

GRID_BLUE = "#234A84"
GRID_BLUE_LIGHT = "#DCE8F8"
GRID_BLUE_SOFT = "#EEF4FB"
GRID_GREEN = "#5C8F3F"
GRID_GREEN_LIGHT = "#EAF3E3"
GRID_RED = "#E54B4B"
GRID_RED_LIGHT = "#FDEAEA"
GRID_AMBER = "#E9A52D"
GRID_AMBER_LIGHT = "#FFF4DB"
GRID_BACKGROUND = "#F7F9FC"
GRID_PANEL = "#FFFFFF"
GRID_TEXT = "#24344D"
GRID_MUTED = "#61708A"
GRID_BORDER = "#C9D6EA"

st.markdown(
    f"""
    <style>
        .stApp {{
            background:
                linear-gradient(180deg, #F9FBFE 0%, {GRID_BACKGROUND} 100%);
            color: {GRID_TEXT};
        }}

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, #F4F7FB 0%, #EAF0F8 100%);
            border-right: 1px solid {GRID_BORDER};
        }}

        [data-testid="stSidebar"] * {{
            color: {GRID_TEXT};
        }}

        .block-container {{
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }}

        h1, h2, h3 {{
            color: {GRID_BLUE};
            letter-spacing: -0.02em;
        }}

        p, li, label, .stMarkdown {{
            color: {GRID_TEXT};
        }}

        [data-testid="metric-container"] {{
            background: {GRID_PANEL};
            border: 1px solid {GRID_BORDER};
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 22px rgba(35,74,132,0.08);
        }}

        [data-testid="metric-container"] label {{
            color: {GRID_MUTED} !important;
            font-weight: 600;
        }}

        [data-testid="metric-container"] [data-testid="stMetricValue"] {{
            color: {GRID_BLUE};
            font-weight: 700;
        }}

        [data-testid="stPlotlyChart"] {{
            background: {GRID_PANEL};
            border: 1px solid {GRID_BORDER};
            border-radius: 16px;
            padding: 0.45rem;
            box-shadow: 0 8px 22px rgba(35,74,132,0.07);
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {GRID_BORDER};
            border-radius: 14px;
            overflow: hidden;
            background: {GRID_PANEL};
        }}

        .stAlert {{
            border-radius: 14px;
            border: 1px solid {GRID_BORDER};
        }}

        .stButton > button,
        .stDownloadButton > button {{
            border-radius: 10px;
            border: 1px solid {GRID_BLUE};
            background: linear-gradient(135deg, {GRID_BLUE}, #3568A9);
            color: white;
            font-weight: 700;
            padding: 0.55rem 1rem;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            border-color: #5D83B8;
            background: linear-gradient(135deg, #2E5A98, #4475B7);
            color: white;
        }}

        .stSelectbox > div > div,
        .stDateInput > div > div {{
            background: white;
            border-radius: 10px;
        }}

        hr {{
            border-color: {GRID_BORDER};
        }}

        .drg-hero {{
            background:
                linear-gradient(135deg, rgba(220,232,248,0.95), rgba(234,243,227,0.88));
            border: 1px solid {GRID_BORDER};
            border-radius: 22px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 30px rgba(35,74,132,0.10);
        }}

        .drg-hero h1 {{
            margin: 0 0 0.45rem 0;
            color: {GRID_BLUE};
            font-size: 2.25rem;
        }}

        .drg-hero p {{
            margin: 0;
            color: {GRID_MUTED};
            font-size: 1.05rem;
        }}

        .drg-section-tag {{
            display: inline-block;
            padding: 0.35rem 0.72rem;
            border-radius: 999px;
            background: {GRID_BLUE_LIGHT};
            border: 1px solid #AFC5E3;
            color: {GRID_BLUE};
            font-weight: 700;
            font-size: 0.82rem;
            margin-bottom: 0.7rem;
        }}

        .drg-footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid {GRID_BORDER};
            color: {GRID_MUTED};
            font-size: 0.9rem;
            text-align: center;
        }}

        div[data-testid="stExpander"] {{
            background: {GRID_PANEL};
            border: 1px solid {GRID_BORDER};
            border-radius: 12px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_plotly_theme(figure):
    """Apply the light DRG infographic theme consistently."""
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=GRID_PANEL,
        plot_bgcolor=GRID_PANEL,
        font=dict(
            color=GRID_TEXT,
            family="Arial",
        ),
        title_font=dict(
            color=GRID_BLUE,
            size=18,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
        ),
        margin=dict(
            l=45,
            r=25,
            t=65,
            b=45,
        ),
        hoverlabel=dict(
            bgcolor=GRID_PANEL,
            font_color=GRID_TEXT,
            bordercolor=GRID_BORDER,
        ),
        colorway=[
            GRID_BLUE,
            GRID_GREEN,
            GRID_RED,
            GRID_AMBER,
            "#6D8FB3",
            "#8CB46A",
        ],
    )
    figure.update_xaxes(
        gridcolor="#E7EDF6",
        zerolinecolor="#D7E0EC",
        linecolor="#C6D2E3",
    )
    figure.update_yaxes(
        gridcolor="#E7EDF6",
        zerolinecolor="#D7E0EC",
        linecolor="#C6D2E3",
    )
    return figure


st.sidebar.markdown(
    """
    <div style="
        padding: 0.95rem 0.9rem 1rem 0.9rem;
        margin-bottom: 0.6rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #DCE8F8, #EAF3E3);
        border: 1px solid #C9D6EA;
        box-shadow: 0 8px 20px rgba(35,74,132,0.08);
    ">
        <div style="font-size:1.15rem;font-weight:800;color:#234A84;">
            ⚡ DRG Control Centre
        </div>
        <div style="font-size:0.82rem;color:#61708A;margin-top:0.2rem;">
            Forecasting • Stress • Electrification
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PROJECT SETTINGS
# =========================================================

DATA_PATH = Path("../data/processed/model_ready_data_v2.csv")
MODEL_PATH = Path("../models/xgboost_demand_model.json")
FEATURES_PATH = Path("../models/feature_names.json")

STRESS_THRESHOLD_KWH = 354.5756

PROJECT_INFOGRAPHIC_PATH = Path("assets/drg_project_infographic.png")

EV_SCENARIOS = {
    "Baseline": 0,
    "Low EV": 20,
    "Medium EV": 40,
    "High EV": 60,
    "Extreme EV": 80,
}

EV_START_HOUR = 18
EV_END_HOUR = 21

HEAT_PUMP_SCENARIOS = {
    "Baseline": 0,
    "Low Heat Pump": 10,
    "Medium Heat Pump": 20,
    "High Heat Pump": 30,
    "Extreme Heat Pump": 40,
}

HEAT_PUMP_START_HOUR = 18
HEAT_PUMP_END_HOUR = 21

COMBINED_SCENARIOS = {
    "Baseline": {"ev_load_kwh": 0, "heat_pump_load_kwh": 0},
    "Low Combined": {"ev_load_kwh": 20, "heat_pump_load_kwh": 10},
    "Medium Combined": {"ev_load_kwh": 40, "heat_pump_load_kwh": 20},
    "High Combined": {"ev_load_kwh": 60, "heat_pump_load_kwh": 30},
    "Extreme Combined": {"ev_load_kwh": 80, "heat_pump_load_kwh": 40},
}

COMBINED_START_HOUR = 18
COMBINED_END_HOUR = 21



# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path.resolve()}"
        )

    data = pd.read_csv(path)

    required_columns = [
        "datetime",
        "total_demand_kwh",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data["datetime"] = pd.to_datetime(
        data["datetime"],
        errors="coerce",
    )

    data = (
        data.dropna(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if "ramp_rate" in data.columns:
        data = data.drop(columns=["ramp_rate"])

    return data


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model(
    model_path: Path,
    feature_path: Path,
):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path.resolve()}"
        )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found at: {feature_path.resolve()}"
        )

    model = XGBRegressor()
    model.load_model(model_path)

    with open(
        feature_path,
        "r",
        encoding="utf-8",
    ) as file:
        features = json.load(file)

    if isinstance(features, dict):
        features = features.get(
            "feature_names",
            [],
        )

    if not isinstance(features, list):
        raise ValueError(
            "feature_names.json must contain a list."
        )

    return model, features


# =========================================================
# INITIALISE
# =========================================================

try:
    df = load_data(DATA_PATH)

except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()


try:
    xgb_model, feature_names = load_model(
        MODEL_PATH,
        FEATURES_PATH,
    )

except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚡ DRG Dashboard")

selected_page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Overview",
        "Demand Forecasting",
        "Stress Detection",
        "EV Simulation",
        "Heat-Pump Simulation",
        "Combined EV–Heat-Pump Simulation",
        "SHAP Explainability",
        "Model Performance & Summary",
    ],
)

st.sidebar.divider()

minimum_date = df["datetime"].min().date()
maximum_date = df["datetime"].max().date()

selected_dates = st.sidebar.date_input(
    "Select date range",
    value=(
        minimum_date,
        maximum_date,
    ),
    min_value=minimum_date,
    max_value=maximum_date,
)


# =========================================================
# GENERAL DATE FILTER
# =========================================================

if (
    isinstance(selected_dates, tuple)
    and len(selected_dates) == 2
):
    start_date, end_date = selected_dates

    filtered_df = df[
        (
            df["datetime"].dt.date >= start_date
        )
        &
        (
            df["datetime"].dt.date <= end_date
        )
    ].copy()

else:
    filtered_df = df.copy()


if filtered_df.empty:
    st.warning(
        "No observations are available for the selected period."
    )
    st.stop()


# =========================================================
# HOME
# =========================================================

if selected_page == "Home":

    # -----------------------------------------------------
    # Cover section
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="drg-hero">
            <div class="drg-section-tag">AI-ENABLED ENERGY ANALYTICS</div>
            <h1>⚡ Dynamic Resilient Grid</h1>
            <p>
                A decision-support dashboard for local electricity-demand
                forecasting, electrification-stress analysis and explainable AI.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if PROJECT_INFOGRAPHIC_PATH.exists():
        st.image(
            str(PROJECT_INFOGRAPHIC_PATH),
            caption=(
                "Dynamic Resilient Grid research workflow: "
                "forecast, detect, simulate, quantify and explain."
            ),
            use_container_width=True,
        )
    else:
        st.info(
            """
            Place `drg_project_infographic.png` inside the dashboard
            `assets` folder to display the project architecture image.
            """
        )

    st.markdown(
        """
        ### Project in one sentence

        The Dynamic Resilient Grid project combines **XGBoost electricity-demand
        forecasting**, **statistical stress detection**, **EV and heat-pump
        scenario analysis**, and **SHAP explainability** to support transparent,
        evidence-based planning for local UK electricity networks.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Headline indicators
    # -----------------------------------------------------

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric(
            "Final forecasting model",
            "XGBoost",
        )

    with metric_2:
        st.metric(
            "Model R²",
            "0.9912",
        )

    with metric_3:
        st.metric(
            "MAE",
            "5.708 kWh",
        )

    with metric_4:
        st.metric(
            "Stress threshold",
            f"{STRESS_THRESHOLD_KWH:.4f} kWh",
        )

    st.divider()

    # -----------------------------------------------------
    # Interactive project tabs
    # -----------------------------------------------------

    overview_tab, workflow_tab, questions_tab, findings_tab, guide_tab = st.tabs(
        [
            "🌍 Project Overview",
            "⚙️ System Workflow",
            "❓ Research Questions",
            "📌 Key Findings",
            "🧭 Dashboard Guide",
        ]
    )

    with overview_tab:

        st.subheader("Why this project matters")

        st.write(
            """
            The electrification of transport and domestic heating is increasing
            the pressure placed on local electricity networks. Electric vehicles
            and heat pumps can create higher evening demand when many households
            use them at similar times.

            Local planners therefore need tools that can forecast demand,
            identify unusual high-load periods and explore future electrification
            scenarios before those pressures become operational problems.
            """
        )

        problem_col, solution_col, outcome_col = st.columns(3)

        with problem_col:
            st.error(
                """
                **Current problem**

                Rising evening demand can increase local peak-load risk and
                place additional pressure on neighbourhood electricity assets.
                """
            )

        with solution_col:
            st.info(
                """
                **DRG solution**

                Forecast demand, detect analytical stress, simulate EV and
                heat-pump loads, and explain model predictions.
                """
            )

        with outcome_col:
            st.success(
                """
                **Planning outcome**

                Provide clearer evidence for resilient, transparent and
                scenario-informed local infrastructure planning.
                """
            )

        st.subheader("Main capabilities")

        capability_1, capability_2 = st.columns(2)

        with capability_1:
            st.markdown(
                """
                **Forecasting and monitoring**

                - Half-hourly electricity-demand forecasting
                - Actual-versus-predicted comparison
                - Baseline stress-event identification
                - Hourly and monthly stress patterns
                """
            )

        with capability_2:
            st.markdown(
                """
                **Scenario planning and explainability**

                - EV charging simulations
                - Heat-pump simulations
                - Combined electrification scenarios
                - Global and local SHAP explanations
                """
            )

    with workflow_tab:

        st.subheader("How the DRG system works")

        step_1, step_2 = st.columns(2)

        with step_1:
            st.info(
                """
                **1. Prepare the data**

                Clean, aggregate and structure half-hourly electricity-demand
                observations for modelling.
                """
            )

            st.info(
                """
                **2. Engineer forecasting features**

                Create time variables, lag features and rolling statistics.
                """
            )

            st.info(
                """
                **3. Forecast electricity demand**

                Apply the final XGBoost model to the unseen chronological test
                period.
                """
            )

        with step_2:
            st.success(
                """
                **4. Detect analytical stress**

                Identify observations above the fixed high-demand threshold.
                """
            )

            st.success(
                """
                **5. Simulate electrification**

                Add controlled EV and heat-pump loads during evening periods.
                """
            )

            st.success(
                """
                **6. Explain and communicate**

                Use SHAP and interactive visualisations to explain the results.
                """
            )

        st.caption(
            """
            The dashboard presents the outputs of Phases 1–12 and acts as the
            final decision-support interface for the dissertation project.
            """
        )

    with questions_tab:

        st.subheader("Research questions")

        st.markdown(
            """
            1. **How accurately can machine learning forecast aggregated
               half-hourly electricity demand?**

            2. **When does local demand exceed the analytical stress threshold?**

            3. **How do EV and heat-pump scenarios affect the number and
               percentage of stress periods?**

            4. **How does combined electrification amplify evening peak demand?**

            5. **Which input features have the greatest influence on XGBoost
               predictions?**
            """
        )

        st.success(
            """
            The project connects forecasting, stress analysis, scenario
            modelling and explainability within one integrated research
            workflow.
            """
        )

    with findings_tab:

        st.subheader("Headline findings")

        finding_1, finding_2, finding_3 = st.columns(3)

        with finding_1:
            st.success(
                """
                **Forecasting performance**

                XGBoost achieved:

                - MAE: **5.7080 kWh**
                - RMSE: **7.6328 kWh**
                - R²: **0.9912**
                """
            )

        with finding_2:
            st.warning(
                """
                **Baseline stress**

                The final dataset contained:

                - **1,002** stress events
                - **4.96%** stress rate
                """
            )

        with finding_3:
            st.error(
                """
                **Medium combined scenario**

                Combined EV and heat-pump demand produced:

                - **1,960** stress events
                - **9.70%** stress rate
                """
            )

        st.subheader("Most influential model features")

        feature_1, feature_2, feature_3 = st.columns(3)

        with feature_1:
            st.metric(
                "1st",
                "lag_1",
            )

        with feature_2:
            st.metric(
                "2nd",
                "lag_48",
            )

        with feature_3:
            st.metric(
                "3rd",
                "hour",
            )

        st.write(
            """
            These findings indicate that recent demand, recurring daily
            patterns and time of day are central to the model's forecasting
            behaviour.
            """
        )

    with guide_tab:

        st.subheader("Recommended dashboard journey")

        journey_1, journey_2 = st.columns(2)

        with journey_1:
            st.markdown(
                """
                **Understand the evidence**

                1. Open **Data Overview**
                2. Review **Demand Forecasting**
                3. Examine **Stress Detection**
                4. Inspect **SHAP Explainability**
                """
            )

        with journey_2:
            st.markdown(
                """
                **Explore future pressures**

                1. Test **EV Simulation**
                2. Test **Heat-Pump Simulation**
                3. Compare **Combined EV–Heat-Pump Simulation**
                4. Finish with **Model Performance & Summary**
                """
            )

        st.warning(
            """
            Use the sidebar date selector to investigate a specific period.
            Scenario values are controlled research assumptions and should not
            be interpreted as precise household-level forecasts.
            """
        )

    st.divider()

    # -----------------------------------------------------
    # Intended users and contribution
    # -----------------------------------------------------

    st.subheader("Who could use this type of dashboard?")

    user_1, user_2, user_3 = st.columns(3)

    with user_1:
        st.info(
            """
            **Electricity-network planners**

            Explore local peak-demand risk and compare electrification
            scenarios.
            """
        )

    with user_2:
        st.info(
            """
            **Local authorities**

            Support transport, housing and Net Zero planning with clearer
            demand evidence.
            """
        )

    with user_3:
        st.info(
            """
            **Researchers and analysts**

            Review forecasting, stress detection, simulation and explainability
            in one transparent system.
            """
        )

    st.subheader("Research contribution")

    st.write(
        """
        The principal contribution is an integrated and explainable
        decision-support workflow. The project does not evaluate forecasting
        accuracy in isolation; it links the model to practical questions about
        local high-demand risk, EV adoption, heat-pump adoption and combined
        electrification pressure.

        The dashboard is intended as a transparent research and planning tool.
        It is not a substitute for detailed feeder-level electrical-engineering
        studies.
        """
    )


# =========================================================
# DATA OVERVIEW
# =========================================================

elif selected_page == "Data Overview":

    st.markdown('<div class="drg-section-tag">DATA INTELLIGENCE</div>', unsafe_allow_html=True)

    st.title("📊 Data Overview")

    st.write(
        """
        This page summarises the final model-ready dataset used for
        forecasting and network-stress analysis.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Records",
            f"{len(filtered_df):,}",
        )

    with col2:
        st.metric(
            "Model features",
            len(feature_names),
        )

    with col3:
        st.metric(
            "Missing values",
            int(filtered_df.isna().sum().sum()),
        )

    with col4:
        st.metric(
            "Average demand",
            f"{filtered_df['total_demand_kwh'].mean():.2f} kWh",
        )

    st.subheader("Dataset period")

    period1, period2 = st.columns(2)

    with period1:
        st.info(
            f"Start: {filtered_df['datetime'].min()}"
        )

    with period2:
        st.info(
            f"End: {filtered_df['datetime'].max()}"
        )

    st.subheader("Historical electricity demand")

    historical_figure = px.line(
        filtered_df,
        x="datetime",
        y="total_demand_kwh",
        title="Half-hourly local electricity demand",
        labels={
            "datetime": "Date and time",
            "total_demand_kwh": "Demand (kWh)",
        },
    )

    historical_figure.update_layout(
        hovermode="x unified",
    )

    historical_figure = apply_plotly_theme(historical_figure)

    st.plotly_chart(
        historical_figure,
        use_container_width=True,
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        hourly_profile = (
            filtered_df
            .groupby(
                "hour",
                as_index=False,
            )["total_demand_kwh"]
            .mean()
        )

        hourly_figure = px.line(
            hourly_profile,
            x="hour",
            y="total_demand_kwh",
            markers=True,
            title="Average demand by hour",
            labels={
                "hour": "Hour",
                "total_demand_kwh": "Average demand (kWh)",
            },
        )

        hourly_figure = apply_plotly_theme(hourly_figure)

        st.plotly_chart(
            hourly_figure,
            use_container_width=True,
        )

    with chart2:

        weekday_names = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }

        weekday_profile = (
            filtered_df
            .groupby(
                "weekday",
                as_index=False,
            )["total_demand_kwh"]
            .mean()
        )

        weekday_profile["weekday_name"] = (
            weekday_profile["weekday"]
            .map(weekday_names)
        )

        weekday_figure = px.bar(
            weekday_profile,
            x="weekday_name",
            y="total_demand_kwh",
            title="Average demand by weekday",
            labels={
                "weekday_name": "Weekday",
                "total_demand_kwh": "Average demand (kWh)",
            },
        )

        weekday_figure = apply_plotly_theme(weekday_figure)

        st.plotly_chart(
            weekday_figure,
            use_container_width=True,
        )

    chart3, chart4 = st.columns(2)

    with chart3:

        monthly_profile = (
            filtered_df
            .groupby(
                "month",
                as_index=False,
            )["total_demand_kwh"]
            .mean()
        )

        monthly_figure = px.bar(
            monthly_profile,
            x="month",
            y="total_demand_kwh",
            title="Average demand by month",
            labels={
                "month": "Month",
                "total_demand_kwh": "Average demand (kWh)",
            },
        )

        monthly_figure = apply_plotly_theme(monthly_figure)

        st.plotly_chart(
            monthly_figure,
            use_container_width=True,
        )

    with chart4:

        distribution_figure = px.histogram(
            filtered_df,
            x="total_demand_kwh",
            nbins=50,
            title="Demand distribution",
            labels={
                "total_demand_kwh": "Demand (kWh)",
            },
        )

        distribution_figure = apply_plotly_theme(distribution_figure)

        st.plotly_chart(
            distribution_figure,
            use_container_width=True,
        )

    st.subheader("Descriptive statistics")

    st.dataframe(
        filtered_df["total_demand_kwh"]
        .describe()
        .to_frame("Value"),
        use_container_width=True,
    )

    with st.expander("View dataset preview"):

        st.dataframe(
            filtered_df.head(100),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# DEMAND FORECASTING
# =========================================================

elif selected_page == "Demand Forecasting":

    st.markdown('<div class="drg-section-tag">FORECASTING ENGINE</div>', unsafe_allow_html=True)

    st.title("📈 Demand Forecasting")

    st.write(
        """
        The final 20% of the chronologically ordered observations are
        used as the unseen test period.
        """
    )

    required_columns = (
        feature_names
        + ["total_demand_kwh"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing forecasting columns: {missing_columns}"
        )
        st.stop()

    modelling_df = (
        df.dropna(subset=required_columns)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    split_index = int(
        len(modelling_df) * 0.8
    )

    X_test = modelling_df[
        feature_names
    ].iloc[split_index:]

    y_test = modelling_df[
        "total_demand_kwh"
    ].iloc[split_index:]

    test_dates = modelling_df[
        "datetime"
    ].iloc[split_index:]

    y_pred = xgb_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred,
        )
    )

    mape = (
        mean_absolute_percentage_error(
            y_test,
            y_pred,
        )
        * 100
    )

    r2 = r2_score(
        y_test,
        y_pred,
    )

    results = pd.DataFrame(
        {
            "datetime": test_dates.values,
            "actual_demand_kwh": y_test.values,
            "predicted_demand_kwh": y_pred,
        }
    )

    results["absolute_error_kwh"] = (
        results["actual_demand_kwh"]
        - results["predicted_demand_kwh"]
    ).abs()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "MAE",
            f"{mae:.4f} kWh",
        )

    with col2:
        st.metric(
            "RMSE",
            f"{rmse:.4f} kWh",
        )

    with col3:
        st.metric(
            "MAPE",
            f"{mape:.2f}%",
        )

    with col4:
        st.metric(
            "R²",
            f"{r2:.4f}",
        )

    st.subheader("Actual versus predicted demand")

    comparison = results.melt(
        id_vars="datetime",
        value_vars=[
            "actual_demand_kwh",
            "predicted_demand_kwh",
        ],
        var_name="Series",
        value_name="Demand (kWh)",
    )

    comparison["Series"] = comparison["Series"].replace(
        {
            "actual_demand_kwh": "Actual demand",
            "predicted_demand_kwh": "Predicted demand",
        }
    )

    forecast_figure = px.line(
        comparison,
        x="datetime",
        y="Demand (kWh)",
        color="Series",
        title="Actual and XGBoost-predicted demand",
    )

    forecast_figure.update_layout(
        hovermode="x unified",
    )

    forecast_figure = apply_plotly_theme(forecast_figure)

    st.plotly_chart(
        forecast_figure,
        use_container_width=True,
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        scatter_figure = px.scatter(
            results,
            x="actual_demand_kwh",
            y="predicted_demand_kwh",
            title="Actual versus predicted values",
            labels={
                "actual_demand_kwh": "Actual demand (kWh)",
                "predicted_demand_kwh": "Predicted demand (kWh)",
            },
        )

        scatter_figure = apply_plotly_theme(scatter_figure)

        st.plotly_chart(
            scatter_figure,
            use_container_width=True,
        )

    with chart2:

        error_figure = px.histogram(
            results,
            x="absolute_error_kwh",
            nbins=40,
            title="Absolute-error distribution",
            labels={
                "absolute_error_kwh": "Absolute error (kWh)",
            },
        )

        error_figure = apply_plotly_theme(error_figure)

        st.plotly_chart(
            error_figure,
            use_container_width=True,
        )

    st.subheader("Largest forecasting errors")

    st.dataframe(
        results.nlargest(
            10,
            "absolute_error_kwh",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        """
        XGBoost achieved strong performance during the unseen test
        period. Remaining errors were larger during unusual or rapidly
        changing demand periods.
        """
    )


# =========================================================
# STRESS DETECTION
# =========================================================

elif selected_page == "Stress Detection":

    st.markdown('<div class="drg-section-tag" style="background:#FDEAEA;border-color:#F3B5B5;color:#B73434;">GRID RISK MONITORING</div>', unsafe_allow_html=True)

    st.title("⚠️ Electricity-Network Stress Detection")

    st.write(
        """
        A stress event occurs when half-hourly aggregated demand exceeds
        the analytical threshold. This threshold is a research proxy,
        not an official engineering capacity limit.
        """
    )

    stress_df = filtered_df.copy()

    stress_df["is_stress_event"] = (
        stress_df["total_demand_kwh"]
        > STRESS_THRESHOLD_KWH
    )

    stress_df["exceedance_kwh"] = np.maximum(
        stress_df["total_demand_kwh"]
        - STRESS_THRESHOLD_KWH,
        0,
    )

    stress_events = int(
        stress_df["is_stress_event"].sum()
    )

    stress_percentage = (
        stress_events
        / len(stress_df)
        * 100
    )

    maximum_demand = (
        stress_df["total_demand_kwh"].max()
    )

    maximum_exceedance = (
        stress_df["exceedance_kwh"].max()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Stress threshold",
            f"{STRESS_THRESHOLD_KWH:.4f} kWh",
        )

    with col2:
        st.metric(
            "Stress events",
            f"{stress_events:,}",
        )

    with col3:
        st.metric(
            "Stress percentage",
            f"{stress_percentage:.2f}%",
        )

    with col4:
        st.metric(
            "Maximum demand",
            f"{maximum_demand:.2f} kWh",
        )

    timeline = px.line(
        stress_df,
        x="datetime",
        y="total_demand_kwh",
        title="Demand and analytical stress threshold",
        labels={
            "datetime": "Date and time",
            "total_demand_kwh": "Demand (kWh)",
        },
    )

    timeline.add_hline(
        y=STRESS_THRESHOLD_KWH,
        line_dash="dash",
        annotation_text="Stress threshold",
    )

    stress_points = stress_df[
        stress_df["is_stress_event"]
    ]

    timeline.add_scatter(
        x=stress_points["datetime"],
        y=stress_points["total_demand_kwh"],
        mode="markers",
        name="Stress events",
    )

    timeline = apply_plotly_theme(timeline)

    st.plotly_chart(
        timeline,
        use_container_width=True,
    )

    st.subheader("Stress events by hour")

    hourly_stress = (
        stress_df[
            stress_df["is_stress_event"]
        ]
        .groupby(
            "hour",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "stress_events",
            }
        )
    )

    hourly_stress_figure = px.bar(
        hourly_stress,
        x="hour",
        y="stress_events",
        title="Stress events by hour",
    )

    hourly_stress_figure = apply_plotly_theme(hourly_stress_figure)

    st.plotly_chart(
        hourly_stress_figure,
        use_container_width=True,
    )

    st.warning(
        f"""
        The selected period contained **{stress_events:,} stress
        events**, representing **{stress_percentage:.2f}%** of all
        observations. The maximum exceedance was
        **{maximum_exceedance:.2f} kWh**.
        """
    )


# =========================================================
# EV SIMULATION
# =========================================================

elif selected_page == "EV Simulation":

    st.markdown('<div class="drg-section-tag" style="background:#FFF4DB;border-color:#F1D18B;color:#9A6812;">EV SCENARIO LAB</div>', unsafe_allow_html=True)

    st.title("🚗 Electric-Vehicle Load Simulation")

    st.write(
        """
        This what-if analysis applies additional EV load during the
        residential evening charging period from 18:00 to 21:59.
        """
    )

    selected_scenario = st.selectbox(
        "Select EV scenario",
        list(EV_SCENARIOS.keys()),
        index=2,
    )

    selected_ev_load = EV_SCENARIOS[
        selected_scenario
    ]

    ev_df = filtered_df.copy()

    charging_window = (
        ev_df["datetime"]
        .dt.hour
        .between(
            EV_START_HOUR,
            EV_END_HOUR,
        )
    )

    ev_df["baseline_stress_event"] = (
        ev_df["total_demand_kwh"]
        > STRESS_THRESHOLD_KWH
    )

    baseline_stress_events = int(
        ev_df["baseline_stress_event"].sum()
    )

    baseline_stress_percentage = (
        baseline_stress_events
        / len(ev_df)
        * 100
    )

    ev_df["ev_load_kwh"] = np.where(
        charging_window,
        selected_ev_load,
        0.0,
    )

    ev_df["simulated_demand_kwh"] = (
        ev_df["total_demand_kwh"]
        + ev_df["ev_load_kwh"]
    )

    ev_df["simulated_stress_event"] = (
        ev_df["simulated_demand_kwh"]
        > STRESS_THRESHOLD_KWH
    )

    simulated_stress_events = int(
        ev_df["simulated_stress_event"].sum()
    )

    simulated_stress_percentage = (
        simulated_stress_events
        / len(ev_df)
        * 100
    )

    stress_event_increase = (
        simulated_stress_events
        - baseline_stress_events
    )

    percentage_point_increase = (
        simulated_stress_percentage
        - baseline_stress_percentage
    )

    maximum_simulated_demand = (
        ev_df["simulated_demand_kwh"].max()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Selected EV load",
            f"{selected_ev_load} kWh",
        )

    with col2:
        st.metric(
            "Simulated stress events",
            f"{simulated_stress_events:,}",
            delta=f"{stress_event_increase:+,}",
        )

    with col3:
        st.metric(
            "Stress percentage",
            f"{simulated_stress_percentage:.2f}%",
            delta=f"{percentage_point_increase:+.2f} pp",
        )

    with col4:
        st.metric(
            "Maximum simulated demand",
            f"{maximum_simulated_demand:.2f} kWh",
        )

    scenario_results = []

    for scenario_name, scenario_load in EV_SCENARIOS.items():

        scenario_df = filtered_df.copy()

        scenario_window = (
            scenario_df["datetime"]
            .dt.hour
            .between(
                EV_START_HOUR,
                EV_END_HOUR,
            )
        )

        scenario_df["ev_load_kwh"] = np.where(
            scenario_window,
            scenario_load,
            0.0,
        )

        scenario_df["simulated_demand_kwh"] = (
            scenario_df["total_demand_kwh"]
            + scenario_df["ev_load_kwh"]
        )

        scenario_df["stress_event"] = (
            scenario_df["simulated_demand_kwh"]
            > STRESS_THRESHOLD_KWH
        )

        scenario_stress_events = int(
            scenario_df["stress_event"].sum()
        )

        scenario_stress_percentage = (
            scenario_stress_events
            / len(scenario_df)
            * 100
        )

        scenario_results.append(
            {
                "Scenario": scenario_name,
                "Additional EV load (kWh)": scenario_load,
                "Stress events": scenario_stress_events,
                "Stress percentage": scenario_stress_percentage,
                "Increase from baseline": (
                    scenario_stress_events
                    - baseline_stress_events
                ),
            }
        )

    scenario_results_df = pd.DataFrame(
        scenario_results
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        scenario_events_figure = px.bar(
            scenario_results_df,
            x="Scenario",
            y="Stress events",
            text_auto=True,
            title="Stress events by EV scenario",
        )

        scenario_events_figure = apply_plotly_theme(scenario_events_figure)

        st.plotly_chart(
            scenario_events_figure,
            use_container_width=True,
        )

    with chart2:

        scenario_percentage_figure = px.line(
            scenario_results_df,
            x="Scenario",
            y="Stress percentage",
            markers=True,
            title="Stress percentage by EV scenario",
        )

        scenario_percentage_figure = apply_plotly_theme(scenario_percentage_figure)

        st.plotly_chart(
            scenario_percentage_figure,
            use_container_width=True,
        )

    st.subheader(
        f"Observed versus simulated demand — {selected_scenario}"
    )

    timeline_data = ev_df[
        [
            "datetime",
            "total_demand_kwh",
            "simulated_demand_kwh",
        ]
    ].melt(
        id_vars="datetime",
        var_name="Series",
        value_name="Demand (kWh)",
    )

    timeline_data["Series"] = timeline_data["Series"].replace(
        {
            "total_demand_kwh": "Observed demand",
            "simulated_demand_kwh": "Simulated EV demand",
        }
    )

    ev_timeline = px.line(
        timeline_data,
        x="datetime",
        y="Demand (kWh)",
        color="Series",
        title="Observed and simulated electricity demand",
    )

    ev_timeline.add_hline(
        y=STRESS_THRESHOLD_KWH,
        line_dash="dash",
        annotation_text="Stress threshold",
    )

    ev_timeline = apply_plotly_theme(ev_timeline)

    st.plotly_chart(
        ev_timeline,
        use_container_width=True,
    )

    st.subheader("EV scenario comparison")

    scenario_display = scenario_results_df.copy()

    scenario_display["Stress percentage"] = (
        scenario_display["Stress percentage"]
        .round(4)
    )

    st.dataframe(
        scenario_display,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        f"""
        Under the **{selected_scenario}** scenario, stress events
        increased from **{baseline_stress_events:,}** to
        **{simulated_stress_events:,}**.

        This is a controlled scenario analysis rather than an exact
        prediction of future EV charging behaviour.
        """
    )

    scenario_csv = (
        scenario_results_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download EV scenario results",
        data=scenario_csv,
        file_name="ev_simulation_results.csv",
        mime="text/csv",
    )

# =========================================================
# HEAT-PUMP SIMULATION
# =========================================================

elif selected_page == "Heat-Pump Simulation":

    st.markdown('<div class="drg-section-tag" style="background:#EAF3E3;border-color:#C6DDB6;color:#4F7B36;">HEAT-PUMP SCENARIO LAB</div>', unsafe_allow_html=True)

    st.title("🏠 Heat-Pump Load Simulation")

    st.write(
        """
        This what-if analysis applies additional heat-pump electricity
        demand during the evening heating period from 18:00 to 21:59.

        The page follows the same scenario-based approach used in Phase 10
        of the DRG project and keeps the analytical stress threshold fixed.
        """
    )

    st.divider()

    selected_hp_scenario = st.selectbox(
        "Select heat-pump scenario",
        list(HEAT_PUMP_SCENARIOS.keys()),
        index=2,
    )

    selected_hp_load = HEAT_PUMP_SCENARIOS[
        selected_hp_scenario
    ]

    hp_df = filtered_df.copy()

    heating_window = (
        hp_df["datetime"]
        .dt.hour
        .between(
            HEAT_PUMP_START_HOUR,
            HEAT_PUMP_END_HOUR,
        )
    )

    hp_df["baseline_stress_event"] = (
        hp_df["total_demand_kwh"]
        > STRESS_THRESHOLD_KWH
    )

    baseline_stress_events = int(
        hp_df["baseline_stress_event"].sum()
    )

    baseline_stress_percentage = (
        baseline_stress_events
        / len(hp_df)
        * 100
    )

    hp_df["heat_pump_load_kwh"] = np.where(
        heating_window,
        selected_hp_load,
        0.0,
    )

    hp_df["simulated_demand_kwh"] = (
        hp_df["total_demand_kwh"]
        + hp_df["heat_pump_load_kwh"]
    )

    hp_df["simulated_stress_event"] = (
        hp_df["simulated_demand_kwh"]
        > STRESS_THRESHOLD_KWH
    )

    simulated_stress_events = int(
        hp_df["simulated_stress_event"].sum()
    )

    simulated_stress_percentage = (
        simulated_stress_events
        / len(hp_df)
        * 100
    )

    stress_event_increase = (
        simulated_stress_events
        - baseline_stress_events
    )

    percentage_point_increase = (
        simulated_stress_percentage
        - baseline_stress_percentage
    )

    maximum_simulated_demand = (
        hp_df["simulated_demand_kwh"].max()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Selected heat-pump load",
            f"{selected_hp_load} kWh",
        )

    with col2:
        st.metric(
            "Simulated stress events",
            f"{simulated_stress_events:,}",
            delta=f"{stress_event_increase:+,}",
        )

    with col3:
        st.metric(
            "Stress percentage",
            f"{simulated_stress_percentage:.2f}%",
            delta=f"{percentage_point_increase:+.2f} pp",
        )

    with col4:
        st.metric(
            "Maximum simulated demand",
            f"{maximum_simulated_demand:.2f} kWh",
        )

    st.caption(
        """
        Heat-pump load is added only between 18:00 and 21:59.
        The analytical stress threshold remains fixed at 354.5756 kWh.
        """
    )

    st.divider()

    hp_results = []

    for scenario_name, scenario_load in (
        HEAT_PUMP_SCENARIOS.items()
    ):

        scenario_df = filtered_df.copy()

        scenario_window = (
            scenario_df["datetime"]
            .dt.hour
            .between(
                HEAT_PUMP_START_HOUR,
                HEAT_PUMP_END_HOUR,
            )
        )

        scenario_df["heat_pump_load_kwh"] = np.where(
            scenario_window,
            scenario_load,
            0.0,
        )

        scenario_df["simulated_demand_kwh"] = (
            scenario_df["total_demand_kwh"]
            + scenario_df["heat_pump_load_kwh"]
        )

        scenario_df["stress_event"] = (
            scenario_df["simulated_demand_kwh"]
            > STRESS_THRESHOLD_KWH
        )

        scenario_stress_events = int(
            scenario_df["stress_event"].sum()
        )

        scenario_stress_percentage = (
            scenario_stress_events
            / len(scenario_df)
            * 100
        )

        hp_results.append(
            {
                "Scenario": scenario_name,
                "Additional heat-pump load (kWh)": scenario_load,
                "Stress events": scenario_stress_events,
                "Stress percentage": scenario_stress_percentage,
                "Increase from baseline": (
                    scenario_stress_events
                    - baseline_stress_events
                ),
            }
        )

    hp_results_df = pd.DataFrame(
        hp_results
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        hp_events_figure = px.bar(
            hp_results_df,
            x="Scenario",
            y="Stress events",
            text_auto=True,
            title="Stress events by heat-pump scenario",
        )

        hp_events_figure = apply_plotly_theme(hp_events_figure)

        st.plotly_chart(
            hp_events_figure,
            use_container_width=True,
        )

    with chart2:

        hp_percentage_figure = px.line(
            hp_results_df,
            x="Scenario",
            y="Stress percentage",
            markers=True,
            title="Stress percentage by heat-pump scenario",
        )

        hp_percentage_figure = apply_plotly_theme(hp_percentage_figure)

        st.plotly_chart(
            hp_percentage_figure,
            use_container_width=True,
        )

    st.subheader(
        f"Observed versus simulated demand — {selected_hp_scenario}"
    )

    timeline_data = hp_df[
        [
            "datetime",
            "total_demand_kwh",
            "simulated_demand_kwh",
        ]
    ].melt(
        id_vars="datetime",
        var_name="Series",
        value_name="Demand (kWh)",
    )

    timeline_data["Series"] = timeline_data["Series"].replace(
        {
            "total_demand_kwh": "Observed demand",
            "simulated_demand_kwh": "Simulated heat-pump demand",
        }
    )

    hp_timeline = px.line(
        timeline_data,
        x="datetime",
        y="Demand (kWh)",
        color="Series",
        title="Observed and simulated heat-pump demand",
    )

    hp_timeline.add_hline(
        y=STRESS_THRESHOLD_KWH,
        line_dash="dash",
        annotation_text="Stress threshold",
    )

    hp_timeline.update_layout(
        hovermode="x unified",
    )

    hp_timeline = apply_plotly_theme(hp_timeline)

    st.plotly_chart(
        hp_timeline,
        use_container_width=True,
    )

    st.subheader("Heat-pump scenario comparison")

    hp_display = hp_results_df.copy()

    hp_display["Stress percentage"] = (
        hp_display["Stress percentage"]
        .round(4)
    )

    st.dataframe(
        hp_display,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        f"Highest-demand periods — {selected_hp_scenario}"
    )

    highest_hp_periods = (
        hp_df
        .nlargest(
            10,
            "simulated_demand_kwh",
        )
        [
            [
                "datetime",
                "total_demand_kwh",
                "heat_pump_load_kwh",
                "simulated_demand_kwh",
                "simulated_stress_event",
            ]
        ]
        .rename(
            columns={
                "datetime": "Date and time",
                "total_demand_kwh": "Observed demand (kWh)",
                "heat_pump_load_kwh": "Heat-pump load (kWh)",
                "simulated_demand_kwh": "Simulated demand (kWh)",
                "simulated_stress_event": "Stress event",
            }
        )
    )

    st.dataframe(
        highest_hp_periods,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Heat-pump simulation interpretation")

    if selected_hp_load == 0:

        st.info(
            """
            The Baseline scenario applies no additional heat-pump
            electricity demand and provides the reference case.
            """
        )

    else:

        st.warning(
            f"""
            Under the **{selected_hp_scenario}** scenario, stress events
            increased from **{baseline_stress_events:,}** to
            **{simulated_stress_events:,}**.

            The selected scenario applies an additional
            **{selected_hp_load} kWh** during the evening heating period.
            """
        )

    st.info(
        """
        This is a controlled what-if analysis rather than a precise
        prediction of household heat-pump demand. It does not model
        property size, insulation, outdoor temperature, coefficient of
        performance, or occupant heating behaviour.
        """
    )

    hp_results_csv = (
        hp_results_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download heat-pump scenario results",
        data=hp_results_csv,
        file_name="heat_pump_simulation_results.csv",
        mime="text/csv",
    )

# =========================================================
# COMBINED EV–HEAT-PUMP SIMULATION
# =========================================================

elif selected_page == "Combined EV–Heat-Pump Simulation":

    st.markdown('<div class="drg-section-tag" style="background:#EAF3E3;border-color:#C6DDB6;color:#4F7B36;">COMBINED ELECTRIFICATION</div>', unsafe_allow_html=True)

    st.title("⚡ Combined EV–Heat-Pump Simulation")

    st.write(
        """
        This page evaluates the combined effect of electric-vehicle
        charging and heat-pump electricity demand on local
        electricity-network stress.

        The analysis follows the Phase 11 scenario structure. EV and
        heat-pump loads are applied together during the evening period
        from 18:00 to 21:59 while the analytical stress threshold
        remains fixed.
        """
    )

    st.divider()

    selected_combined_scenario = st.selectbox(
        "Select combined electrification scenario",
        options=list(COMBINED_SCENARIOS.keys()),
        index=2,
    )

    selected_combined_values = COMBINED_SCENARIOS[
        selected_combined_scenario
    ]

    selected_ev_load = selected_combined_values[
        "ev_load_kwh"
    ]

    selected_hp_load = selected_combined_values[
        "heat_pump_load_kwh"
    ]

    selected_total_load = (
        selected_ev_load
        + selected_hp_load
    )

    combined_df = filtered_df.copy()

    evening_window = (
        combined_df["datetime"]
        .dt.hour
        .between(
            COMBINED_START_HOUR,
            COMBINED_END_HOUR,
        )
    )

    combined_df["baseline_stress_event"] = (
        combined_df["total_demand_kwh"]
        > STRESS_THRESHOLD_KWH
    )

    baseline_stress_events = int(
        combined_df["baseline_stress_event"].sum()
    )

    baseline_stress_percentage = (
        baseline_stress_events
        / len(combined_df)
        * 100
    )

    combined_df["ev_load_kwh"] = np.where(
        evening_window,
        selected_ev_load,
        0.0,
    )

    combined_df["heat_pump_load_kwh"] = np.where(
        evening_window,
        selected_hp_load,
        0.0,
    )

    combined_df["combined_additional_load_kwh"] = (
        combined_df["ev_load_kwh"]
        + combined_df["heat_pump_load_kwh"]
    )

    combined_df["simulated_demand_kwh"] = (
        combined_df["total_demand_kwh"]
        + combined_df["combined_additional_load_kwh"]
    )

    combined_df["simulated_stress_event"] = (
        combined_df["simulated_demand_kwh"]
        > STRESS_THRESHOLD_KWH
    )

    simulated_stress_events = int(
        combined_df["simulated_stress_event"].sum()
    )

    simulated_stress_percentage = (
        simulated_stress_events
        / len(combined_df)
        * 100
    )

    stress_event_increase = (
        simulated_stress_events
        - baseline_stress_events
    )

    percentage_point_increase = (
        simulated_stress_percentage
        - baseline_stress_percentage
    )

    maximum_simulated_demand = (
        combined_df["simulated_demand_kwh"].max()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Combined additional load",
            f"{selected_total_load} kWh",
        )

    with col2:
        st.metric(
            "Simulated stress events",
            f"{simulated_stress_events:,}",
            delta=f"{stress_event_increase:+,}",
        )

    with col3:
        st.metric(
            "Stress percentage",
            f"{simulated_stress_percentage:.2f}%",
            delta=f"{percentage_point_increase:+.2f} pp",
        )

    with col4:
        st.metric(
            "Maximum simulated demand",
            f"{maximum_simulated_demand:.2f} kWh",
        )

    load_col1, load_col2 = st.columns(2)

    with load_col1:
        st.info(
            f"Selected EV load: **{selected_ev_load} kWh**"
        )

    with load_col2:
        st.info(
            f"Selected heat-pump load: **{selected_hp_load} kWh**"
        )

    st.caption(
        """
        Both additional loads are applied only from 18:00 to 21:59.
        The analytical stress threshold remains fixed at 354.5756 kWh.
        """
    )

    st.divider()

    combined_results = []

    for scenario_name, scenario_values in (
        COMBINED_SCENARIOS.items()
    ):

        scenario_df = filtered_df.copy()

        scenario_window = (
            scenario_df["datetime"]
            .dt.hour
            .between(
                COMBINED_START_HOUR,
                COMBINED_END_HOUR,
            )
        )

        scenario_ev_load = scenario_values[
            "ev_load_kwh"
        ]

        scenario_hp_load = scenario_values[
            "heat_pump_load_kwh"
        ]

        scenario_df["ev_load_kwh"] = np.where(
            scenario_window,
            scenario_ev_load,
            0.0,
        )

        scenario_df["heat_pump_load_kwh"] = np.where(
            scenario_window,
            scenario_hp_load,
            0.0,
        )

        scenario_df["combined_additional_load_kwh"] = (
            scenario_df["ev_load_kwh"]
            + scenario_df["heat_pump_load_kwh"]
        )

        scenario_df["simulated_demand_kwh"] = (
            scenario_df["total_demand_kwh"]
            + scenario_df["combined_additional_load_kwh"]
        )

        scenario_df["stress_event"] = (
            scenario_df["simulated_demand_kwh"]
            > STRESS_THRESHOLD_KWH
        )

        scenario_stress_events = int(
            scenario_df["stress_event"].sum()
        )

        scenario_stress_percentage = (
            scenario_stress_events
            / len(scenario_df)
            * 100
        )

        combined_results.append(
            {
                "Scenario": scenario_name,
                "EV load (kWh)": scenario_ev_load,
                "Heat-pump load (kWh)": scenario_hp_load,
                "Combined load (kWh)": (
                    scenario_ev_load
                    + scenario_hp_load
                ),
                "Stress events": scenario_stress_events,
                "Stress percentage": scenario_stress_percentage,
                "Increase from baseline": (
                    scenario_stress_events
                    - baseline_stress_events
                ),
            }
        )

    combined_results_df = pd.DataFrame(
        combined_results
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        combined_events_figure = px.bar(
            combined_results_df,
            x="Scenario",
            y="Stress events",
            text_auto=True,
            title=(
                "Stress events by combined electrification scenario"
            ),
        )

        combined_events_figure.add_hline(
            y=baseline_stress_events,
            line_dash="dash",
            annotation_text=(
                f"Baseline: {baseline_stress_events:,}"
            ),
        )

        combined_events_figure = apply_plotly_theme(combined_events_figure)

        st.plotly_chart(
            combined_events_figure,
            use_container_width=True,
        )

    with chart2:

        combined_percentage_figure = px.line(
            combined_results_df,
            x="Scenario",
            y="Stress percentage",
            markers=True,
            title=(
                "Stress percentage by combined scenario"
            ),
        )

        combined_percentage_figure.add_hline(
            y=baseline_stress_percentage,
            line_dash="dash",
            annotation_text=(
                f"Baseline: {baseline_stress_percentage:.2f}%"
            ),
        )

        combined_percentage_figure = apply_plotly_theme(combined_percentage_figure)

        st.plotly_chart(
            combined_percentage_figure,
            use_container_width=True,
        )

    st.subheader(
        f"Observed versus combined simulated demand — "
        f"{selected_combined_scenario}"
    )

    combined_timeline_data = combined_df[
        [
            "datetime",
            "total_demand_kwh",
            "simulated_demand_kwh",
        ]
    ].melt(
        id_vars="datetime",
        value_vars=[
            "total_demand_kwh",
            "simulated_demand_kwh",
        ],
        var_name="Series",
        value_name="Demand (kWh)",
    )

    combined_timeline_data["Series"] = (
        combined_timeline_data["Series"]
        .replace(
            {
                "total_demand_kwh": "Observed demand",
                "simulated_demand_kwh": (
                    "Combined simulated demand"
                ),
            }
        )
    )

    combined_timeline = px.line(
        combined_timeline_data,
        x="datetime",
        y="Demand (kWh)",
        color="Series",
        title=(
            "Observed and combined EV–heat-pump demand"
        ),
    )

    combined_timeline.add_hline(
        y=STRESS_THRESHOLD_KWH,
        line_dash="dash",
        annotation_text="Stress threshold",
    )

    combined_timeline.update_layout(
        hovermode="x unified",
    )

    combined_timeline = apply_plotly_theme(combined_timeline)

    st.plotly_chart(
        combined_timeline,
        use_container_width=True,
    )

    st.subheader("Combined scenario comparison")

    combined_display = combined_results_df.copy()

    combined_display["Stress percentage"] = (
        combined_display["Stress percentage"].round(4)
    )

    st.dataframe(
        combined_display,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        f"Highest-demand periods — "
        f"{selected_combined_scenario}"
    )

    highest_combined_periods = (
        combined_df
        .nlargest(
            10,
            "simulated_demand_kwh",
        )
        [
            [
                "datetime",
                "total_demand_kwh",
                "ev_load_kwh",
                "heat_pump_load_kwh",
                "combined_additional_load_kwh",
                "simulated_demand_kwh",
                "simulated_stress_event",
            ]
        ]
        .rename(
            columns={
                "datetime": "Date and time",
                "total_demand_kwh": "Observed demand (kWh)",
                "ev_load_kwh": "EV load (kWh)",
                "heat_pump_load_kwh": "Heat-pump load (kWh)",
                "combined_additional_load_kwh": (
                    "Combined additional load (kWh)"
                ),
                "simulated_demand_kwh": (
                    "Simulated demand (kWh)"
                ),
                "simulated_stress_event": "Stress event",
            }
        )
    )

    st.dataframe(
        highest_combined_periods,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Combined electrification interpretation"
    )

    if selected_total_load == 0:

        st.info(
            """
            The Baseline scenario applies no additional EV or
            heat-pump demand and provides the reference case.
            """
        )

    else:

        st.warning(
            f"""
            Under the **{selected_combined_scenario}** scenario,
            stress events increased from
            **{baseline_stress_events:,}** to
            **{simulated_stress_events:,}**.

            The scenario applies **{selected_ev_load} kWh** of EV
            load and **{selected_hp_load} kWh** of heat-pump load,
            producing a combined evening increase of
            **{selected_total_load} kWh**.
            """
        )

    st.info(
        """
        This is a controlled what-if analysis. It assumes fixed,
        simultaneous evening loads and does not represent diversity
        in charging times, charger ratings, heat-pump efficiency,
        weather, building characteristics or occupant behaviour.
        """
    )

    combined_results_csv = (
        combined_results_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    selected_combined_csv = (
        combined_df[
            [
                "datetime",
                "total_demand_kwh",
                "ev_load_kwh",
                "heat_pump_load_kwh",
                "combined_additional_load_kwh",
                "simulated_demand_kwh",
                "simulated_stress_event",
            ]
        ]
        .to_csv(index=False)
        .encode("utf-8")
    )

    download1, download2 = st.columns(2)

    with download1:
        st.download_button(
            "Download combined scenario results",
            data=combined_results_csv,
            file_name=(
                "combined_ev_heat_pump_results.csv"
            ),
            mime="text/csv",
        )

    with download2:
        st.download_button(
            f"Download {selected_combined_scenario} timeline",
            data=selected_combined_csv,
            file_name=(
                selected_combined_scenario
                .lower()
                .replace(" ", "_")
                .replace("–", "-")
                + "_timeline.csv"
            ),
            mime="text/csv",
        )

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

elif selected_page == "SHAP Explainability":

    st.markdown('<div class="drg-section-tag">EXPLAINABLE AI</div>', unsafe_allow_html=True)

    st.title("🧠 SHAP Model Explainability")

    st.write(
        """
        This page explains how the final XGBoost model uses each input
        feature when forecasting half-hourly electricity demand.

        SHAP values estimate how strongly each feature pushes an
        individual prediction above or below the model's baseline output.
        Global importance is calculated as the mean absolute SHAP value
        across a representative sample of the unseen test period.
        """
    )

    st.divider()

    required_columns = feature_names + ["total_demand_kwh"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing columns required for SHAP analysis: {missing_columns}"
        )
        st.stop()

    shap_modelling_df = (
        df.dropna(subset=required_columns)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    shap_split_index = int(len(shap_modelling_df) * 0.8)

    shap_test_df = (
        shap_modelling_df
        .iloc[shap_split_index:]
        .reset_index(drop=True)
    )

    # Use a representative sample to keep the dashboard responsive.
    maximum_shap_rows = 1000

    if len(shap_test_df) > maximum_shap_rows:
        shap_sample_positions = np.linspace(
            0,
            len(shap_test_df) - 1,
            maximum_shap_rows,
            dtype=int,
        )

        shap_sample_df = (
            shap_test_df
            .iloc[shap_sample_positions]
            .reset_index(drop=True)
        )
    else:
        shap_sample_df = shap_test_df.copy()

    shap_sample_features = shap_sample_df[feature_names].copy()

    booster = xgb_model.get_booster()

    shap_matrix = DMatrix(
        shap_sample_features,
        feature_names=feature_names,
    )

    shap_contributions = booster.predict(
        shap_matrix,
        pred_contribs=True,
    )

    # The final column is the model's expected value (bias term).
    shap_values = shap_contributions[:, :-1]
    shap_base_values = shap_contributions[:, -1]

    mean_absolute_shap = np.abs(shap_values).mean(axis=0)

    shap_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean absolute SHAP value": mean_absolute_shap,
        }
    ).sort_values(
        "Mean absolute SHAP value",
        ascending=False,
    ).reset_index(drop=True)

    top_three_features = shap_importance.head(3)["Feature"].tolist()

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric(
            "Most influential feature",
            top_three_features[0] if top_three_features else "N/A",
        )

    with metric_2:
        st.metric(
            "Second feature",
            top_three_features[1] if len(top_three_features) > 1 else "N/A",
        )

    with metric_3:
        st.metric(
            "Third feature",
            top_three_features[2] if len(top_three_features) > 2 else "N/A",
        )

    with metric_4:
        st.metric(
            "Explained test rows",
            f"{len(shap_sample_df):,}",
        )

    st.caption(
        "The ranking is recalculated directly from the saved XGBoost model "
        "and the current model-ready dataset."
    )

    st.divider()

    st.subheader("Global SHAP feature importance")

    shap_bar_figure = px.bar(
        shap_importance.sort_values(
            "Mean absolute SHAP value",
            ascending=True,
        ),
        x="Mean absolute SHAP value",
        y="Feature",
        orientation="h",
        title="Mean absolute SHAP importance across the unseen test sample",
        labels={
            "Mean absolute SHAP value": "Mean |SHAP value| (kWh contribution)",
            "Feature": "Model feature",
        },
    )

    shap_bar_figure = apply_plotly_theme(shap_bar_figure)

    st.plotly_chart(
        shap_bar_figure,
        use_container_width=True,
    )

    st.subheader("SHAP importance table")

    shap_display = shap_importance.copy()
    shap_display["Mean absolute SHAP value"] = (
        shap_display["Mean absolute SHAP value"].round(4)
    )
    shap_display.insert(0, "Rank", range(1, len(shap_display) + 1))

    st.dataframe(
        shap_display,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Local explanation for one test observation")

    selected_local_position = st.slider(
        "Select an observation from the SHAP test sample",
        min_value=0,
        max_value=len(shap_sample_df) - 1,
        value=0,
        step=1,
    )

    selected_row = shap_sample_df.iloc[selected_local_position].copy()
    selected_shap_values = shap_values[selected_local_position]
    selected_base_value = float(shap_base_values[selected_local_position])

    # Convert the selected model inputs to a strictly numeric frame.
    # A mixed pandas Series can otherwise turn all feature columns into
    # object dtype when transposed, which XGBoost rejects.
    selected_numeric_values = pd.to_numeric(
        selected_row[feature_names],
        errors="raise",
    ).astype(float)

    selected_feature_input = pd.DataFrame(
        [selected_numeric_values.to_numpy(dtype=float)],
        columns=feature_names,
    )

    selected_feature_frame = pd.DataFrame(
        {
            "Feature": feature_names,
            "Feature value": selected_feature_input.iloc[0].to_numpy(),
            "SHAP contribution": selected_shap_values,
        }
    )

    selected_feature_frame["Absolute contribution"] = (
        selected_feature_frame["SHAP contribution"].abs()
    )

    selected_feature_frame = selected_feature_frame.sort_values(
        "Absolute contribution",
        ascending=False,
    ).reset_index(drop=True)

    selected_prediction = float(
        xgb_model.predict(selected_feature_input)[0]
    )

    local_col_1, local_col_2, local_col_3, local_col_4 = st.columns(4)

    with local_col_1:
        st.metric(
            "Observation time",
            str(selected_row["datetime"]),
        )

    with local_col_2:
        st.metric(
            "Actual demand",
            f"{selected_row['total_demand_kwh']:.2f} kWh",
        )

    with local_col_3:
        st.metric(
            "Predicted demand",
            f"{selected_prediction:.2f} kWh",
        )

    with local_col_4:
        st.metric(
            "Model baseline",
            f"{selected_base_value:.2f} kWh",
        )

    local_shap_figure = px.bar(
        selected_feature_frame.sort_values(
            "SHAP contribution",
            ascending=True,
        ),
        x="SHAP contribution",
        y="Feature",
        orientation="h",
        title="Feature contributions for the selected demand forecast",
        labels={
            "SHAP contribution": "Contribution to prediction (kWh)",
            "Feature": "Model feature",
        },
        hover_data={
            "Feature value": True,
            "Absolute contribution": False,
        },
    )

    local_shap_figure.add_vline(
        x=0,
        line_dash="dash",
    )

    local_shap_figure = apply_plotly_theme(local_shap_figure)

    st.plotly_chart(
        local_shap_figure,
        use_container_width=True,
    )

    st.dataframe(
        selected_feature_frame[
            [
                "Feature",
                "Feature value",
                "SHAP contribution",
            ]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Explainability interpretation")

    st.success(
        f"""
        The most influential features in this model are
        **{', '.join(top_three_features)}**. This indicates that the model
        relies primarily on recent and recurring electricity-demand
        patterns when generating half-hourly forecasts.

        Positive SHAP contributions increase the predicted demand relative
        to the model baseline, while negative contributions reduce it.
        """
    )

    st.info(
        """
        SHAP explains the behaviour of the trained model; it does not prove
        that a feature causes electricity demand to change. The explanations
        are also conditional on the available features, the selected data
        period and the final XGBoost model.
        """
    )

    shap_csv = (
        shap_importance
        .to_csv(index=False)
        .encode("utf-8")
    )

    local_shap_csv = (
        selected_feature_frame[
            [
                "Feature",
                "Feature value",
                "SHAP contribution",
            ]
        ]
        .to_csv(index=False)
        .encode("utf-8")
    )

    download_1, download_2 = st.columns(2)

    with download_1:
        st.download_button(
            "Download global SHAP importance",
            data=shap_csv,
            file_name="shap_global_importance.csv",
            mime="text/csv",
        )

    with download_2:
        st.download_button(
            "Download selected local explanation",
            data=local_shap_csv,
            file_name="shap_local_explanation.csv",
            mime="text/csv",
        )


# =========================================================
# MODEL PERFORMANCE AND PROJECT SUMMARY
# =========================================================

elif selected_page == "Model Performance & Summary":

    st.markdown('<div class="drg-section-tag">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)

    st.title("📑 Model Performance and Project Summary")

    st.write(
        """
        This page consolidates the final forecasting, network-stress and
        electrification-scenario results from the Dynamic Resilient Grid
        project. It provides a concise comparison of the baseline model,
        the final XGBoost model and the principal scenario findings.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Recalculate final XGBoost test performance
    # -----------------------------------------------------

    required_columns = feature_names + ["total_demand_kwh"]

    modelling_df = (
        df.dropna(subset=required_columns)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    split_index = int(len(modelling_df) * 0.8)

    X_test = modelling_df[
        feature_names
    ].iloc[split_index:].copy()

    y_test = modelling_df[
        "total_demand_kwh"
    ].iloc[split_index:].copy()

    y_pred = xgb_model.predict(X_test)

    final_xgb_mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    final_xgb_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred,
        )
    )

    final_xgb_mape = (
        mean_absolute_percentage_error(
            y_test,
            y_pred,
        )
        * 100
    )

    final_xgb_r2 = r2_score(
        y_test,
        y_pred,
    )

    # Linear Regression baseline values from Phase 6.
    linear_mae = 10.7455
    linear_rmse = 13.7777
    linear_r2 = 0.9713

    # -----------------------------------------------------
    # Headline project indicators
    # -----------------------------------------------------

    baseline_stress_events = int(
        (
            df["total_demand_kwh"]
            > STRESS_THRESHOLD_KWH
        ).sum()
    )

    baseline_stress_percentage = (
        baseline_stress_events
        / len(df)
        * 100
    )

    headline_1, headline_2, headline_3, headline_4 = (
        st.columns(4)
    )

    with headline_1:
        st.metric(
            "Final forecasting model",
            "XGBoost",
        )

    with headline_2:
        st.metric(
            "Final R²",
            f"{final_xgb_r2:.4f}",
        )

    with headline_3:
        st.metric(
            "Baseline stress events",
            f"{baseline_stress_events:,}",
        )

    with headline_4:
        st.metric(
            "Baseline stress rate",
            f"{baseline_stress_percentage:.2f}%",
        )

    st.divider()

    # -----------------------------------------------------
    # Model comparison
    # -----------------------------------------------------

    st.subheader("Linear Regression versus XGBoost")

    model_comparison = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "XGBoost",
            ],
            "MAE (kWh)": [
                linear_mae,
                final_xgb_mae,
            ],
            "RMSE (kWh)": [
                linear_rmse,
                final_xgb_rmse,
            ],
            "R²": [
                linear_r2,
                final_xgb_r2,
            ],
        }
    )

    model_comparison["MAE improvement (%)"] = [
        0.0,
        (
            (linear_mae - final_xgb_mae)
            / linear_mae
            * 100
        ),
    ]

    model_comparison["RMSE improvement (%)"] = [
        0.0,
        (
            (linear_rmse - final_xgb_rmse)
            / linear_rmse
            * 100
        ),
    ]

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = (
        st.columns(4)
    )

    with metric_col_1:
        st.metric(
            "XGBoost MAE",
            f"{final_xgb_mae:.4f} kWh",
            delta=(
                f"{linear_mae - final_xgb_mae:.4f} kWh lower"
            ),
            delta_color="normal",
        )

    with metric_col_2:
        st.metric(
            "XGBoost RMSE",
            f"{final_xgb_rmse:.4f} kWh",
            delta=(
                f"{linear_rmse - final_xgb_rmse:.4f} kWh lower"
            ),
            delta_color="normal",
        )

    with metric_col_3:
        st.metric(
            "XGBoost MAPE",
            f"{final_xgb_mape:.2f}%",
        )

    with metric_col_4:
        st.metric(
            "R² improvement",
            f"{final_xgb_r2 - linear_r2:+.4f}",
        )

    comparison_long = model_comparison.melt(
        id_vars="Model",
        value_vars=[
            "MAE (kWh)",
            "RMSE (kWh)",
        ],
        var_name="Metric",
        value_name="Error (kWh)",
    )

    comparison_chart = px.bar(
        comparison_long,
        x="Metric",
        y="Error (kWh)",
        color="Model",
        barmode="group",
        text_auto=".2f",
        title="Forecasting-error comparison",
    )

    comparison_chart = apply_plotly_theme(comparison_chart)

    st.plotly_chart(
        comparison_chart,
        use_container_width=True,
    )

    st.dataframe(
        model_comparison.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        f"""
        XGBoost reduced MAE by approximately
        **{((linear_mae - final_xgb_mae) / linear_mae * 100):.2f}%**
        and RMSE by approximately
        **{((linear_rmse - final_xgb_rmse) / linear_rmse * 100):.2f}%**
        relative to the Linear Regression baseline. This supports the
        selection of XGBoost as the final forecasting model.
        """
    )

    # -----------------------------------------------------
    # Scenario summary
    # -----------------------------------------------------

    st.subheader("Electrification scenario summary")

    evening_window = (
        df["datetime"]
        .dt.hour
        .between(18, 21)
    )

    combined_scenarios = {
        "Baseline": {
            "EV load (kWh)": 0,
            "Heat-pump load (kWh)": 0,
        },
        "Low": {
            "EV load (kWh)": 20,
            "Heat-pump load (kWh)": 10,
        },
        "Medium": {
            "EV load (kWh)": 40,
            "Heat-pump load (kWh)": 20,
        },
        "High": {
            "EV load (kWh)": 60,
            "Heat-pump load (kWh)": 30,
        },
        "Extreme": {
            "EV load (kWh)": 80,
            "Heat-pump load (kWh)": 40,
        },
    }

    scenario_summary_rows = []

    for scenario_name, values in combined_scenarios.items():

        combined_load = (
            values["EV load (kWh)"]
            + values["Heat-pump load (kWh)"]
        )

        simulated_demand = (
            df["total_demand_kwh"]
            + np.where(
                evening_window,
                combined_load,
                0.0,
            )
        )

        scenario_stress_events = int(
            (
                simulated_demand
                > STRESS_THRESHOLD_KWH
            ).sum()
        )

        scenario_stress_percentage = (
            scenario_stress_events
            / len(df)
            * 100
        )

        scenario_summary_rows.append(
            {
                "Scenario": scenario_name,
                "EV load (kWh)": values["EV load (kWh)"],
                "Heat-pump load (kWh)": (
                    values["Heat-pump load (kWh)"]
                ),
                "Combined load (kWh)": combined_load,
                "Stress events": scenario_stress_events,
                "Stress percentage": (
                    scenario_stress_percentage
                ),
                "Increase from baseline": (
                    scenario_stress_events
                    - baseline_stress_events
                ),
                "Maximum demand (kWh)": (
                    simulated_demand.max()
                ),
            }
        )

    scenario_summary = pd.DataFrame(
        scenario_summary_rows
    )

    scenario_chart_col_1, scenario_chart_col_2 = (
        st.columns(2)
    )

    with scenario_chart_col_1:

        scenario_events_chart = px.bar(
            scenario_summary,
            x="Scenario",
            y="Stress events",
            text_auto=True,
            title=(
                "Stress events across combined electrification "
                "scenarios"
            ),
        )

        scenario_events_chart.add_hline(
            y=baseline_stress_events,
            line_dash="dash",
            annotation_text=(
                f"Baseline: {baseline_stress_events:,}"
            ),
        )

        scenario_events_chart = apply_plotly_theme(scenario_events_chart)

        st.plotly_chart(
            scenario_events_chart,
            use_container_width=True,
        )

    with scenario_chart_col_2:

        scenario_rate_chart = px.line(
            scenario_summary,
            x="Scenario",
            y="Stress percentage",
            markers=True,
            title=(
                "Stress percentage across combined scenarios"
            ),
        )

        scenario_rate_chart.add_hline(
            y=baseline_stress_percentage,
            line_dash="dash",
            annotation_text=(
                f"Baseline: "
                f"{baseline_stress_percentage:.2f}%"
            ),
        )

        scenario_rate_chart = apply_plotly_theme(scenario_rate_chart)

        st.plotly_chart(
            scenario_rate_chart,
            use_container_width=True,
        )

    scenario_display = scenario_summary.copy()

    scenario_display["Stress percentage"] = (
        scenario_display["Stress percentage"]
        .round(4)
    )

    scenario_display["Maximum demand (kWh)"] = (
        scenario_display["Maximum demand (kWh)"]
        .round(4)
    )

    st.dataframe(
        scenario_display,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # Principal findings
    # -----------------------------------------------------

    st.subheader("Principal project findings")

    finding_1, finding_2, finding_3 = st.columns(3)

    with finding_1:
        st.success(
            f"""
            **Forecasting performance**

            The final XGBoost model achieved an MAE of
            **{final_xgb_mae:.4f} kWh**, an RMSE of
            **{final_xgb_rmse:.4f} kWh** and an R² of
            **{final_xgb_r2:.4f}**.
            """
        )

    with finding_2:
        st.warning(
            f"""
            **Baseline network stress**

            The analytical threshold identified
            **{baseline_stress_events:,} stress events**, equivalent
            to **{baseline_stress_percentage:.2f}%** of the final
            model-ready observations.
            """
        )

    with finding_3:

        medium_row = scenario_summary[
            scenario_summary["Scenario"] == "Medium"
        ].iloc[0]

        st.error(
            f"""
            **Combined electrification**

            The Medium combined scenario increased stress events to
            **{int(medium_row['Stress events']):,}**, representing
            **{medium_row['Stress percentage']:.2f}%** of observations.
            """
        )

    # -----------------------------------------------------
    # Research contribution
    # -----------------------------------------------------

    st.subheader("Research contribution")

    st.write(
        """
        The project contributes an integrated decision-support workflow
        that combines half-hourly electricity-demand forecasting,
        analytical stress detection, electrification scenario analysis
        and explainable machine learning.

        Rather than assessing forecasting accuracy in isolation, the
        framework connects model outputs with practical questions about
        how EV and heat-pump adoption may alter the frequency of high-load
        periods in an urban electricity-demand profile.
        """
    )

    # -----------------------------------------------------
    # Limitations
    # -----------------------------------------------------

    st.subheader("Key limitations")

    st.markdown(
        """
        - The Low Carbon London observations are historical and may not
          fully represent current household technologies or behaviour.
        - The stress threshold is an analytical proxy rather than a
          feeder-specific engineering capacity rating.
        - EV and heat-pump simulations apply fixed evening loads and do
          not model household-level diversity.
        - Weather variables, local generation, voltage, transformer
          loading and network topology are not included.
        - Strong predictive performance does not establish causal
          relationships between the input features and demand.
        - The dashboard results depend on the final processed dataset,
          saved model and feature definitions used in this implementation.
        """
    )

    # -----------------------------------------------------
    # Conclusion
    # -----------------------------------------------------

    st.subheader("Final conclusion")

    st.info(
        """
        The DRG project demonstrates that XGBoost can forecast aggregated
        half-hourly urban electricity demand with high accuracy using
        recent and recurring demand features. The scenario analysis shows
        that increasing EV and heat-pump loads can materially increase the
        frequency of periods above the analytical stress threshold,
        particularly when both technologies contribute simultaneously
        during evening peak periods.

        The dashboard should therefore be interpreted as a transparent
        research and planning tool for exploring relative risk under
        controlled scenarios, rather than as a substitute for detailed
        electrical-network engineering studies.
        """
    )

    # -----------------------------------------------------
    # Downloads
    # -----------------------------------------------------

    model_summary_csv = (
        model_comparison
        .to_csv(index=False)
        .encode("utf-8")
    )

    scenario_summary_csv = (
        scenario_summary
        .to_csv(index=False)
        .encode("utf-8")
    )

    download_col_1, download_col_2 = st.columns(2)

    with download_col_1:
        st.download_button(
            "Download model-performance summary",
            data=model_summary_csv,
            file_name="model_performance_summary.csv",
            mime="text/csv",
        )

    with download_col_2:
        st.download_button(
            "Download scenario summary",
            data=scenario_summary_csv,
            file_name="electrification_scenario_summary.csv",
            mime="text/csv",
        )

st.markdown(
    """
    <div class="drg-footer">
        Dynamic Resilient Grid • MSc Data Science Dissertation Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
