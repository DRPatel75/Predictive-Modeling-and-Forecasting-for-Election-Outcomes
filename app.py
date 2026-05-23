import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------------
# Page Config
# ------------------------------

st.set_page_config(
    page_title="Election Outcome Prediction",
    page_icon="🗳️",
    layout="wide"
)

# ------------------------------
# Load Models
# ------------------------------

rf = joblib.load("models/random_forest.pkl")

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.title("🗳️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Prediction",
        "Model Comparison",
        "Graphs",
        "About"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.title("🗳️ Election Outcome Prediction & Forecasting")

    st.markdown("---")

    st.write("""
    This project predicts election outcomes using Machine Learning.

    ### Models Used

    - Logistic Regression
    - Random Forest
    - XGBoost

    ### Forecasting Models

    - ARIMA
    - LSTM

    ### Evaluation Metrics

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - AUC
    """)

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.title("🏆 Election Winner Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st_name = st.number_input(
            "State Code",
            min_value=0,
            value=0
        )

        year = st.number_input(
            "Election Year",
            min_value=1977,
            max_value=2035,
            value=2014
        )

        pc_no = st.number_input(
            "Constituency Number",
            min_value=1,
            value=1
        )

        pc_name = st.number_input(
            "Constituency Name Code",
            min_value=0,
            value=37
        )

        pc_type = st.selectbox(
            "Constituency Type Code",
            [0, 1, 2, 3, 4]
        )

    with col2:

        cand_sex = st.selectbox(
            "Candidate Gender Code",
            [0, 1, 2, 3]
        )

        partyname = st.number_input(
            "Party Code",
            min_value=0,
            value=560
        )

        partyabbre = st.number_input(
            "Party Abbreviation Code",
            min_value=0,
            value=414
        )

        electors = st.number_input(
            "Number of Electors",
            min_value=0,
            value=100000
        )

    if st.button("Predict Winner"):

        input_data = pd.DataFrame(
            [[
                st_name,
                year,
                pc_no,
                pc_name,
                pc_type,
                cand_sex,
                partyname,
                partyabbre,
                electors
            ]],
            columns=[
                'st_name',
                'year',
                'pc_no',
                'pc_name',
                'pc_type',
                'cand_sex',
                'partyname',
                'partyabbre',
                'electors'
            ]
        )

        prediction = rf.predict(input_data)

        if prediction[0] == 1:

            st.success("🎉 Predicted Winner")

        else:

            st.error("❌ Predicted Loser")
# =====================================================
# MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    comparison = pd.DataFrame({
        "Model":[
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy":[
            0.0131,
            0.9894,
            0.9893
        ]
    })

    st.dataframe(comparison)

    st.bar_chart(
        comparison.set_index("Model")
    )

# =====================================================
# GRAPHS PAGE
# =====================================================

elif page == "Graphs":

    st.title("📈 Project Visualizations")

    st.subheader("Accuracy Comparison")
    st.image(
        "graphs/accuracy_comparison.png",
        use_container_width=True
    )

    st.subheader("ARIMA Forecast")
    st.image(
        "graphs/arima_forecast.png",
        use_container_width=True
    )

    st.subheader("LSTM Forecast")
    st.image(
        "graphs/lstm_graph.png",
        use_container_width=True
    )

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About":

    st.title("ℹ️ About Project")

    st.write("""
    ### Predictive Modeling and Forecasting for Election Outcomes

    Internship Project

    Technologies Used:

    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - XGBoost
    - TensorFlow
    - StatsModels
    - Streamlit

    Developed to predict election winners and forecast future election trends.
    """)