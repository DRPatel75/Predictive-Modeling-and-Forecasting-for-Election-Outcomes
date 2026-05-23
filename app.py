import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Election Outcome Prediction",
    page_icon="🗳️",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

rf = joblib.load("models/random_forest.pkl")

# =====================================================
# LOAD ENCODERS
# =====================================================

state_encoder = joblib.load("models/st_namest_name_encoder.pkl")
pc_encoder = joblib.load("models/pc_namest_name_encoder.pkl")
party_encoder = joblib.load("models/partynamest_name_encoder.pkl")
partyabbre_encoder = joblib.load("models/partyabbrest_name_encoder.pkl")
gender_encoder = joblib.load("models/cand_sexst_name_encoder.pkl")
pctype_encoder = joblib.load("models/pc_typest_name_encoder.pkl")

# =====================================================
# SIDEBAR
# =====================================================

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
# HOME
# =====================================================

if page == "Home":

    df = pd.read_csv("dataset/election_data.csv")

    st.metric("Records", len(df))
    st.metric("States", df['st_name'].nunique())
    st.metric("Parties", df['partyname'].nunique())

    st.title("🗳️ Election Outcome Prediction & Forecasting")

    st.markdown("---")

    st.write("""
    This project predicts election outcomes using machine learning models.

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
# PREDICTION
# =====================================================

elif page == "Prediction":

    st.title("🏆 Election Winner Prediction")

    col1, col2 = st.columns(2)

    with col1:

        state = st.selectbox(
            "State",
            list(state_encoder.classes_)
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

        constituency = st.selectbox(
            "Constituency",
            list(pc_encoder.classes_)
        )

        constituency_type = st.selectbox(
            "Constituency Type",
            list(pctype_encoder.classes_)
        )

    with col2:

        gender = st.selectbox(
            "Candidate Gender",
            list(gender_encoder.classes_)
        )

        party = st.selectbox(
            "Party",
            list(party_encoder.classes_)
        )

        party_abbre = st.selectbox(
            "Party Abbreviation",
            list(partyabbre_encoder.classes_)
        )

        electors = st.number_input(
            "Number of Electors",
            min_value=0,
            value=100000
        )

    if st.button("Predict Winner"):

        st_name = state_encoder.transform([state])[0]

        pc_name = pc_encoder.transform([constituency])[0]

        pc_type = pctype_encoder.transform([constituency_type])[0]

        cand_sex = gender_encoder.transform([gender])[0]

        partyname = party_encoder.transform([party])[0]

        partyabbre = partyabbre_encoder.transform([party_abbre])[0]

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

            prob = rf.predict_proba(input_data)[0][1]

            st.success(f"Winner Probability: {prob*100:.2f}%")

        else:

            st.error("❌ Predicted Not Winner")

# =====================================================
# MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    comparison = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],

        "Accuracy": [
            0.7859,
            0.9897,
            0.9893
        ]
    })

    st.dataframe(comparison)

    st.bar_chart(
        comparison.set_index("Model")
    )

# =====================================================
# GRAPHS
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
# ABOUT
# =====================================================

elif page == "About":

    st.title("ℹ️ About Project")

    st.write("""
    ### Predictive Modeling and Forecasting for Election Outcomes

    Technologies Used:

    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - XGBoost
    - TensorFlow
    - Statsmodels
    - Streamlit

    Models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - ARIMA
    - LSTM
    """)