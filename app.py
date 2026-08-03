import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Health Insurance Claim Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "❌ best_model.pkl was not found. "
        "Make sure it is in the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL APP
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(37, 99, 235, 0.15),
                transparent 35%
            ),
            radial-gradient(
                circle at top right,
                rgba(124, 58, 237, 0.12),
                transparent 35%
            ),
            #08111f;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        background:
            linear-gradient(
                135deg,
                #2563eb 0%,
                #4f46e5 45%,
                #9333ea 100%
            );

        padding: 42px;
        border-radius: 28px;

        margin-bottom: 30px;

        border: 1px solid rgba(255,255,255,0.15);

        box-shadow:
            0 20px 60px rgba(37,99,235,0.30);
    }

    .hero-title {
        color: white;
        font-size: 42px;
        font-weight: 900;
        line-height: 1.2;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #e0e7ff;
        font-size: 18px;
        line-height: 1.7;
        max-width: 900px;
    }


    /* ======================================================
       SECTION CARD
       ====================================================== */

    .section-card {
        background:
            linear-gradient(
                135deg,
                #111827,
                #172033
            );

        padding: 25px;

        border-radius: 22px;

        margin: 25px 0;

        border: 1px solid #263449;

        box-shadow:
            0 12px 35px rgba(0,0,0,0.25);
    }

    .section-title {
        color: white;

        font-size: 26px;

        font-weight: 800;

        margin-bottom: 8px;
    }

    .section-description {
        color: #94a3b8;

        font-size: 15px;

        line-height: 1.6;
    }


    /* ======================================================
       PREDICTION CARD
       ====================================================== */

    .prediction-card {

        background:
            linear-gradient(
                135deg,
                #064e3b,
                #047857,
                #059669
            );

        padding: 38px;

        border-radius: 25px;

        text-align: center;

        margin: 25px 0;

        border: 1px solid #10b981;

        box-shadow:
            0 18px 50px rgba(16,185,129,0.25);
    }

    .prediction-label {

        color: #d1fae5;

        font-size: 18px;

        font-weight: 600;

        letter-spacing: 1px;

        margin-bottom: 10px;
    }

    .prediction-value {

        color: white;

        font-size: 52px;

        font-weight: 900;

        letter-spacing: 1px;
    }

    .prediction-subtitle {

        color: #a7f3d0;

        font-size: 14px;

        margin-top: 10px;
    }


    /* ======================================================
       FEATURE CARDS
       ====================================================== */

    .feature-card {

        background:
            linear-gradient(
                135deg,
                #172033,
                #263449
            );

        padding: 22px;

        border-radius: 20px;

        text-align: center;

        border: 1px solid #334155;

        box-shadow:
            0 8px 25px rgba(0,0,0,0.20);

        min-height: 145px;
    }

    .feature-icon {
        font-size: 30px;
        margin-bottom: 7px;
    }

    .feature-title {

        color: #94a3b8;

        font-size: 14px;

        margin-bottom: 6px;
    }

    .feature-value {

        color: white;

        font-size: 24px;

        font-weight: 800;
    }


    /* ======================================================
       ANALYSIS CARD
       ====================================================== */

    .analysis-card {

        background:
            linear-gradient(
                135deg,
                #172554,
                #1e3a8a
            );

        border: 1px solid #3b82f6;

        padding: 25px;

        border-radius: 22px;

        margin: 25px 0;

        box-shadow:
            0 12px 35px rgba(59,130,246,0.15);
    }


    /* ======================================================
       INSIGHT CARD
       ====================================================== */

    .insight-card {

        background:
            linear-gradient(
                135deg,
                #422006,
                #78350f
            );

        border: 1px solid #f59e0b;

        padding: 25px;

        border-radius: 22px;

        margin: 25px 0;
    }

    .insight-title {

        color: #fef3c7;

        font-size: 21px;

        font-weight: 800;

        margin-bottom: 10px;
    }

    .insight-text {

        color: #fde68a;

        line-height: 1.8;

        font-size: 15px;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #0b1324,
                #111827
            );
    }


    /* ======================================================
       FORM
       ====================================================== */

    [data-testid="stForm"] {

        background:
            linear-gradient(
                135deg,
                #111827,
                #172033
            );

        border: 1px solid #263449;

        padding: 28px;

        border-radius: 22px;

        box-shadow:
            0 12px 35px rgba(0,0,0,0.25);
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    div.stButton > button {

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #7c3aed
            );

        color: white;

        border: none;

        border-radius: 12px;

        font-weight: 800;

        min-height: 48px;

        transition: all 0.25s ease;
    }

    div.stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 10px 25px rgba(124,58,237,0.35);
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {

        text-align: center;

        padding: 30px;

        margin-top: 50px;

        color: #64748b;

        border-top: 1px solid #1e293b;

        line-height: 1.8;
    }

    .footer-title {

        color: #94a3b8;

        font-size: 17px;

        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:15px;
        ">

        <div style="
            font-size:45px;
        ">
            💳
        </div>

        <div style="
            color:white;
            font-size:23px;
            font-weight:800;
        ">
            Insurance AI
        </div>

        <div style="
            color:#94a3b8;
            font-size:13px;
            margin-top:5px;
        ">
            Intelligent Claim Prediction
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.info(
        """
        **Random Forest Regressor**

        A Supervised machine learning model
        trained to estimate health insurance
        claim costs.
        """
    )

    st.markdown("### 🛡️ Prediction Reliability")

    st.info(
        """
        This estimate is generated by a machine learning model
        trained on historical insurance data.

        The model has been evaluated on data it has not seen during
        training to check how well it performs on new customers.
        """
    )

    reliability_col1, reliability_col2 = st.columns(2)

    with reliability_col1:

        st.metric(
            "🎯 Prediction Accuracy",
            "85%"
        )

    with reliability_col2:

        st.metric(
            "📉 Typical Prediction Difference",
            "≈ $3,600"
        )

    st.caption(
        "These figures describe the model's performance and are not a guarantee "
        "of the actual insurance claim."
    )

    st.divider()
    
    st.markdown("### 💡 How This Works")

    st.info(
        """
        **Get your estimated insurance claim in 3 simple steps:**

        1. 👤 Enter your personal information
        2. 🩺 Provide your health and lifestyle details
        3. 🎯 Click **Predict Insurance Claim**

        The system will estimate your expected insurance
        claim based on the information you provide.
        """
    )

    st.markdown("### 📊 Explore Your Result")

    st.write("After getting your prediction, you can explore:")

    st.write("📈 How age may affect your estimated claim")
    st.write("🚬 The difference between smoker and non-smoker estimates")
    st.write("⚖️ How BMI changes the estimated claim")
    st.write("❤️ How blood pressure changes the estimated claim")

    st.markdown("### ⚠️ Important Note")

    st.warning(
        """
        This prediction is an estimate generated by a
        machine learning model. It should not be considered
        an official insurance quote or medical advice.
        """
    )

# ============================================================
# HERO SECTION
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-title">
            💳 Health Insurance Claim Predictor
        </div>

        <div class="hero-subtitle">
            AI-powered estimation of health insurance claim costs
            using demographic, health, lifestyle, and geographic
            characteristics.
        </div>

    </div>
    """
)


# ============================================================
# CUSTOMER PROFILE HEADER
# ============================================================

st.html(
    """
    <div class="section-card">

        <div class="section-title">
            👤 Customer Profile
        </div>

        <div class="section-description">
            Enter the customer's demographic, health, lifestyle,
            and regional information to generate an estimated
            insurance claim.
        </div>

    </div>
    """
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    with col1:

        st.markdown("### 👤 Personal Information")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )


    # --------------------------------------------------------
    # HEALTH INFORMATION
    # --------------------------------------------------------

    with col2:

        st.markdown("### 🩺 Health Information")

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        bloodpressure = st.number_input(
            "Blood Pressure",
            min_value=60,
            max_value=200,
            value=120,
            step=1
        )

        diabetic = st.selectbox(
            "Diabetic",
            ["No", "Yes"]
        )


    # --------------------------------------------------------
    # LIFESTYLE & REGION
    # --------------------------------------------------------

    with col3:

        st.markdown("### 🧬 Lifestyle & Region")

        smoker = st.selectbox(
            "Smoker",
            ["No", "Yes"]
        )

        region = st.selectbox(
            "Region",
            [
                "Northeast",
                "Northwest",
                "Southeast",
                "Southwest"
            ]
        )

        st.write("")

        submitted = st.form_submit_button(
            "🚀 Predict Insurance Claim",
            use_container_width=True
        )


# ============================================================
# CREATE PREDICTION
# ============================================================

if submitted:

    input_data = pd.DataFrame({

        "age": [age],

        "gender": [gender],

        "bmi": [bmi],

        "bloodpressure": [bloodpressure],

        "diabetic": [diabetic],

        "children": [children],

        "smoker": [smoker],

        "region": [region]

    })


    try:

        prediction = float(
            model.predict(input_data)[0]
        )

        st.session_state["prediction"] = prediction

        st.session_state["input_data"] = input_data

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )

        st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.html(
        """
        <div class="section-card">

            <div class="section-title">
                🎯 Prediction Result
            </div>

            <div class="section-description">
                Estimated insurance claim based on the
                customer's profile.
            </div>

        </div>
        """
    )


    # ========================================================
    # PREDICTION CARD
    # ========================================================

    st.html(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                💰 Estimated Insurance Claim
            </div>

            <div class="prediction-value">
                ${prediction:,.2f}
            </div>

            <div class="prediction-subtitle">
                Machine Learning Estimated Value Per Year
            </div>

        </div>
        """
    )


    # ========================================================
    # CUSTOMER SNAPSHOT
    # ========================================================

    st.html(
        """
        <div class="section-card">

            <div class="section-title">
                📋 Customer Snapshot
            </div>

            <div class="section-description">
                Key characteristics used by the prediction model.
            </div>

        </div>
        """
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.html(
            f"""
            <div class="feature-card">

                <div class="feature-icon">
                    🎂
                </div>

                <div class="feature-title">
                    Age
                </div>

                <div class="feature-value">
                    {age}
                </div>

            </div>
            """
        )


    with c2:

        st.html(
            f"""
            <div class="feature-card">

                <div class="feature-icon">
                    ⚖️
                </div>

                <div class="feature-title">
                    BMI
                </div>

                <div class="feature-value">
                    {bmi:.1f}
                </div>

            </div>
            """
        )


    with c3:

        st.html(
            f"""
            <div class="feature-card">

                <div class="feature-icon">
                    ❤️
                </div>

                <div class="feature-title">
                    Blood Pressure
                </div>

                <div class="feature-value">
                    {bloodpressure}
                </div>

            </div>
            """
        )


    with c4:

        smoking_icon = "🚬" if smoker == "Yes" else "🌿"

        st.html(
            f"""
            <div class="feature-card">

                <div class="feature-icon">
                    {smoking_icon}
                </div>

                <div class="feature-title">
                    Smoking Status
                </div>

                <div class="feature-value">
                    {smoker}
                </div>

            </div>
            """
        )


    # ========================================================
    # ADDITIONAL METRICS
    # ========================================================

    st.write("")

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "👶 Children",
            children
        )

    with m2:

        st.metric(
            "🩺 Diabetic",
            diabetic
        )

    with m3:

        st.metric(
            "🌎 Region",
            region.title()
        )

    with m4:

        st.metric(
            "💰 Prediction",
            f"${prediction:,.0f}"
        )


    # ========================================================
    # INTERACTIVE ANALYSIS
    # ========================================================

    st.html(
        """
        <div class="analysis-card">

            <div class="section-title">
                🔬 Interactive What-If Analysis
            </div>

            <div class="section-description">
                Explore how changing individual characteristics
                affects the model's predicted insurance claim.
                The other customer characteristics remain fixed.
            </div>

        </div>
        """
    )


    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎂 Age Impact",
            "🚬 Smoking Impact",
            "⚖️ BMI Impact",
            "❤️ Blood Pressure Impact"
        ]
    )


    # ========================================================
    # AGE ANALYSIS
    # ========================================================

    with tab1:

        age_range = np.arange(
            18,
            81
        )


        age_data = pd.DataFrame({

            "age": age_range,

            "gender": gender,

            "bmi": bmi,

            "bloodpressure": bloodpressure,

            "diabetic": diabetic,

            "children": children,

            "smoker": smoker,

            "region": region

        })


        age_predictions = model.predict(
            age_data
        )


        age_analysis = pd.DataFrame({

            "Age": age_range,

            "Predicted Claim": age_predictions

        })


        fig = px.line(

            age_analysis,

            x="Age",

            y="Predicted Claim",

            markers=True,

            title="📈 Effect of Age on Predicted Claim"

        )


        fig.update_layout(

            template="plotly_dark",

            height=500,

            hovermode="x unified",

            yaxis_title="Predicted Claim ($)",

            xaxis_title="Age"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        current_age_prediction = float(
            model.predict(
                pd.DataFrame({
                    "age": [age],
                    "gender": [gender],
                    "bmi": [bmi],
                    "bloodpressure": [bloodpressure],
                    "diabetic": [diabetic],
                    "children": [children],
                    "smoker": [smoker],
                    "region": [region]
                })
            )[0]
        )


        young_prediction = float(
            model.predict(
                pd.DataFrame({
                    "age": [18],
                    "gender": [gender],
                    "bmi": [bmi],
                    "bloodpressure": [bloodpressure],
                    "diabetic": [diabetic],
                    "children": [children],
                    "smoker": [smoker],
                    "region": [region]
                })
            )[0]
        )


        old_prediction = float(
            model.predict(
                pd.DataFrame({
                    "age": [80],
                    "gender": [gender],
                    "bmi": [bmi],
                    "bloodpressure": [bloodpressure],
                    "diabetic": [diabetic],
                    "children": [children],
                    "smoker": [smoker],
                    "region": [region]
                })
            )[0]
        )


        a1, a2, a3 = st.columns(3)

        with a1:

            st.metric(
                "At Age 18",
                f"${young_prediction:,.0f}"
            )

        with a2:

            st.metric(
                "Current Age",
                f"${current_age_prediction:,.0f}"
            )

        with a3:

            st.metric(
                "At Age 80",
                f"${old_prediction:,.0f}"
            )


    # ========================================================
    # SMOKING ANALYSIS
    # ========================================================

    with tab2:

        smoker_data = pd.DataFrame({

            "age": [age, age],

            "gender": [gender, gender],

            "bmi": [bmi, bmi],

            "bloodpressure":
                [bloodpressure, bloodpressure],

            "diabetic": [diabetic, diabetic],

            "children": [children, children],

            "smoker": ["No", "Yes"],

            "region": [region, region]

        })


        smoker_predictions = model.predict(
            smoker_data
        )


        smoker_analysis = pd.DataFrame({

            "Smoking Status":
                ["Non-Smoker", "Smoker"],

            "Predicted Claim":
                smoker_predictions

        })


        fig = px.bar(

            smoker_analysis,

            x="Smoking Status",

            y="Predicted Claim",

            text_auto=".2f",

            title="🚬 Smoking Status Comparison"

        )


        fig.update_layout(

            template="plotly_dark",

            height=500,

            yaxis_title="Predicted Claim ($)",

            xaxis_title=""

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


        difference = (

            smoker_predictions[1]

            -

            smoker_predictions[0]

        )


        st.metric(

            "Estimated Smoking Impact",

            f"${difference:,.2f}"

        )


    # ========================================================
    # BMI ANALYSIS
    # ========================================================

    with tab3:

        bmi_range = np.arange(

            18,

            46,

            0.5

        )


        bmi_data = pd.DataFrame({

            "age": age,

            "gender": gender,

            "bmi": bmi_range,

            "bloodpressure": bloodpressure,

            "diabetic": diabetic,

            "children": children,

            "smoker": smoker,

            "region": region

        })


        bmi_predictions = model.predict(
            bmi_data
        )


        bmi_analysis = pd.DataFrame({

            "BMI": bmi_range,

            "Predicted Claim":
                bmi_predictions

        })


        fig = px.line(

            bmi_analysis,

            x="BMI",

            y="Predicted Claim",

            markers=True,

            title="⚖️ Effect of BMI on Predicted Claim"

        )


        fig.update_layout(

            template="plotly_dark",

            height=500,

            hovermode="x unified",

            yaxis_title="Predicted Claim ($)",

            xaxis_title="BMI"

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


    # ========================================================
    # BLOOD PRESSURE ANALYSIS
    # ========================================================

    with tab4:

        bp_range = np.arange(

            80,

            161,

            5

        )


        bp_data = pd.DataFrame({

            "age": age,

            "gender": gender,

            "bmi": bmi,

            "bloodpressure": bp_range,

            "diabetic": diabetic,

            "children": children,

            "smoker": smoker,

            "region": region

        })


        bp_predictions = model.predict(
            bp_data
        )


        bp_analysis = pd.DataFrame({

            "Blood Pressure":
                bp_range,

            "Predicted Claim":
                bp_predictions

        })


        fig = px.line(

            bp_analysis,

            x="Blood Pressure",

            y="Predicted Claim",

            markers=True,

            title="❤️ Effect of Blood Pressure on Predicted Claim"

        )


        fig.update_layout(

            template="plotly_dark",

            height=500,

            hovermode="x unified",

            yaxis_title="Predicted Claim ($)",

            xaxis_title="Blood Pressure"

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


    # ========================================================
    # MODEL INSIGHT
    # ========================================================

    st.html(
        f"""
        <div class="insight-card">

            <div class="insight-title">
                💡 Prediction Insight
            </div>

            <div class="insight-text">

                Based on the entered customer profile, the
                Random Forest model estimates an insurance claim
                of approximately

                <strong>${prediction:,.2f}</strong>.

                <br><br>

                Use the interactive analysis above to investigate
                how changes in age, smoking status, BMI, and blood
                pressure influence the model's prediction while
                keeping the remaining characteristics constant.

            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <div class="footer-title">
            💳 Health Insurance Claim Prediction System
        </div>

        <br>

        Built with Python • Pandas • Scikit-learn • Plotly • Streamlit

        <br><br>

        Machine Learning Regression Application

    </div>
    """
)