
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cpu")


# CNN model
class BreastCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


# Load saved models
@st.cache_resource
def load_models():
    ml_model = joblib.load(
        os.path.join(MODEL_DIR, "breast_cancer_lr_5_features.pkl")
    )

    features = joblib.load(
        os.path.join(MODEL_DIR, "selected_features_5.pkl")
    )

    cnn_model = BreastCancerCNN()

    weights = torch.load(
        os.path.join(MODEL_DIR, "breastmnist_cnn_final.pth"),
        map_location=DEVICE
    )

    cnn_model.load_state_dict(weights)
    cnn_model.eval()

    return ml_model, features, cnn_model


# Application styling
st.markdown("""
<style>

/* Overall page */
.stApp {
    background-color: #f5f8f5 !important;
    color: #17251c !important;
    color-scheme: light !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"] {
    background-color: #f5f8f5 !important;
}

/* Main heading */
.hero {
    background: linear-gradient(110deg, #103b27, #26764c);
    padding: 27px 32px;
    border-radius: 15px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 2.1rem;
    margin: 0;
}

.hero p {
    color: #e5f1e8 !important;
    margin: 8px 0 0;
}

.hero small {
    color: #d5e8da !important;
}

/* Navigation tabs */
div[data-testid="stTabs"] [role="tablist"] {
    gap: 10px;
    border-bottom: 2px solid #cbd8ce;
}

div[data-testid="stTabs"] button[role="tab"] {
    background: #e7f0e9 !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid #d2e1d5 !important;
    padding: 12px 18px !important;
    opacity: 1 !important;
    filter: none !important;
}

div[data-testid="stTabs"] button[role="tab"] *,
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] span {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    opacity: 1 !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #d5eadb !important;
    border-bottom: 3px solid #176b40 !important;
}

div[data-testid="stTabs"] button[role="tab"]:hover {
    background: #dceee0 !important;
}

/* Section cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #d8e5da !important;
    border-radius: 13px !important;
}

/* Headings and captions inside cards */
div[data-testid="stVerticalBlockBorderWrapper"] h3,
div[data-testid="stVerticalBlockBorderWrapper"] h4 {
    color: #173b27 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] p,
div[data-testid="stVerticalBlockBorderWrapper"] label,
div[data-testid="stVerticalBlockBorderWrapper"] label p,
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"],
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] p {
    color: #263d2e !important;
    -webkit-text-fill-color: #263d2e !important;
    opacity: 1 !important;
}

/* Number input labels */
div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInput"] label p,
div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"],
div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p {
    color: #263d2e !important;
    -webkit-text-fill-color: #263d2e !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Number input boxes */
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: #ffffff !important;
    border: 1px solid #b9cdbd !important;
    border-radius: 8px !important;
}

div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: #17251c !important;
    -webkit-text-fill-color: #17251c !important;
    opacity: 1 !important;
}

div[data-testid="stNumberInput"] button {
    background: #edf5ef !important;
    color: #176b40 !important;
}

div[data-testid="stNumberInput"] button svg {
    fill: #176b40 !important;
    color: #176b40 !important;
}

/* Prediction metrics */
div[data-testid="stMetric"] {
    background: #edf5ef !important;
    border: 1px solid #d5e6d9 !important;
    border-radius: 10px !important;
    padding: 13px !important;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #355840 !important;
    -webkit-text-fill-color: #355840 !important;
    opacity: 1 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #145c35 !important;
    -webkit-text-fill-color: #145c35 !important;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    background: #176b40 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 650 !important;
    min-height: 43px;
}

.stButton button p,
.stFormSubmitButton button p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #10532f !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: #f8fbf8 !important;
    border: 1px dashed #8fb49a !important;
    border-radius: 11px !important;
    padding: 12px !important;
}

div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] label {
    color: #263d2e !important;
    -webkit-text-fill-color: #263d2e !important;
}

/* Prediction result panels */
.result-success {
    background: #e8f5eb !important;
    border: 1px solid #b7ddc1;
    border-left: 5px solid #19804a;
    border-radius: 9px;
    padding: 15px;
    color: #145b32 !important;
    font-size: 19px;
    font-weight: 700;
}

.result-warning {
    background: #fff0ef !important;
    border: 1px solid #efc5c1;
    border-left: 5px solid #c23b36;
    border-radius: 9px;
    padding: 15px;
    color: #982b27 !important;
    font-size: 19px;
    font-weight: 700;
}

/* Information panels */
.info-box {
    background: #eff6f0 !important;
    border: 1px solid #d7e6da;
    border-left: 4px solid #23804b;
    border-radius: 9px;
    padding: 15px;
    margin-top: 12px;
    color: #294b34 !important;
    font-size: 0.9rem;
    line-height: 1.7;
}

.info-box strong {
    color: #145c35 !important;
}

.section-caption {
    color: #536b5a !important;
    font-size: 0.95rem;
}

.stProgress > div > div > div > div {
    background: #19804a !important;
}

.footer {
    text-align: center;
    color: #536b5b !important;
    font-size: 13px;
    padding-top: 18px;
    border-top: 1px solid #dce6de;
    margin-top: 28px;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="hero">
    <h1>🩺 Breast Cancer Detection System</h1>
    <p>Classification using Machine Learning and Deep Learning</p>
    <small>WDBC Dataset &nbsp; • &nbsp; BreastMNIST &nbsp; • &nbsp; Python</small>
</div>
""", unsafe_allow_html=True)


# Initialize models
try:
    ml_model, features, cnn_model = load_models()

    if len(features) != 5:
        st.error("The ML model must use exactly five features.")
        st.stop()

except Exception as error:
    st.error(f"Unable to load models: {error}")
    st.info("Check that all three model files are inside your models folder.")
    st.stop()


# Two prediction sections
ml_tab, dl_tab = st.tabs([
    "📊 Machine Learning",
    "🖼️ Deep Learning"
])


# Machine Learning section
with ml_tab:

    st.subheader("WDBC · Logistic Regression")
    st.markdown(
        '<p class="section-caption">Enter five numerical measurements '
        'to generate a classification prediction.</p>',
        unsafe_allow_html=True
    )

    input_col, result_col = st.columns([1.05, 0.95], gap="large")

    with input_col:

        with st.container(border=True):

            st.markdown("### Patient Measurements")
            st.caption("Enter the numerical feature values.")

            defaults = {
                "concave_points_mean": 0.05,
                "radius_worst": 16.0,
                "perimeter_worst": 105.0,
                "area_worst": 800.0,
                "concave_points_worst": 0.08
            }

            with st.form("ml_form"):

                values = {}
                columns = st.columns(2)

                for i, feature in enumerate(features):

                    with columns[i % 2]:

                        values[feature] = st.number_input(
                            feature.replace("_", " ").title(),
                            min_value=0.0,
                            value=float(defaults.get(feature, 0.0)),
                            format="%.5f",
                            key=f"ml_{feature}"
                        )

                st.write("")

                predict_ml = st.form_submit_button(
                    "Predict Result",
                    use_container_width=True
                )

            st.markdown("""
            <div class="info-box">
                <strong>Input information</strong><br>
                Dataset: WDBC<br>
                Selected features: 5<br>
                Classifier: Logistic Regression
            </div>
            """, unsafe_allow_html=True)

    with result_col:

        with st.container(border=True):

            st.markdown("### Prediction Result")
            st.caption("Classification output and probability scores.")

            if predict_ml:

                input_df = pd.DataFrame([values], columns=features)

                prediction = int(ml_model.predict(input_df)[0])
                probabilities = ml_model.predict_proba(input_df)[0]
                classes = list(ml_model.classes_)

                benign_prob = float(
                    probabilities[classes.index(0)]
                ) * 100

                malignant_prob = float(
                    probabilities[classes.index(1)]
                ) * 100

                if prediction == 0:
                    st.markdown(
                        '<div class="result-success">'
                        '✓ &nbsp; Prediction: Benign'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-warning">'
                        '! &nbsp; Prediction: Malignant'
                        '</div>',
                        unsafe_allow_html=True
                    )

                st.write("")

                col1, col2 = st.columns(2)

                col1.metric("Benign Probability", f"{benign_prob:.2f}%")
                col2.metric("Malignant Probability", f"{malignant_prob:.2f}%")

                st.progress(
                    malignant_prob / 100,
                    text="Malignant probability"
                )

                st.markdown("""
                <div class="info-box">
                    <strong>Result interpretation</strong><br>
                    These values represent estimated model probabilities,
                    not clinical certainty.
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="info-box">
                    <strong>Ready for prediction</strong><br><br>
                    Enter the five measurements and select
                    <b>Predict Result</b>.<br><br>
                    Your predicted class and probability scores
                    will appear here.
                </div>
                """, unsafe_allow_html=True)

                st.write("")

                col1, col2 = st.columns(2)

                col1.metric("Input Features", "05")
                col2.metric("Model", "LR")

                st.caption("Logistic Regression · WDBC")

            st.markdown("---")
            st.caption("Educational prototype only. Not intended for medical diagnosis.")


# Deep Learning section
with dl_tab:

    st.subheader("BreastMNIST · Convolutional Neural Network")
    st.markdown(
        '<p class="section-caption">Upload an ultrasound image '
        'to generate a CNN classification prediction.</p>',
        unsafe_allow_html=True
    )

    upload_col, result_col = st.columns([1.05, 0.95], gap="large")

    with upload_col:

        with st.container(border=True):

            st.markdown("### Ultrasound Image")
            st.caption("Upload a PNG, JPG or JPEG image.")

            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["png", "jpg", "jpeg"],
                key="cnn_upload"
            )

            image = None

            if uploaded_file is not None:

                try:
                    image = Image.open(uploaded_file).convert("L")

                    st.image(
                        image,
                        caption="Uploaded ultrasound image",
                        use_container_width=True
                    )

                except Exception:
                    st.error("Unable to read this image.")
                    image = None

            else:

                st.markdown("""
                <div class="info-box">
                    <strong>Image upload</strong><br><br>
                    Select an ultrasound image from your device.
                    Its preview will appear here.
                </div>
                """, unsafe_allow_html=True)

            classify = st.button(
                "Classify Image",
                use_container_width=True,
                disabled=image is None
            )

            st.markdown("""
            <div class="info-box">
                <strong>Image processing</strong><br>
                Grayscale conversion<br>
                Resize: 28 × 28 pixels<br>
                Model: CNN
            </div>
            """, unsafe_allow_html=True)

    with result_col:

        with st.container(border=True):

            st.markdown("### Prediction Result")
            st.caption("CNN classification and probability scores.")

            if image is None:

                st.markdown("""
                <div class="info-box">
                    <strong>Waiting for image</strong><br><br>
                    Upload an ultrasound image in the left panel.
                    Then select <b>Classify Image</b> to view the result.
                </div>
                """, unsafe_allow_html=True)

                st.write("")

                col1, col2 = st.columns(2)

                col1.metric("Image Size", "28 × 28")
                col2.metric("Model", "CNN")

            elif classify:

                resized = image.resize(
                    (28, 28),
                    Image.Resampling.BILINEAR
                )

                image_array = np.asarray(
                    resized,
                    dtype=np.float32
                ) / 255.0

                tensor = torch.tensor(
                    image_array,
                    dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(DEVICE)

                with torch.no_grad():

                    output = cnn_model(tensor)
                    probabilities = torch.softmax(output, dim=1)[0]

                    prediction = int(
                        torch.argmax(probabilities).item()
                    )

                malignant_prob = float(probabilities[0]) * 100
                normal_prob = float(probabilities[1]) * 100

                if prediction == 0:
                    st.markdown(
                        '<div class="result-warning">'
                        '! &nbsp; Prediction: Malignant'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-success">'
                        '✓ &nbsp; Prediction: Normal / Benign'
                        '</div>',
                        unsafe_allow_html=True
                    )

                st.write("")

                col1, col2 = st.columns(2)

                col1.metric("Normal / Benign", f"{normal_prob:.2f}%")
                col2.metric("Malignant", f"{malignant_prob:.2f}%")

                st.progress(
                    malignant_prob / 100,
                    text="Malignant probability"
                )

                st.markdown("""
                <div class="info-box">
                    <strong>Result interpretation</strong><br>
                    The CNN returns probabilities for its two
                    BreastMNIST output classes.
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="info-box">
                    <strong>Image ready</strong><br><br>
                    Your image has been uploaded. Select
                    <b>Classify Image</b> to run the trained CNN.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            st.caption(
                "BreastMNIST images are 28 × 28 grayscale. "
                "External images may differ from the training distribution."
            )


# Footer
st.markdown("""
<div class="footer">
    Built using Python • Scikit-learn • PyTorch • Streamlit
</div>
""", unsafe_allow_html=True)
