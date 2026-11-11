# 🔬 Breast Cancer Diagnostic Assistant

An interactive machine learning and computer vision web application developed as an internship project for breast cancer screening. The application integrates two diagnostic modalities:

1. **Tabular Fine Needle Aspirate (FNA) Cytology Analysis** using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset with the top 5 features selected via Random Forest importance.
2. **Medical Ultrasound Image Classification** using a 2-stage **PyTorch Convolutional Neural Network (CNN)** trained on the **BreastMNIST** benchmark dataset.

---

## 🌟 Key Features

- **Clean, Minimalist Medical UI:** Styled in a white, medical green, and dark charcoal palette designed for readability, high contrast, and ease of navigation.
- **Top 5 Feature Modeling (WDBC):**
  - Uses 5 key morphometric cell features: `concave_points_mean`, `radius_worst`, `perimeter_worst`, `area_worst`, and `concave_points_worst`.
  - Supports switching between **Logistic Regression** (95.6% Accuracy, 95.2% Malignant Recall) and **Random Forest** (93.9% Accuracy).
  - One-click presets: *Load Typical Benign Case*, *Load Typical Malignant Case*, or input custom values.
- **Ultrasound Deep Learning (BreastMNIST CNN):**
  - Accepts uploaded scans (.png, .jpg, .jpeg) or lets users test directly using built-in BreastMNIST test samples.
  - Automatically converts scans to 28×28 grayscale, normalizes pixel values, and predicts Malignant vs. Normal/Benign with confidence breakdown.
- **Experimental Benchmarks & Reporting:**
  - Includes full evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and confusion matrices from the exploratory notebooks.

---

## 📁 Repository Structure

```text
├── app.py                              # Main Streamlit web application
├── requirements.txt                    # Project dependencies for deployment
├── .gitignore                          # Git ignore rules (excludes venv, caches)
├── README.md                           # Documentation and deployment guide
├── models/
│   ├── breast_cancer_lr_5_features.pkl # Trained Logistic Regression pipeline
│   ├── breast_cancer_rf_5_features.pkl # Trained Random Forest classifier
│   ├── selected_features_5.pkl         # List of 5 selected features
│   └── breastmnist_cnn_final.pth       # Trained PyTorch CNN weights
├── sample_images/                      # Benchmark test ultrasound images
│   ├── sample_benign_1.png
│   ├── sample_benign_2.png
│   ├── sample_benign_3.png
│   ├── sample_malignant_1.png
│   ├── sample_malignant_2.png
│   └── sample_malignant_3.png
├── ML_Model_Breast_Cancer.ipynb        # Data preparation, EDA & tabular modeling
├── DL_Model_Breast_cancer.ipynb        # BreastMNIST PyTorch CNN training
└── breast+cancer+wisconsin+diagnostic/ # WDBC raw dataset files
```

---

## 🚀 Running Locally

1. **Clone the repository or open the project folder:**
   ```bash
   cd Breast_Cancer_Intern_task
   ```

2. **Activate your virtual environment (if using one):**
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   The application will open automatically in your browser at `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Community Cloud

You can deploy this application for free on Streamlit Community Cloud:

1. **Push the project to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Breast cancer screening app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
   - Click **"New app"**.
   - Select your GitHub repository (`<your-username>/<your-repo-name>`), branch (`main`), and set the main file path to:
     ```text
     app.py
     ```
   - Click **"Deploy!"**. Streamlit Cloud will read `requirements.txt` and launch your live web application.

---

## ⚠️ Academic Disclaimer

This application was developed as an academic internship study to demonstrate machine learning and deep learning applications in healthcare informatics. It is designed for educational and demonstration purposes only and should not be used as a substitute for professional medical diagnosis or clinical consultation.
