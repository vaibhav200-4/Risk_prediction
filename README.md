# Risk_prediction

Here is a **clean, professional, and deployment-ready README.md** for your Failure Risk Prediction Streamlit app.
https://riskprediction-kdqyvba8b9hcd2yygh9qjd.streamlit.app/

---

# 📘 **KMRL – Failure Risk Prediction App (XGBoost)**

This is a Streamlit-based web application that predicts the **Failure Risk (0–1)** for Kochi Metro trainsets using a pre-trained **XGBoost model** (`xgb_failure_risk.json`).

The app supports:

* 🚂 **Manual single-row prediction**
* 📂 **Batch prediction from CSV**
* 📊 **Feature importance visualization**
* ⚙️ Automated feature preparation (dates → derived features)

---

## 🚀 **Features**

### ✅ **1. Load Pre-trained Model**

The app automatically loads your model:

```
xgb_failure_risk.json
```

No training is required.

---

### ✅ **2. Batch Prediction**

Upload a CSV with any of the following columns:

| Feature                     | Description                             |
| --------------------------- | --------------------------------------- |
| mileage_total               | Total mileage                           |
| days_since_last_maintenance | Derived if last_maintenance_date exists |
| days_since_FC_validation    | Derived if validation_date_of_FC exists |
| open_jobcard_count          | Number of open jobcards                 |
| high_priority_jobcard_count | High priority jobcards                  |
| fitness_certificate_status  | "valid" or "expired"                    |

The app prepares missing fields automatically.

Output CSV includes:

```
pred_failure_risk   # value from 0 to 1
```

---

### ✅ **3. Single Input Prediction**

Enter each parameter manually via the UI and get:

* **Predicted failure risk**
* **Interpretation**

  * 0–0.33 → Low
  * 0.33–0.66 → Medium
  * > 0.66 → High

---

### ✅ **4. Feature Importance (Gain)**

Visual bar chart showing how important each feature is to the XGBoost model.

---

## 📁 **Project Structure**

```
/
├── xgb_failure_risk.json      # Pre-trained XGBoost model
├── app.py (or risk_app.py)    # Streamlit application file
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 📦 **Installation (Local)**

### **1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### **2. Create environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Run the Streamlit App**

```bash
streamlit run app.py
```

(Use your actual app filename, e.g., `risk_app.py`)

---

## 🌐 **Deploying on Streamlit Cloud**

1. Push your project to GitHub
2. Go to: [https://share.streamlit.io](https://share.streamlit.io)
3. Select:

   * Repository
   * Branch
   * Main file → **app.py** or **risk_app.py**
4. Deploy 🎉

Make sure `requirements.txt` includes:

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.23.0
xgboost==2.1.3
plotly>=5.0.0
joblib>=1.2.0
python-dateutil>=2.8.0
```

---

## 🧠 **How Failure Risk is Computed**

The XGBoost model reads numeric and derived features:

* mileage_total
* days_since_last_maintenance
* days_since_FC_validation
* open_jobcard_count
* high_priority_jobcard_count
* fc_expired_flag (0 = valid, 1 = expired)

It outputs a continuous value: **0 → 1**

Where:

* **0.0–0.33 → Low failure risk**
* **0.33–0.66 → Medium**
* **0.66–1.00 → High**

---

## 🤝 **Contributing**

Issues and pull requests are welcome.
Feel free to open a ticket if you want:

* Dataset cleaning enhancements
* Support for training new models
* UI dashboard improvements

---

## 📜 **License**

This project is for internal research & decision support.
Copyright © 2025.

---

If you want, I can:

📌 customize README with screenshots
📌 generate a logo banner
📌 add API instructions
📌 create CI/CD workflow for auto deploy

Just tell me!

