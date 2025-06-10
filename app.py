# app.py (ใน VS Code)
import streamlit as st
import joblib
import pandas as pd
import numpy as np # สำหรับ np.nan

# --- ย้าย st.set_page_config() มาอยู่บนสุดของไฟล์ ---
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# --- 1. โหลดโมเดลและส่วนประกอบที่จำเป็น ---
# (ส่วนนี้จะไม่มี st.success/st.error ในตอนโหลดแล้ว เพื่อหลีกเลี่ยง error)
try:
    # โหลด Logistic Regression model (ตามที่คุณเลือก)
    model = joblib.load("optimized_logistic_regression_model.pkl")
    is_logistic_regression_model = True
    # st.success("Optimized Logistic Regression Model loaded successfully!") # <-- ลบออกไปก่อน
except FileNotFoundError:
    st.error("Error: Optimized Logistic Regression Model file 'optimized_logistic_regression_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

try:
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    median_total_charges_train = joblib.load("median_total_charges_train.pkl")
    # st.success("Scaler, Feature Columns, and Median TotalCharges loaded successfully!") # <-- ลบออกไปก่อน
except Exception as e:
    st.error(f"Error loading assets (scaler/features/median_total_charges): {e}")
    st.stop()

# --- 2. กำหนดคอลัมน์ Categorical ที่ต้องทำ One-Hot Encoding (ต้องตรงกับตอน Preprocessing ใน Colab) ---
categorical_cols_for_api = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod',
    'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'
]

# --- 3. สร้าง User Interface (UI) ด้วย Streamlit ---
# ตอนนี้ st.title และ st.write สามารถอยู่ต่อจาก st.set_page_config ได้แล้ว
st.title("Telco Customer Churn Prediction")
st.markdown("---")
st.write("ป้อนข้อมูลลูกค้าเพื่อทำนายแนวโน้มการยกเลิกบริการ (Churn).")

# เพิ่ม st.success/st.error ที่โหลดไฟล์เข้ามาได้ ตรงนี้ UI จะปรากฏแล้ว
# เพื่อให้ผู้ใช้เห็นสถานะการโหลด
if 'model' in locals(): # ตรวจสอบว่าตัวแปร model ถูกสร้างขึ้นมาแล้ว
    st.success("Optimized Logistic Regression Model loaded successfully!")
if 'scaler' in locals() and 'feature_columns' in locals() and 'median_total_charges_train' in locals():
    st.success("Scaler, Feature Columns, and Median TotalCharges loaded successfully!")

# จัดเรียง Input Widgets เป็นคอลัมน์เพื่อให้ดูดี
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Gender of the customer.")
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Is the customer a senior citizen?")
    partner = st.selectbox("Partner", ["Yes", "No"], help="Does the customer have a partner?")
    dependents = st.selectbox("Dependents", ["Yes", "No"], help="Does the customer have dependents?")
    tenure = st.slider("Tenure (Months)", 0, 72, 1, help="Number of months the customer has stayed with the company.")

with col2:
    st.subheader("Service Information")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], help="Does the customer have phone service?")
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], help="Does the customer have multiple lines?")
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], help="Type of internet service.")
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], help="Does the customer have online security service?")
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], help="Does the customer have online backup service?")
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], help="Does the customer have device protection service?")

with col3:
    st.subheader("Billing & Contract")
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], help="Does the customer have tech support service?")
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], help="Does the customer have streaming TV service?")
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], help="Does the customer have streaming movies service.")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help="The contract term of the customer.")
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], help="Does the customer use paperless billing?")
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], help="The customer's payment method.")
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0, step=0.1, help="The amount charged to the customer monthly.")
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=9000.0, value=1000.0, step=0.1, help="The total amount charged to the customer.")

st.markdown("---") # เส้นคั่น

# --- 4. ปุ่มทำนายผล ---
if st.button("Predict Churn"):
    # 4.1. รวบรวมข้อมูลจาก UI เป็น DataFrame (1 แถว)
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    input_df = pd.DataFrame([input_data])

    # 4.2. Preprocessing ข้อมูล Input ให้เหมือนตอน Train โมเดล (ต้องตรงกับใน Colab เป๊ะๆ)
    # 4.2.1 จัดการ TotalCharges (เติม NaN ด้วย median_total_charges_train)
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
    input_df['TotalCharges'].fillna(median_total_charges_train, inplace=True)

    # 4.2.2 จัดการ Binary Categorical Features (แปลง Yes/No เป็น 1/0, Male/Female เป็น 1/0)
    binary_cols_map = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols_map:
        if col in input_df.columns:
            input_df[col] = input_df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # 4.2.3 จัดการ 'No internet service' และ 'No phone service'
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']:
        if col in input_df.columns:
            input_df[col] = input_df[col].replace('No internet service', 'No')
    if 'MultipleLines' in input_df.columns:
        input_df['MultipleLines'] = input_df['MultipleLines'].replace('No phone service', 'No')

    # 4.2.4 One-Hot Encoding สำหรับ Multi-Categorical Features
    input_df_processed = pd.get_dummies(input_df, columns=[col for col in categorical_cols_for_api if col in input_df.columns], drop_first=True)

    # 4.2.5 จัดการคอลัมน์ที่ขาดหายไปหรือเกินมา (สำคัญมาก!)
    final_input = pd.DataFrame(columns=feature_columns)
    final_input = pd.concat([final_input, input_df_processed], ignore_index=True)
    final_input.fillna(0, inplace=True)
    final_input = final_input[feature_columns]


    # 4.2.6 Feature Scaling (สำหรับ Logistic Regression เท่านั้น)
    final_input_scaled = scaler.transform(final_input)

    # --- 5. ทำนายผล ---
    churn_proba = model.predict_proba(final_input_scaled)[:, 1][0]
    churn_prediction = "Yes" if churn_proba >= 0.5 else "No"

    # --- 6. แสดงผลลัพธ์บน UI ---
    st.subheader("Prediction Result:")
    if churn_prediction == "Yes":
        st.error(f"**Customer is predicted to CHURN! 📉**")
        st.write(f"Probability of Churn: **{churn_proba:.2%}**")
        st.warning("Recommended Action: Engage with this customer with retention strategies (e.g., offer special promotions, personalized support).")
    else:
        st.success(f"**Customer is predicted to NOT CHURN. 🎉**")
        st.write(f"Probability of Churn: **{churn_proba:.2%}**")
        st.info("Recommended Action: Continue to monitor and provide excellent service to maintain satisfaction.")

    st.markdown("---")
    st.write("Disclaimer: This is a predictive model. Actual outcomes may vary.")