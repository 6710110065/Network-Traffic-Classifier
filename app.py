import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="VNAT Traffic Classifier",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Network Traffic Classification App")

# 1. โหลด Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('rf_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    feature_names = list(joblib.load('feature_names.joblib'))
    return model, label_encoder, feature_names

try:
    rf_model, label_encoder, feature_names = load_artifacts()
except Exception as e:
    st.error(f"ไม่สามารถโหลดไฟล์โมเดลได้: {e}")
    st.stop()

# 2. อ่านไฟล์ข้อมูล sample_data_with_labels.csv โดยตรง
try:
    input_df = pd.read_csv('sample_data_with_labels.csv')
except FileNotFoundError:
    st.error("ไม่พบไฟล์ sample_data_with_labels.csv ในโฟลเดอร์ปัจจุบัน")
    st.stop()

# 3. ระบุคอลัมน์ Target
target_col = None
for col in ['labels', 'label', 'Label', 'Labels']:
    if col in input_df.columns:
        target_col = col
        break

# 4. ส่วนเลือกคลาสและปุ่มทำนายผล
if target_col:
    unique_classes = list(input_df[target_col].unique())
    class_options = ["All Classes"] + unique_classes
else:
    class_options = ["All Classes"]

col_select, col_btn = st.columns([3, 1])

with col_select:
    selected_class = st.selectbox("🎯 เลือกคลาสที่ต้องการทดสอบ:", class_options)

with col_btn:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Predict", type="primary", width="stretch")

# 5. การสุ่มตัวอย่างและประมวลผลทำนาย
filtered_df = input_df.copy()
if target_col and selected_class != "All Classes":
    filtered_df = input_df[input_df[target_col] == selected_class].reset_index(drop=True)

if predict_btn:
    # สุ่มข้อมูล 5 รายการ
    n_samples = min(5, len(filtered_df))
    sample_df = filtered_df.sample(n=n_samples).reset_index(drop=True)

    if target_col:
        actual_labels = sample_df[target_col].copy()
        features_df = sample_df.drop(columns=[target_col]).copy()
    else:
        actual_labels = None
        features_df = sample_df.copy()

    # จัดเตรียม Feature ตามลำดับ feature_names
    X_input = features_df.reindex(columns=feature_names)
    X_input = X_input.replace([np.inf, -np.inf], np.nan)

    # ใส่ .values เพื่อแปลง DataFrame เป็น Array ป้องกัน Scikit-Learn Warning
    preds_encoded = rf_model.predict(X_input.values)
    preds_labels = label_encoder.inverse_transform(preds_encoded)
    probs = rf_model.predict_proba(X_input.values)
    confidences = np.max(probs, axis=1) * 100

    # แสดงตารางผลการทำนาย 5 รายการ
    results_df = pd.DataFrame({
        'รายการที่': range(1, len(preds_labels) + 1),
        'ผลทำนาย': preds_labels,
        'ความมั่นใจ': [f"{c:.2f}%" for c in confidences]
    })

    if actual_labels is not None:
        results_df['ค่าจริง'] = actual_labels.values
        results_df['ผลลัพธ์'] = np.where(results_df['ผลทำนาย'] == results_df['ค่าจริง'], '✅ ถูกต้อง', '❌ ผิด')
        results_df = results_df[['รายการที่', 'ค่าจริง', 'ผลทำนาย', 'ความมั่นใจ', 'ผลลัพธ์']]

    st.subheader("🚀 ผลการทำนาย (สุ่ม 5 ตัวอย่าง)")
    st.dataframe(results_df, width="stretch", hide_index=True)

    # 6. สรุปผล Confidence และ Accuracy 
    st.subheader("📌 สรุปผลการประเมิน ")

    avg_conf = np.mean(confidences)

    if actual_labels is not None:
        correct_count = (preds_labels == actual_labels.values).sum()
        acc = (correct_count / len(preds_labels)) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("ความมั่นใจเฉลี่ยรวม", f"{avg_conf:.2f}%")
        col2.metric("ความแม่นยำ", f"{acc:.2f}%")
    else:
        st.metric("ความมั่นใจเฉลี่ยรวม", f"{avg_conf:.2f}%")
