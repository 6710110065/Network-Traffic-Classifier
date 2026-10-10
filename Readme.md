# Traffic Analysis & Model Web App

ระบบวิเคราะห์ข้อมูล Traffic ด้วย Random Forest และจำลองหน้าเว็บด้วย Streamlit

## สมาชิกในกลุ่ม / ผู้จัดทำ
- ชื่อ-นามสกุล (รหัสนักศึกษา)

## โครงสร้างไฟล์ในโปรเจกต์
- `app.py` : โค้ดระบบหน้าเว็บ Streamlit
- `TrafficRF.ipynb` : Jupyter Notebook สำหรับการทำ Data Prep และ Train Model
- `rf_model.joblib` / `VNAT_Feature_Dataframe_release_1.h5` : ไฟล์โมเดลและข้อมูลที่เทรนแล้ว
- `sample_data_with_labels.csv` : ไฟล์ข้อมูลตัวอย่างสำหรับทดสอบ

## วิธีการติดตั้งและรันโปรเจกต์

1. **ติดตั้ง Library ที่จำเป็น:**
   ```bash
   pip install -r requirements.txt
   ```

2. **เปิดใช้งาน Web Application:**
   ```bash
   streamlit run app.py
   ```