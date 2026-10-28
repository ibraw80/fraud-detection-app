import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# إعدادات الصفحة وعنوان الموقع
st.set_page_config(page_title="AI Fraud & AML Risk Analytics", layout="wide")

st.title("🛡️ AI-Powered Fraud Detection & AML Risk Dashboard")
st.markdown("نظام ذكاء اصطناعي مخصص لتحليل العمليات المالية واكتشاف الشذوذ ومخاطر غسيل الأموال لحظياً.")

# توليد بيانات التحليل (عينة 10,000 عملية) وتدريب النموذج
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 10000
    data = {
        'amount': np.concatenate([np.random.exponential(scale=100, size=9900), np.random.uniform(5000, 50000, size=100)]),
        'oldbalanceOrg': np.random.uniform(1000, 100000, size=n_samples),
        'newbalanceOrig': np.random.uniform(0, 90000, size=n_samples)
    }
    df = pd.DataFrame(data)
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df[['amount', 'oldbalanceOrg', 'newbalanceOrig']])
    return df, model

df_sample, model = load_data()
df_sample['Risk_Flag'] = model.predict(df_sample[['amount', 'oldbalanceOrg', 'newbalanceOrig']])
df_sample['Risk_Level'] = df_sample['Risk_Flag'].map({1: 'Normal', -1: 'HIGH RISK (Fraud/AML)'})

# إحصائيات سريعة في واجهة الموقع (Metrics)
col1, col2 = st.columns(2)
col1.metric("إجمالي العمليات المفحوصة", f"{len(df_sample):,}")
alerts = df_sample[df_sample['Risk_Level'] == 'HIGH RISK (Fraud/AML)']
col2.metric("التنبيهات عالية المخاطر (Fraud Alerts)", len(alerts), delta_color="inverse")

st.divider()

# قسم تفاعلي جديد: اختبار عملية مالية جديدة مباشرة من المستخدم
st.subheader("🔍 اختبار عملية مالية جديدة (Interactive Risk Assessment)")
st.markdown("قم بتعديل قيم المعاملات المالية بالأسفل لاختبار تقييم نموذج الذكاء الاصطناعي واكتشاف ما إذا كانت مشبوهة أم لا:")

col_input1, col_input2, col_input3 = st.columns(3)
with col_input1:
    input_amount = st.number_input("قيمة العملية (Amount)", min_value=0.0, max_value=100000.0, value=150.0, step=10.0)
with col_input2:
    input_old_bal = st.number_input("الرصيد السابق للمرسل (Old Balance)", min_value=0.0, max_value=200000.0, value=5000.0, step=100.0)
with col_input3:
    input_new_bal = st.number_input("الرصيد اللاحق (New Balance)", min_value=0.0, max_value=200000.0, value=4850.0, step=100.0)

# زر التقييم الفوري
if st.button("فحص واختبار العملية لحظياً ⚡"):
    input_data = pd.DataFrame([[input_amount, input_old_bal, input_new_bal]], columns=['amount', 'oldbalanceOrg', 'newbalanceOrig'])
    prediction = model.predict(input_data)[0]
    
    if prediction == -1:
        st.error("🚨 **تحذير خطير: تم رصد شذوذ (HIGH RISK - احتمال احتيال أو غسيل أموال)!**")
    else:
        st.success("✅ **العملية طبيعية (Normal Transaction) - لا توجد مخاطر مرصودة.**")

st.divider()

# جدول العمليات المشبوهة التلقائي
st.subheader("🚨 قائمة العمليات المالية المشبوهة المرصودة في العينة التاريخية")
st.dataframe(alerts[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'Risk_Level']])
