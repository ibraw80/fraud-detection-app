import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# إعدادات الصفحة وعنوان الموقع
st.set_page_config(page_title="AI Fraud & AML Risk Analytics", layout="wide")

st.title("🛡️ AI-Powered Fraud Detection & AML Risk Dashboard")
st.markdown("نظام ذكاء اصطناعي مخصص لتحليل العمليات المالية واكتشاف الشذوذ ومخاطر غسيل الأموال لحظياً.")

# توليد بيانات التحليل (عينة 10,000 عملية)
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
    df['Risk_Flag'] = model.predict(df[['amount', 'oldbalanceOrg', 'newbalanceOrig']])
    df['Risk_Level'] = df['Risk_Flag'].map({1: 'Normal', -1: 'HIGH RISK (Fraud/AML)'})
    return df

df_sample = load_data()

# إحصائيات سريعة في واجهة الموقع (Metrics)
col1, col2 = st.columns(2)
col1.metric("إجمالي العمليات المفحوصة", f"{len(df_sample):,}")
alerts = df_sample[df_sample['Risk_Level'] == 'HIGH RISK (Fraud/AML)']
col2.metric("التنبيهات عالية المخاطر (Fraud Alerts)", len(alerts), delta_color="inverse")

# جدول العمليات المشبوهة
st.subheader("🚨 قائمة العمليات المالية المشبوهة المرصودة")
st.dataframe(alerts[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'Risk_Level']])
