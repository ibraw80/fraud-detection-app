import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# إعدادات الصفحة
st.set_page_config(page_title="Enterprise AML & Compliance Risk Engine", layout="wide")

st.title("🛡️ Enterprise Fraud, AML & Compliance Analytics Engine")
st.markdown("منظومة حوكمة ومخاطر والتزام (GRC) متكاملة لاكتشاف الشذوذ المالي، التفتيش الرقابي، ومتابعة قواعد الامتثال.")

# توليد البيانات الأساسية وتدريب نموذج AI
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 10000
    data = {
        'transaction_id': [f"TXN-{100000 + i}" for i in range(n_samples)],
        'transaction_type': np.random.choice(['TRANSFER', 'CASH_OUT', 'PAYMENT'], size=n_samples, p=[0.4, 0.4, 0.2]),
        'amount': np.concatenate([np.random.exponential(scale=150, size=9900), np.random.uniform(8000, 60000, size=100)]),
        'oldbalanceOrg': np.random.uniform(1000, 100000, size=n_samples),
        'newbalanceOrig': np.random.uniform(0, 90000, size=n_samples),
        'frequency_24h': np.random.choice([1, 2, 3, 4, 8, 12], size=n_samples, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
    }
    df = pd.DataFrame(data)
    
    # تدريب نموذج AI Isolation Forest
    model = IsolationForest(contamination=0.015, random_state=42)
    df['AI_Anomaly'] = model.fit_predict(df[['amount', 'oldbalanceOrg', 'newbalanceOrig']])
    return df, model

df_sample, model = load_data()

# تطبيق قواعد الالتزام والتفتيش (Rule-Based Compliance Checks)
def apply_compliance_rules(row):
    reasons = []
    # قاعدة التكرار المرتفع خلال 24 ساعة (Velocity Check)
    if row['frequency_24h'] >= 6:
        reasons.append("تكرار عمليات مرتفع (High Frequency > 5/24h)")
    # قاعدة تجاوز الحد الأعلى للتحويل السريع (Threshold Rule)
    if row['amount'] >= 20000:
        reasons.append("تجاوز حد العمليات الكبير (Threshold > 20,000 SAR)")
    # رصد AI للشذوذ
    if row['AI_Anomaly'] == -1:
        reasons.append("شذوذ في نمط سلوك الأرصدة (AI Anomaly Detected)")
        
    risk_level = "HIGH RISK" if len(reasons) > 0 else "NORMAL"
    return pd.Series([risk_level, ", ".join(reasons) if reasons else "عملية طبيعية متوافقة"])

df_sample[['Risk_Level', 'Risk_Reasons']] = df_sample.apply(apply_compliance_rules, axis=1)

# تقسيم الواجهة إلى تبويبات احترافية (Tabs)
tab1, tab2, tab3 = st.tabs(["⚡ فحص عملية جديدة (Interactive Sandbox)", "🔍 لوحة التفتيش والالتزام (Compliance Dashboard)", "📊 مؤشرات المخاطر العامة"])

# ----------------- التبويب الأول: فحص فوري ومباشر -----------------
with tab1:
    st.subheader("🔍 اختبار فوري لعميل/عملية جديدة")
    st.markdown("إدخال معاملات عملية مالية لاختبار نموذج الذكاء الاصطناعي وقواعد الالتزام محلياً:")

    col_in1, col_in2, col_in3, col_in4 = st.columns(4)
    with col_in1:
        txn_type = st.selectbox("نوع المعاملة (Type)", ['TRANSFER', 'CASH_OUT', 'PAYMENT'])
    with col_in2:
        input_amount = st.number_input("المبلغ (Amount)", min_value=0.0, max_value=100000.0, value=250.0, step=50.0)
    with col_in3:
        input_old_bal = st.number_input("الرصيد السابق (Old Balance)", min_value=0.0, max_value=200000.0, value=5000.0, step=100.0)
    with col_in4:
        input_freq = st.number_input("تكرار العمليات في 24 ساعة", min_value=1, max_value=20, value=1)

    input_new_bal = max(0.0, input_old_bal - input_amount)

    if st.button("تشغيل التفتيش والفحص اللحظي ⚡"):
        # تقييم النموذج والـ Rules
        ai_pred = model.predict([[input_amount, input_old_bal, input_new_bal]])[0]
        
        reasons = []
        if input_freq >= 6:
            reasons.append("تكرار عمليات عالي جداً في فترة قصيرة (High Velocity)")
        if input_amount >= 20000:
            reasons.append("مبلغ مرتفع يتجاوز العتبة الرقابية (Large Threshold)")
        if ai_pred == -1:
            reasons.append("انحراف سلوكي في حركة الأرصدة (AI Anomaly)")

        if reasons:
            st.error("🚨 **تنبيه مخاطر/مخالفة التزام عالية (HIGH RISK FLAG)!**")
            st.warning("**أسباب رصد المخاطر:**\n* " + "\n* ".join(reasons))
        else:
            st.success("✅ **العملية سليمة ومطابقة لقواعد الالتزام مع عدم وجود مخاطر AI.**")

# ----------------- التبويب الثاني: لوحة التفتيش والالتزام -----------------
with tab2:
    st.subheader("📋 سجل التدقيق والتفتيش الرقابي (Audit & Compliance Log)")
    st.markdown("قائمة المعاملات المرصودة للتفتيش والمراجعة بناءً على قواعد الالتزام والذكاء الاصطناعي:")
    
    high_risk_df = df_sample[df_sample['Risk_Level'] == 'HIGH RISK']
    
    # أدوات تصفية وتفتيش (Filters)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_type = st.multiselect("تصفية حسب نوع العملية", options=['TRANSFER', 'CASH_OUT', 'PAYMENT'], default=['TRANSFER', 'CASH_OUT'])
    with col_f2:
        search_id = st.text_input("بحث برقم المعاملة (Transaction ID)")

    filtered_df = high_risk_df[high_risk_df['transaction_type'].isin(selected_type)]
    if search_id:
        filtered_df = filtered_df[filtered_df['transaction_id'].str.contains(search_id, case=False)]

    st.write(f"إجمالي الحالات المشبوهة المكتشفة للتفتيش: **{len(filtered_df)}** حالة")
    st.dataframe(filtered_df[['transaction_id', 'transaction_type', 'amount', 'frequency_24h', 'Risk_Reasons']], use_container_width=True)

# ----------------- التبويب الثالث: المؤشرات الإحصائية العامة -----------------
with tab3:
    st.subheader("📊 مؤشرات أداء الالتزام إدارة المخاطر (GRC KPIs)")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("إجمالي العمليات المفحوصة", f"{len(df_sample):,}")
    c2.metric("حالات الـ High Risk المرصودة", len(high_risk_df), delta_color="inverse")
    c3.metric("نسبة المخاطر الإجمالية", f"{(len(high_risk_df)/len(df_sample))*100:.2f}%")
    
    st.divider()
    st.markdown("### توزيع حالات المخاطر حسب نوع العملية")
    st.bar_chart(high_risk_df['transaction_type'].value_counts())
