"""
HR Workforce Analytics - Interactive Streamlit Dashboard
========================================================
Run this with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI Workforce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA AND MODEL
# ============================================

@st.cache_data
def load_data():
    """Load the HR dataset"""
    try:
        df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        # Convert Attrition to numeric for calculations
        df['Attrition_Numeric'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure 'WA_Fn-UseC_-HR-Employee-Attrition.csv' is in the same directory.")
        st.stop()

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load("attrition_model.pkl")
        return model
    except FileNotFoundError:
        st.warning("⚠️ Model not found! Please train the model first by running 'hr_attrition_complete.py'")
        return None

# Load data
df = load_data()
model = load_model()

# ============================================
# HEADER
# ============================================

st.markdown('<h1 class="main-header">📊 AI-Powered Workforce Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR FILTERS
# ============================================

st.sidebar.header("🔍 Filter Employees")
st.sidebar.markdown("---")

# Department filter
departments = ["All"] + sorted(df["Department"].unique().tolist())
selected_department = st.sidebar.selectbox("📁 Department", departments)

# OverTime filter
overtime_options = ["All"] + sorted(df["OverTime"].unique().tolist())
selected_overtime = st.sidebar.selectbox("⏰ OverTime", overtime_options)

# Gender filter
gender_options = ["All"] + sorted(df["Gender"].unique().tolist())
selected_gender = st.sidebar.selectbox("👥 Gender", gender_options)

# Age range filter
min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("🎂 Age Range", min_age, max_age, (min_age, max_age))

# Apply filters
filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[filtered_df["Department"] == selected_department]

if selected_overtime != "All":
    filtered_df = filtered_df[filtered_df["OverTime"] == selected_overtime]

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]

filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Showing **{len(filtered_df)}** of **{len(df)}** employees")

# ============================================
# KEY PERFORMANCE INDICATORS (KPIs)
# ============================================

st.header("📈 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="👥 Total Employees",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df) - len(df)} from total"
    )

with col2:
    attrition_rate = filtered_df['Attrition_Numeric'].mean() * 100
    st.metric(
        label="⚠️ Attrition Rate",
        value=f"{attrition_rate:.1f}%",
        delta=f"{attrition_rate - (df['Attrition_Numeric'].mean() * 100):.1f}%"
    )

with col3:
    avg_salary = filtered_df['MonthlyIncome'].mean()
    st.metric(
        label="💰 Avg Monthly Income",
        value=f"${avg_salary:,.0f}",
        delta=f"${avg_salary - df['MonthlyIncome'].mean():,.0f}"
    )

with col4:
    avg_satisfaction = filtered_df['JobSatisfaction'].mean()
    st.metric(
        label="😊 Avg Job Satisfaction",
        value=f"{avg_satisfaction:.2f}/4",
        delta=f"{avg_satisfaction - df['JobSatisfaction'].mean():.2f}"
    )

st.markdown("---")

# ============================================
# VISUALIZATIONS
# ============================================

st.header("📊 Analytics & Insights")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["📊 Attrition Analysis", "💵 Income Analysis", "👥 Demographics", "⏰ Work Patterns"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Department")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        attrition_dept = filtered_df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        attrition_dept.plot(kind='bar', stacked=False, ax=ax1, color=['#2ecc71', '#e74c3c'])
        ax1.set_xlabel("Department")
        ax1.set_ylabel("Count")
        ax1.set_title("Employee Attrition by Department")
        ax1.legend(title="Attrition", labels=['No', 'Yes'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Attrition by Job Role")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        top_roles = filtered_df['JobRole'].value_counts().head(6).index
        role_data = filtered_df[filtered_df['JobRole'].isin(top_roles)]
        sns.countplot(data=role_data, y='JobRole', hue='Attrition', ax=ax2, palette=['#2ecc71', '#e74c3c'])
        ax2.set_xlabel("Count")
        ax2.set_ylabel("Job Role")
        ax2.set_title("Top 6 Job Roles - Attrition Comparison")
        ax2.legend(title="Attrition", labels=['No', 'Yes'])
        plt.tight_layout()
        st.pyplot(fig2)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income vs Job Satisfaction")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=filtered_df,
            x="MonthlyIncome",
            y="JobSatisfaction",
            hue="Attrition",
            palette=['#2ecc71', '#e74c3c'],
            alpha=0.6,
            ax=ax3
        )
        ax3.set_xlabel("Monthly Income ($)")
        ax3.set_ylabel("Job Satisfaction (1-4)")
        ax3.set_title("Income vs Job Satisfaction")
        ax3.legend(title="Attrition", labels=['No', 'Yes'])
        plt.tight_layout()
        st.pyplot(fig3)
    
    with col2:
        st.subheader("Income Distribution by Attrition")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        filtered_df.boxplot(column='MonthlyIncome', by='Attrition', ax=ax4)
        ax4.set_xlabel("Attrition")
        ax4.set_ylabel("Monthly Income ($)")
        ax4.set_title("Income Distribution by Attrition Status")
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig4)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        filtered_df['Age'].hist(bins=20, ax=ax5, color='steelblue', edgecolor='black', alpha=0.7)
        ax5.set_xlabel("Age")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Age Distribution of Employees")
        ax5.axvline(filtered_df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {filtered_df["Age"].mean():.1f}')
        ax5.legend()
        plt.tight_layout()
        st.pyplot(fig5)
    
    with col2:
        st.subheader("Gender Distribution")
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        gender_counts = filtered_df['Gender'].value_counts()
        ax6.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
        ax6.set_title("Employee Gender Distribution")
        plt.tight_layout()
        st.pyplot(fig6)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("OverTime Impact on Attrition")
        fig7, ax7 = plt.subplots(figsize=(8, 5))
        overtime_attrition = filtered_df.groupby(['OverTime', 'Attrition']).size().unstack(fill_value=0)
        overtime_attrition.plot(kind='bar', ax=ax7, color=['#2ecc71', '#e74c3c'])
        ax7.set_xlabel("OverTime")
        ax7.set_ylabel("Count")
        ax7.set_title("Impact of OverTime on Attrition")
        ax7.legend(title="Attrition", labels=['No', 'Yes'])
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig7)
    
    with col2:
        st.subheader("Work-Life Balance Distribution")
        fig8, ax8 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=filtered_df, x='WorkLifeBalance', hue='Attrition', ax=ax8, palette=['#2ecc71', '#e74c3c'])
        ax8.set_xlabel("Work-Life Balance (1-4)")
        ax8.set_ylabel("Count")
        ax8.set_title("Work-Life Balance vs Attrition")
        ax8.legend(title="Attrition", labels=['No', 'Yes'])
        plt.tight_layout()
        st.pyplot(fig8)

st.markdown("---")

# ============================================
# ATTRITION PREDICTION TOOL
# ============================================

if model is not None:
    st.header("🔮 Predict Employee Attrition Risk")
    st.markdown("Enter employee details to predict their attrition risk:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("👤 Age", min_value=18, max_value=65, value=30)
        income = st.number_input("💰 Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=100)
        distance = st.number_input("🏠 Distance From Home (km)", min_value=0, max_value=50, value=5)
    
    with col2:
        jobsat = st.slider("😊 Job Satisfaction (1-4)", 1, 4, 3)
        years = st.slider("📅 Years at Company", 0, 40, 3)
        overtime_input = st.selectbox("⏰ OverTime", ["Yes", "No"])
    
    with col3:
        wlb = st.slider("⚖️ Work Life Balance (1-4)", 1, 4, 3)
        envsat = st.slider("🌟 Environment Satisfaction (1-4)", 1, 4, 3)
    
    if st.button("🚀 Predict Attrition Risk", type="primary"):
        # Prepare sample input
        sample = pd.DataFrame({
            "Age": [age],
            "MonthlyIncome": [income],
            "DistanceFromHome": [distance],
            "JobSatisfaction": [jobsat],
            "YearsAtCompany": [years],
            "OverTime": [overtime_input],
            "WorkLifeBalance": [wlb],
            "EnvironmentSatisfaction": [envsat]
        })
        
        # Encode categorical
        sample = pd.get_dummies(sample, drop_first=True)
        
        # Align columns with training data
        train_cols = model.feature_names_in_
        for col in train_cols:
            if col not in sample.columns:
                sample[col] = 0
        sample = sample[train_cols]
        
        # Predict
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0]
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("### 🚨 HIGH ATTRITION RISK")
                st.markdown(f"**Probability of leaving:** {probability[1]:.1%}")
            else:
                st.success("### ✅ LOW ATTRITION RISK")
                st.markdown(f"**Probability of staying:** {probability[0]:.1%}")
        
        with col2:
            st.markdown("### 📊 Confidence Breakdown")
            st.progress(probability[0], text=f"Stay: {probability[0]:.1%}")
            st.progress(probability[1], text=f"Leave: {probability[1]:.1%}")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>📊 HR Workforce Analytics Dashboard | Powered by Machine Learning 🤖</p>
        <p>Built with Streamlit ❤️</p>
    </div>
""", unsafe_allow_html=True)
