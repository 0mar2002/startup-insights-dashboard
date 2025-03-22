import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Page config
st.set_page_config(page_title="Startup Insights Dashboard", layout="wide")

# Load dataset
df = pd.read_csv('data/startup_data.csv')
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Derived columns
df['revenue_to_funding'] = df['revenue_(m_usd)'] / df['funding_amount_(m_usd)']
df['valuation_per_employee'] = df['valuation_(m_usd)'] / df['employees']
df['is_unicorn'] = df['valuation_(m_usd)'].apply(lambda x: 1 if x >= 1000 else 0)

# Unicorn score calculation
features = df[['valuation_(m_usd)', 'revenue_to_funding', 'profitable', 'market_share_(%)', 'valuation_per_employee']].copy()
features.replace([float('inf'), -float('inf')], 0, inplace=True)
features.fillna(0, inplace=True)
scaler = MinMaxScaler()
normalized = scaler.fit_transform(features)
normalized_df = pd.DataFrame(normalized, columns=features.columns)
weights = {
    'valuation_(m_usd)': 0.30,
    'revenue_to_funding': 0.25,
    'profitable': 0.20,
    'market_share_(%)': 0.15,
    'valuation_per_employee': 0.10
}
df['unicorn_score'] = sum(normalized_df[col] * weight for col, weight in weights.items())

# Sidebar Filters
st.sidebar.title("📊 Filters")
selected_industry = st.sidebar.selectbox("Filter by Industry", ["All"] + sorted(df['industry'].unique()))
min_score = st.sidebar.slider("Minimum Unicorn Score", 0.0, 1.0, 0.5, 0.01)

# Filtered Data
filtered_df = df.copy()
if selected_industry != "All":
    filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
filtered_df = filtered_df[filtered_df['unicorn_score'] >= min_score]

# Tabs
st.title("🚀 Startup Insights Dashboard")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Overview", "🔮 Unicorn Trends", "💸 Funding vs Valuation", "🧠 Industry Insights", "📉 Profitability", "📝 Takeaways"])

# Overview Tab
with tab1:
    st.header("📌 Ecosystem Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Startups", df.shape[0])
    with col2:
        st.metric("Unicorns (Valuation ≥ $1B)", df[df['is_unicorn'] == 1].shape[0])
    with col3:
        st.metric("Avg. Unicorn Score", f"{round(df['unicorn_score'].mean(), 2)}", help="The closer to 1, the higher the potential to become a unicorn.")

    st.markdown("---")
    st.subheader("🏆 Top 3 Startups by Unicorn Score")
    top3 = df.sort_values(by='unicorn_score', ascending=False).head(3)
    st.dataframe(top3[['startup_name', 'industry', 'unicorn_score', 'valuation_(m_usd)', 'revenue_to_funding', 'market_share_(%)', 'profitable']])
    st.caption("These startups show strong valuation, funding efficiency, and market traction. Metrics closer to 1 indicate stronger unicorn potential.")

# Unicorn Trends Tab
with tab2:
    st.header("🦄 Top Predicted Unicorns")
    st.dataframe(filtered_df[['startup_name', 'industry', 'unicorn_score', 'funding_amount_(m_usd)', 'valuation_(m_usd)']].sort_values(by='unicorn_score', ascending=False).head(10))
    st.caption("Unicorn Score is calculated based on valuation, revenue-to-funding ratio, profitability, market share, and valuation per employee.")

    st.subheader("🧪 Unicorn Distribution by Industry")
    unicorn_counts = df[df['is_unicorn'] == 1]['industry'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(unicorn_counts.values, labels=unicorn_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    st.caption("This pie chart shows the proportion of unicorns found in each industry.")

# Funding vs Valuation Tab
with tab3:
    st.header("💰 Funding vs Valuation")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='funding_amount_(m_usd)', y='valuation_(m_usd)', hue='industry', ax=ax)
    sns.regplot(data=filtered_df, x='funding_amount_(m_usd)', y='valuation_(m_usd)', scatter=False, ax=ax, color='black')
    ax.set_title("Funding Amount vs Valuation with Regression Line")
    st.pyplot(fig)
    st.info("The regression line (in black) indicates the average trend — generally, more funding correlates with higher valuation, but the spread shows that not all startups convert funding into value equally. Color-coding by industry reveals how some sectors perform above or below the trend.")

# Industry Insights Tab
with tab4:
    st.header("📊 Funding Efficiency by Industry")
    df['valuation_to_funding'] = df['valuation_(m_usd)'] / df['funding_amount_(m_usd)']
    industry_efficiency = df.groupby('industry')['valuation_to_funding'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=industry_efficiency.values, y=industry_efficiency.index, ax=ax)
    ax.set_xlabel("Valuation to Funding Ratio")
    st.pyplot(fig)
    st.caption("Industries with a higher valuation-to-funding ratio are using their capital more efficiently. This helps identify sectors where investments are more likely to yield higher company value.")

# Profitability Tab
with tab5:
    st.header("📉 Profitability vs Funding Rounds")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='funding_rounds', y='funding_amount_(m_usd)', hue='profitable', ax=ax)
    ax.set_title("Profitability by Funding Rounds")
    st.pyplot(fig)
    st.warning("This boxplot shows how funding amounts are distributed across different rounds, split by whether a startup is profitable. Taller boxes = more variation. If profitable and unprofitable startups overlap heavily, it means funding rounds alone don’t predict profitability.")

# Takeaways Tab
with tab6:
    st.header("📝 Key Takeaways")
    st.markdown("""
    ### 🔮 Unicorn Prediction
    - Only a small percentage of startups become unicorns. ([See: **Unicorn Trends** tab])
    - Industries like **AI** and **FinTech** show higher unicorn creation rates 🟢.

    ### 💸 Funding vs Valuation
    - High funding does **not** guarantee high valuation ❌. ([See: **Funding vs Valuation** tab])
    - Regression line shows a general upward trend, but wide spread highlights inconsistent value creation.
    - **Color-coded industries** show how some sectors perform above or below the trend.

    ### 🧠 Capital Efficiency
    - Certain industries show stronger **capital efficiency**, such as **HealthTech** 🟢. ([See: **Industry Insights** tab])
    - The valuation-to-funding ratio helps spot where money is well spent.

    ### 📉 Profitability Trends
    - Many startups raise millions without becoming profitable ❌. ([See: **Profitability** tab])
    - More funding rounds ≠ more profit. External factors like burn rate and market timing matter.

    ---
    ✅ Use the filters to explore different industries and spot high-potential startups by Unicorn Score.
    """)