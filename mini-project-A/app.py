import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Titanic Exploratory Data Analysis Dashboard",
    page_icon="🚢",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    # Construct path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'Titanic-Dataset.csv')
    return pd.read_csv(data_path)

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# 2. Header
st.title("🚢 Titanic Exploratory Data Analysis Dashboard")
st.markdown("Welcome to the interactive EDA dashboard for the Titanic dataset. "
            "Explore passenger demographics, survival rates, and key findings.")
st.markdown("---")

# 4. Sidebar - Filter
st.sidebar.header("Filters")
pclass_filter = st.sidebar.selectbox(
    "Select Passenger Class:",
    ("All", "First Class", "Second Class", "Third Class")
)

# Apply filter
if pclass_filter == "First Class":
    filtered_df = df[df['Pclass'] == 1]
elif pclass_filter == "Second Class":
    filtered_df = df[df['Pclass'] == 2]
elif pclass_filter == "Third Class":
    filtered_df = df[df['Pclass'] == 3]
else:
    filtered_df = df

# 3. KPI Row
st.header("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_passengers = len(filtered_df)
survival_rate = (filtered_df['Survived'].mean() * 100) if total_passengers > 0 else 0

female_df = filtered_df[filtered_df['Sex'] == 'female']
female_survival_rate = (female_df['Survived'].mean() * 100) if len(female_df) > 0 else 0

first_class_df = filtered_df[filtered_df['Pclass'] == 1]
first_class_survival_rate = (first_class_df['Survived'].mean() * 100) if len(first_class_df) > 0 else 0

with col1:
    st.metric(label="Total Passengers", value=f"{total_passengers:,}")
with col2:
    st.metric(label="Survival Rate", value=f"{survival_rate:.2f}%")
with col3:
    st.metric(label="Female Survival Rate", value=f"{female_survival_rate:.2f}%")
with col4:
    st.metric(label="First Class Survival Rate", value=f"{first_class_survival_rate:.2f}%")

st.markdown("---")

# 5. Interactive Plotly Charts
st.header("Interactive Visualizations")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Chart 1: Survival by Sex
    survival_sex_df = filtered_df.groupby(['Sex', 'Survived']).size().reset_index(name='Count')
    survival_sex_df['Survived'] = survival_sex_df['Survived'].map({0: 'Died', 1: 'Survived'})
    fig1 = px.bar(survival_sex_df, x='Sex', y='Count', color='Survived', 
                  barmode='group', title="Survival by Sex",
                  color_discrete_sequence=['#ef553b', '#00cc96'])
    st.plotly_chart(fig1, use_container_width=True)
    st.info("**Insight:** Female passengers had a significantly higher survival rate than males.")

with chart_col2:
    # Chart 2: Survival by Passenger Class
    survival_pclass_df = filtered_df.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
    survival_pclass_df['Survived'] = survival_pclass_df['Survived'].map({0: 'Died', 1: 'Survived'})
    survival_pclass_df['Pclass'] = survival_pclass_df['Pclass'].astype(str)
    fig2 = px.bar(survival_pclass_df, x='Pclass', y='Count', color='Survived', 
                  barmode='group', title="Survival by Passenger Class",
                  color_discrete_sequence=['#ef553b', '#00cc96'])
    st.plotly_chart(fig2, use_container_width=True)
    st.info("**Insight:** Passengers travelling in First Class had the highest probability of survival.")

st.markdown("<br>", unsafe_allow_html=True)

# Chart 3: Fare Distribution (Full width)
fig3 = px.histogram(filtered_df, x='Fare', nbins=50, title="Fare Distribution",
                    color_discrete_sequence=['#636efa'])
st.plotly_chart(fig3, use_container_width=True)
st.info("**Insight:** Fare distribution is highly right-skewed with several premium ticket outliers.")

st.markdown("---")

# 6. Dataset Summary Section & 7. Additional Insights
st.header("Analysis Summary")
summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.subheader("Dataset Overview")
    with st.container(border=True):
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        st.write(f"**Feature Count:** {df.shape[1]}")

with summary_col2:
    st.subheader("Key Findings from EDA")
    with st.container(border=True):
        st.markdown(
            """
            - **Overall Survival Rate:** Only **38.38%** of passengers survived the disaster.
            - **Highest Survival by Gender:** Female passengers had an overwhelming survival rate of **74.2%**.
            - **Highest Survival by Class:** First-class passengers were prioritized, resulting in a **63.0%** survival rate.
            - **Age Group Observations:** Children had the highest survival chances (**58.0%**), while seniors had the lowest.
            - **Missing Value Observations:** The 'Cabin' feature is missing **77.1%** of its data and is recommended to be dropped, while 'Age' requires median imputation.
            """
        )

st.markdown("---")

# 8. Footer
st.markdown(
    "<div style='text-align: center; color: gray;'>Created for FWC Week-2 Mini Project A</div>", 
    unsafe_allow_html=True
)
