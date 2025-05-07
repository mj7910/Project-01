import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

# ---------------------------------------------------
# Set up Streamlit Page
# ---------------------------------------------------
st.set_page_config(page_title="Cancer Support Dashboard", layout="wide")

# ---------------------------------------------------
# Load and Clean Dat
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel('data/Data.xlsx')
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')

    # Fix Amount column
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Parse dates
    if 'Grant_Req_Date' in df.columns:
        df['Grant_Req_Date'] = pd.to_datetime(df['Grant_Req_Date'], errors='coerce')
    if 'Payment_Submitted?' in df.columns:
        df['Payment_Submitted?'] = pd.to_datetime(df['Payment_Submitted?'], format='%m/%d/%Y', errors='coerce')
    if 'DOB' in df.columns:
        df['DOB'] = pd.to_datetime(df['DOB'], format='%m/%d/%Y', errors='coerce')

    # Clean text columns
    text_cols = ['Pt_City', 'Pt_State', 'Referral_Source', 'Referred_By:', 'Type_of_Assistance_(CLASS)', 'Payment_Method', 'Notes']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Add Age and Days_to_Payment
    if 'DOB' in df.columns:
        df['Age'] = (pd.Timestamp.today() - df['DOB']).dt.days // 365
    if 'Grant_Req_Date' in df.columns and 'Payment_Submitted?' in df.columns:
        df['Days_to_Payment'] = (df['Payment_Submitted?'] - df['Grant_Req_Date']).dt.days

    return df

# Load the data
df = load_data()

# ---------------------------------------------------
# Set ID Column
# ---------------------------------------------------
anon_col = 'Patient_ID#'  # Based on your uploaded file

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Ready for Review", "Support by Demographics", "Time to Payment", "Unused Grants and Assistance", "Impact Summary"],
        icons=["check-square", "bar-chart", "clock", "clipboard-data", "award"],
        menu_icon="cast",
        default_index=0
    )

# ---------------------------------------------------
# Page 1: Reafddy for Review (Based Only on Application Signed?)
# ---------------------------------------------------
if selected == "Ready for Review":
    st.title("📄 Applications Ready for Review")

    signed_filter = st.radio("Filter by Application Signed?", ("All", "Signed", "Not Signed"))

    if signed_filter == "Signed":
        ready_df = df[df['Application_Signed?'] == 'Y']
    elif signed_filter == "Not Signed":
        ready_df = df[df['Application_Signed?'] != 'Y']
    else:
        ready_df = df

    st.metric("Total Applications", len(ready_df))
    st.dataframe(ready_df)

# ---------------------------------------------------
# Page 2: Support by Demographics
# ---------------------------------------------------
elif selected == "Support by Demographics":
    st.title("📊 Support by Demographics")

    if 'Gender' in df.columns:
        st.subheader("Support by Gender")
        support_gender = df.groupby('Gender')['Amount'].sum().sort_values()
        st.bar_chart(support_gender)

    if 'Insurance_Type' in df.columns:
        st.subheader("Support by Insurance Type")
        support_insurance = df.groupby('Insurance_Type')['Amount'].sum().sort_values()
        st.bar_chart(support_insurance)

    if 'Pt_City' in df.columns:
        st.subheader("Support by City (Top 20)")
        support_city = df.groupby('Pt_City')['Amount'].sum().sort_values(ascending=False).head(20)
        st.bar_chart(support_city)

    if 'Total_Household_Gross_Monthly_Income' in df.columns:
        st.subheader("Support by Household Income")
        income_support = df.groupby('Total_Household_Gross_Monthly_Income')['Amount'].sum()
        st.line_chart(income_support)

# ---------------------------------------------------
# Page 3: Time to Payment
# ---------------------------------------------------
elif selected == "Time to Payment":
    st.title("⏳ Time from Request to Payment")

    if 'Days_to_Payment' in df.columns:
        st.write("Histogram showing days between Grant Request and Payment Submission.")
        fig = px.histogram(df.dropna(subset=['Days_to_Payment']), x='Days_to_Payment', nbins=30, title="Distribution of Days to Payment")
        st.plotly_chart(fig)

# ---------------------------------------------------
# Page 4: Unused Grants and Assistance Types
# ---------------------------------------------------
elif selected == "Unused Grants and Assistance":
    st.title("📋 Unused Grants and Assistance Types")

    if 'Remaining_Balance' in df.columns:
        unused_grants = df[df['Remaining_Balance'] > 0]
        st.subheader("Patients with Remaining Grant Balance")

        if anon_col in df.columns:
            st.dataframe(unused_grants[[anon_col, 'App_Year', 'Remaining_Balance']])

        st.subheader("Average Grant Amount by Assistance Type")
        if 'Type_of_Assistance_(CLASS)' in df.columns:
            avg_assistance = df.groupby('Type_of_Assistance_(CLASS)')['Amount'].mean().sort_values(ascending=False)
            st.bar_chart(avg_assistance)

# ---------------------------------------------------
# Page 5: Impact Summary
# ---------------------------------------------------
elif selected == "Impact Summary":
    st.title("🏆 Foundation Impact Summary")

    if anon_col in df.columns:
        total_patients = df[anon_col].nunique()
    else:
        total_patients = df.shape[0]

    total_amount = df['Amount'].sum()
    avg_grant = df['Amount'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients Helped", total_patients)
    col2.metric("Total Support Given", f"${total_amount:,.2f}")
    col3.metric("Average Grant Amount", f"${avg_grant:,.2f}")

    st.success("Amazing work helping families across Nebraska!")
