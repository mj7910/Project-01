import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from streamlit_option_menu import option_menu
import base64


# ---------------------------------------------------
# Set up Streamlit Page
# ---------------------------------------------------
st.set_page_config(page_title="Cancer Support Dashboard", layout="wide")

# ---------------------------------------------------
# Load and Clean Data
# ---------------------------------------------------
# @st.cache_data
def load_data():
    df = pd.read_excel('data/Data.xlsx')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')

    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    if 'Grant_Req_Date' in df.columns:
        df['Grant_Req_Date'] = pd.to_datetime(df['Grant_Req_Date'], errors='coerce')
    if 'Payment_Submitted?' in df.columns:
        df['Payment_Submitted?'] = pd.to_datetime(df['Payment_Submitted?'], format='%m/%d/%Y', errors='coerce')
    if 'DOB' in df.columns:
        df['DOB'] = pd.to_datetime(df['DOB'], format='%m/%d/%Y', errors='coerce')

    text_cols = ['Pt_City', 'Pt_State', 'Referral_Source', 'Referred_By:', 'Type_of_Assistance_(CLASS)', 'Payment_Method', 'Notes']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    if 'Pt_Zip' in df.columns:
        df['Pt_Zip'] = df['Pt_Zip'].astype(str).str.strip()


    if 'DOB' in df.columns:
        df['Age'] = (pd.Timestamp.today() - df['DOB']).dt.days // 365

    if 'Grant_Req_Date' in df.columns and 'Payment_Submitted?' in df.columns and 'Request_Status' in df.columns:
        mask = (
            (df['Request_Status'].str.lower() == 'approved') &
            (pd.to_datetime(df['Payment_Submitted?'], errors='coerce').notnull()) &
            (~df['Payment_Submitted?'].astype(str).str.lower().str.contains("yes"))
        )
        df.loc[mask, 'Days_to_Payment'] = (
            pd.to_datetime(df.loc[mask, 'Payment_Submitted?'], errors='coerce') - 
            pd.to_datetime(df.loc[mask, 'Grant_Req_Date'], errors='coerce')
        ).dt.days

    return df

# Load the data
df = load_data()
# Home Page: Welcome and Overview
# ---------------------------------------------------

demo_df = df.copy()

# Clean data for demographics visualizations
demo_df['Gender'] = demo_df['Gender'].astype(str).str.strip().str.title()
demo_df['Gender'] = demo_df['Gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'Other')

if 'Race' in demo_df.columns:
    demo_df['Race'] = demo_df['Race'].astype(str).str.strip().str.title()
    demo_df['Race'] = demo_df['Race'].replace({
        'American Indian Or Alaskan Native': 'American Indian/Alaskan Native',
        'American Indian/Alaskan Native': 'American Indian/Alaskan Native',
        'American Indian Or Alaska Native': 'American Indian/Alaskan Native',
        'White': 'White',
        'white': 'White',
        'Missing': pd.NA,
        'missing': pd.NA,
        'Na': pd.NA
    })
    demo_df = demo_df.dropna(subset=['Race'])

if 'Insurance_Type' in demo_df.columns:
    demo_df['Insurance_Type'] = demo_df['Insurance_Type'].astype(str).str.strip().str.title()
    demo_df['Insurance_Type'] = demo_df['Insurance_Type'].replace({
        'Uninsured': 'Uninsured',
        'Unisured': 'Uninsured',
        'Unnsured': 'Uninsured',
        'Missing': pd.NA,
        'Unknown': pd.NA,
        '': pd.NA
    })
    demo_df = demo_df.dropna(subset=['Insurance_Type'])

anon_col = 'Patient_ID#'

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Ready for Review", "Support by Demographics", "Time to Payment", "Unused Grants and Assistance", "Impact Summary"],
        icons=["check-square", "bar-chart", "clock", "clipboard-data", "award"],
        menu_icon="cast",
        default_index=0,
        key="main_menu"
    )

# ---------------------------------------------------
# Page 1: Applications Ready for Review
# ---------------------------------------------------
if selected == "Ready for Review":
    st.title("📄 Applications Ready for Review")

    if 'Application_Signed?' in df.columns:
        df['Application_Signed?'] = (
            df['Application_Signed?']
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({'yes': 'Y', 'y': 'Y', 'no': 'N', 'n': 'N', 'missing': 'N', 'none': 'N'})
        )

    ready_df = df[df['Application_Signed?'] == 'Y'].copy()

    ready_df['Missing_Info'] = ready_df[['Pt_City', 'Pt_State', 'Pt_Zip']].apply(
        lambda x: any(str(v).lower() in ['missing', 'none', 'nan'] for v in x), axis=1
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Ready Applications", len(ready_df))
    col2.metric("⚠️ With Missing Info", ready_df['Missing_Info'].sum())
    if 'Grant_Req_Date' in ready_df.columns:
        latest_date = ready_df['Grant_Req_Date'].max()
        col3.metric("📅 Latest Request", latest_date.strftime('%Y-%m-%d') if pd.notna(latest_date) else "N/A")

    if 'Request_Status' in ready_df.columns:
        statuses = sorted(ready_df['Request_Status'].dropna().unique())
        selected_status = st.selectbox("Filter by Request Status", ["All"] + statuses, index=(["All"] + statuses).index("Pending") if "Pending" in statuses else 0)
        if selected_status != "All":
            ready_df = ready_df[ready_df['Request_Status'] == selected_status]

    st.download_button("📥 Download Ready Applications", ready_df.to_csv(index=False), file_name="ready_applications.csv")
    st.dataframe(ready_df)

    st.subheader("👤 View Patient Profile")

    def safe_get(value):
        if pd.isna(value) or str(value).strip().lower() in ["missing", "nan", "nat", "none"]:
            return "N/A"
        return str(value).strip()

    if not ready_df.empty:
        default_id = 240264
        available_ids = ready_df[anon_col].unique().tolist()

        if 'current_index' not in st.session_state:
            st.session_state['current_index'] = (
                available_ids.index(default_id) if default_id in available_ids else 0
            )

        # ✅ SAFETY CHECK FOR INDEX RANGE
        if st.session_state['current_index'] >= len(available_ids):
            st.session_state['current_index'] = 0

        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.button("⬅️ Previous") and st.session_state['current_index'] > 0:
                st.session_state['current_index'] -= 1
        with col_next:
            if st.button("Next ➡️") and st.session_state['current_index'] < len(available_ids) - 1:
                st.session_state['current_index'] += 1

        selected_id = st.selectbox("Select Patient ID", available_ids, index=st.session_state['current_index'])
        st.session_state['current_index'] = available_ids.index(selected_id)

        profile = ready_df[ready_df[anon_col] == selected_id]
        if not profile.empty:
            row = profile.iloc[0]

            gender = safe_get(row.get('Gender')).capitalize()
            age = safe_get(row.get('Age'))
            race = safe_get(row.get('Race'))
            insurance = safe_get(row.get('Insurance_Type'))
            amount = f"${row.get('Amount', 0):,.2f}"
            balance = f"${row.get('Remaining_Balance', 0):,.2f}"
            grant_date = safe_get(row.get('Grant_Req_Date'))
            phone = safe_get(row.get('Phone_Number'))
            language = safe_get(row.get('Language'))
            marital = safe_get(row.get('Marital_Status'))
            assistance = safe_get(row.get('Type_of_Assistance_(CLASS)'))

            city = safe_get(row.get('Pt_City')).title()
            state = safe_get(row.get('Pt_State')).upper()
            zip_code = safe_get(row.get('Pt_Zip'))
            full_address = f"{city}, {state} {zip_code}" if zip_code != "N/A" else f"{city}, {state}"

            if gender.lower() == "female":
                gender_icon_url = "https://cdn-icons-png.flaticon.com/512/236/236832.png"
            elif gender.lower() == "male":
                gender_icon_url = "https://cdn-icons-png.flaticon.com/512/236/236831.png"
            else:
                gender_icon_url = "https://cdn-icons-png.flaticon.com/512/149/149071.png"

            st.markdown(f"""
            <div style=\"background-color:#ffffff10;padding:25px;border-radius:16px;margin-top:10px;color:#e8e8e8;display:flex;gap:40px\">
                <div style=\"flex-shrink:0;\">
                    <img src=\"{gender_icon_url}\" width=\"140\" style=\"border-radius:12px;border:2px solid #aaa\"/>
                </div>
                <div style=\"flex-grow:1;display:grid;grid-template-columns:repeat(3, 1fr);gap:15px;\">
                    <div>
                        <p><strong>🧑 Gender:</strong><br> {gender}</p>
                        <p><strong>🧍 Age:</strong><br> {age}</p>
                        <p><strong>🌎 Race:</strong><br> {race}</p>
                        <p><strong>🛡 Insurance:</strong><br> {insurance}</p>
                    </div>
                    <div>
                        <p><strong>💰 Requested:</strong><br> {amount}</p>
                        <p><strong>💳 Balance:</strong><br> {balance}</p>
                        <p><strong>📅 Request Date:</strong><br> {grant_date}</p>
                    </div>
                    <div>
                        <p><strong>📍 Location:</strong><br> {full_address}</p>
                        <p><strong>📞 Phone:</strong><br> {phone}</p>
                        <p><strong>🗣 Language:</strong><br> {language}</p>
                        <p><strong>💍 Marital:</strong><br> {marital}</p>
                        <p><strong>🧾 Assistance:</strong><br> {assistance}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------
# ---------------------------------------------------
# Page 2: Support by Demographics (Styled Visualizations)
# ---------------------------------------------------
if selected == "Support by Demographics":
    st.title("📊 Support by Demographics")

    demo_df = df.copy()
    demo_df['Gender'] = demo_df['Gender'].astype(str).str.strip().str.title()
    demo_df['Gender'] = demo_df['Gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'Other')

    if 'Race' in demo_df.columns:
        demo_df['Race'] = demo_df['Race'].astype(str).str.strip().str.title()
        demo_df['Race'] = demo_df['Race'].replace({
            'American Indian Or Alaskan Native': 'American Indian/Alaskan Native',
            'American Indian/Alaskan Native': 'American Indian/Alaskan Native',
            'American Indian Or Alaska Native': 'American Indian/Alaskan Native',
            'White': 'White',
            'white': 'White',
            'Missing': pd.NA,
            'missing': pd.NA,
            'Na': pd.NA
        })
        demo_df = demo_df.dropna(subset=['Race'])

    if 'Insurance_Type' in demo_df.columns:
        demo_df['Insurance_Type'] = demo_df['Insurance_Type'].astype(str).str.strip().str.title()
        demo_df['Insurance_Type'] = demo_df['Insurance_Type'].replace({
            'Uninsured': 'Uninsured',
            'Unisured': 'Uninsured',
            'Unnsured': 'Uninsured',
            'Missing': pd.NA,
            'Unknown': pd.NA,
            '': pd.NA
        })
        demo_df = demo_df.dropna(subset=['Insurance_Type'])

    if 'Total_Household_Gross_Monthly_Income' in demo_df.columns:
        demo_df['Total_Household_Gross_Monthly_Income'] = pd.to_numeric(demo_df['Total_Household_Gross_Monthly_Income'], errors='coerce')

    # Filters
    st.sidebar.header("Filter Demographics")
    age_filter = st.sidebar.slider("Select Age Range", 0, 100, (0, 100))
    gender_filter = st.sidebar.multiselect("Select Gender", options=demo_df['Gender'].unique(), default=list(demo_df['Gender'].unique()))
    city_options = sorted(demo_df['Pt_City'].dropna().unique())
    city_filter = st.sidebar.multiselect("Select City", options=city_options, default=[])
    income_min = int(demo_df['Total_Household_Gross_Monthly_Income'].min()) if 'Total_Household_Gross_Monthly_Income' in demo_df.columns else 0
    income_max = int(demo_df['Total_Household_Gross_Monthly_Income'].max()) if 'Total_Household_Gross_Monthly_Income' in demo_df.columns else 10000
    income_filter = st.sidebar.slider("Income Range", income_min, income_max, (income_min, income_max))

    filtered_df = demo_df[
        (demo_df['Age'].between(age_filter[0], age_filter[1])) &
        (demo_df['Gender'].isin(gender_filter)) &
        ((demo_df['Pt_City'].isin(city_filter)) if city_filter else True) &
        (demo_df['Total_Household_Gross_Monthly_Income'].between(income_filter[0], income_filter[1]))
    ]

    # Breakdown and Chart Selection
    column_name_map = {
        'Gender': 'Gender',
        'Race': 'Race',
        'Insurance_Type': 'Insurance Type',
        'Marital_Status': 'Marital Status',
        'Pt_City': 'City',
        'Age': 'Age'
    }
    breakdown_options = list(column_name_map.values())
    breakdown_label_to_col = {v: k for k, v in column_name_map.items()}
    selected_label = st.selectbox("Breakdown Support By", breakdown_options)
    chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart", "Donut Chart", "Heatmap"])
    breakdown_column = breakdown_label_to_col[selected_label]

    if breakdown_column in filtered_df.columns:
        support_breakdown = filtered_df.groupby(breakdown_column)['Amount'].sum().sort_values(ascending=False).reset_index()

        if chart_type == "Bar Chart":
            fig = px.bar(support_breakdown, x=breakdown_column, y='Amount', color=breakdown_column)
        elif chart_type == "Pie Chart":
            fig = px.pie(support_breakdown, names=breakdown_column, values='Amount', title=f"Support by {selected_label}")
        elif chart_type == "Donut Chart":
            fig = px.pie(support_breakdown, names=breakdown_column, values='Amount', hole=0.4, title=f"Support by {selected_label}")
        elif chart_type == "Heatmap":
            fig = ff.create_annotated_heatmap(
                z=[[v] for v in support_breakdown['Amount']],
                x=["Total Support"],
                y=support_breakdown[breakdown_column].astype(str).tolist(),
                colorscale='Viridis',
                showscale=True
            )
            fig.update_layout(height=400 + len(support_breakdown)*20)

        fig.update_layout(
            title={"text": f"Support Breakdown by {selected_label}", "x": 0.5, "xanchor": "center"},
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(family='Arial', color='white', size=16),
            xaxis=dict(showgrid=False, showline=True, linewidth=2, linecolor='white'),
            yaxis=dict(showgrid=False, showline=True, linewidth=2, linecolor='white')
        )
        st.plotly_chart(fig)
    else:
        st.warning("Please select a valid breakdown field.")

# ---------------------------------------------------
# Page 3: Time to Payment Analysis
# ---------------------------------------------------

if selected == "Time to Payment":
    if 'Days_to_Payment' in df.columns:
        st.title("⏳ Time from Request to Payment")

        valid_df = df.dropna(subset=['Days_to_Payment'])

        avg_days = round(valid_df['Days_to_Payment'].mean(), 2)
        median_days = valid_df['Days_to_Payment'].median()
        min_days = valid_df['Days_to_Payment'].min()
        max_days = valid_df['Days_to_Payment'].max()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Days", avg_days)
        col2.metric("Median Days", median_days)
        col3.metric("Shortest Wait", min_days)
        col4.metric("Longest Wait", max_days)

        st.markdown("---")

        st.subheader("📊 Distribution of Time to Payment")
        fig = px.histogram(valid_df, x='Days_to_Payment', nbins=30, title="Distribution of Days to Payment")
        fig.update_layout(
            title={"x": 0.5},
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(family='Arial', size=16, color='white'),
            xaxis=dict(showgrid=False, showline=True, linewidth=2, linecolor='white', title='Days to Payment'),
            yaxis=dict(showgrid=False, showline=True, linewidth=2, linecolor='white', title='Count')
        )
        st.plotly_chart(fig)

        st.markdown("---")

        st.subheader("📈 Average Days to Payment by Demographics")
        tab_labels = {
            "Race": "🧬 Race",
            "Insurance_Type": "🛡 Insurance Type",
            "Gender": "🧑 Gender",
            "Pt_City": "🏙 City"
        }

        valid_df['Race'] = valid_df['Race'].astype(str).str.strip().str.title()
        valid_df = valid_df[valid_df['Race'].str.lower() != 'missing']

        valid_df['Insurance_Type'] = valid_df['Insurance_Type'].astype(str).str.strip().str.title()
        valid_df['Insurance_Type'] = valid_df['Insurance_Type'].replace({
            'Uninsured': 'Uninsured',
            'Unisured': 'Uninsured',
            'Unnsured': 'Uninsured',
            'Medicare': 'Medicare',
            'MEdicare': 'Medicare',
            'Medicaid': 'Medicaid',
            'medicaid': 'Medicaid'
        })

        valid_df['Gender'] = valid_df['Gender'].astype(str).str.strip().str.title()
        valid_df = valid_df[valid_df['Gender'].str.lower() != 'missing']

        valid_df['Pt_City'] = valid_df['Pt_City'].astype(str).str.strip().str.title()
        valid_df = valid_df[valid_df['Pt_City'].str.lower() != 'nan']

        tabs = st.tabs([label for label in tab_labels.values()])

        for idx, col in enumerate(tab_labels.keys()):
            with tabs[idx]:
                if col in valid_df.columns:
                    demo_avg = valid_df.groupby(col)['Days_to_Payment'].mean().sort_values().reset_index()
                    fig = px.bar(
                        demo_avg,
                        x=col,
                        y='Days_to_Payment',
                        text='Days_to_Payment',
                        title=f"Average Time to Payment by {tab_labels[col]}"
                    )
                    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                    fig.update_layout(
                        title={"x": 0.5},
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(family='Arial', size=16, color='white'),
                        xaxis=dict(
                            title=col.replace("_", " "),
                            tickangle=45,
                            showgrid=False,
                            showline=True,
                            linewidth=2,
                            linecolor='white'
                        ),
                        yaxis=dict(
                            title='Average Days',
                            showgrid=False,
                            showline=True,
                            linewidth=2,
                            linecolor='white'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("'Days_to_Payment' column not found or contains insufficient data.")

# ---------------------------------------------------
# Page 4: Unused Grants and Assistance Types
# ---------------------------------------------------


if selected == "Unused Grants and Assistance":
    st.title("📋 Unused Grants and Assistance Types")

    if 'Remaining_Balance' in df.columns and 'App_Year' in df.columns:
        approved_df = df[(df['Request_Status'].str.lower() == 'approved') &
                         df['Remaining_Balance'].notnull() &
                         df['Amount'].notnull() &
                         (df['Amount'] > 0)]

        unused_df = approved_df[approved_df['Remaining_Balance'] > 0]

        st.subheader("📈 Patients with Unused Support by Year")
        count_by_year = unused_df.groupby('App_Year')['Patient_ID#'].nunique().reset_index()
        count_by_year.columns = ['Year', 'Patients']
        fig = px.bar(count_by_year, x='Year', y='Patients', text='Patients', color='Patients', color_continuous_scale='Blues')
        fig.update_traces(textposition='outside')
        fig.update_layout(
            title={"text": "Number of Patients with Unused Grants by Year", "x": 0.5},
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(family='Arial', color='white', size=16),
            xaxis=dict(title="Application Year", showgrid=False, showline=True, linewidth=2, linecolor='white'),
            yaxis=dict(title="Number of Patients", showgrid=False, showline=True, linewidth=2, linecolor='white')
        )
        st.plotly_chart(fig)

        st.subheader("💰 Average Support Amount by Assistance Type")
        avg_support = approved_df.groupby('Type_of_Assistance_(CLASS)')['Amount'].mean().reset_index().dropna()
        avg_support.columns = ['Assistance Type', 'Support Amount ($)']
        fig2 = px.bar(avg_support, x='Support Amount ($)', y='Assistance Type', orientation='h', text='Support Amount ($)', color='Support Amount ($)', color_continuous_scale='Cividis')
        fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig2.update_layout(
            title={"text": "Average Support Amount by Assistance Type", "x": 0.5},
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(family='Arial', color='white', size=16),
            xaxis=dict(title="Average Amount ($)", showgrid=False, showline=True, linewidth=2, linecolor='white'),
            yaxis=dict(title="Assistance Type", showgrid=False, showline=True, linewidth=2, linecolor='white')
        )
        st.plotly_chart(fig2)

        

        st.download_button("📥 Download Unused Grants Data", unused_df.to_csv(index=False), file_name="unused_grants_data.csv")

    else:
        st.warning("Required columns for this analysis are missing.")






def add_footer_logo():
    file_path = "data/long_logo.png"  # Replace with your correct image path
    with open(file_path, "rb") as f:
        img_data = f.read()
        b64 = base64.b64encode(img_data).decode()

    footer_html = f"""
    <div style="margin-top: 50px; padding: 20px 0; background-color: #0e1117; text-align: center;">
        <img src="data:image/png;base64,{b64}" alt="Hope Foundation Logo"
            style="max-width: 100%; width: 700px; height: auto;" />
        <p style="color: #aaa; font-size: 16px; margin-top: 15px;">
            Empowering Nebraska families through <strong>compassion</strong>, <strong>support</strong>, and <strong>hope</strong>.
        </p>
        <p style="color: #77baff; font-size: 14px; margin-top: 5px;">
            <a href="https://ncshopefoundation.org/" style="color: #77baff; text-decoration: none;" target="_blank">
                Visit the NCS HOPE Foundation Website →
            </a>
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
    
# Page 5: Impact Summary for Stakeholders
# ---------------------------------------------------

if selected == "Impact Summary":
    st.title("🏆 Foundation Impact Summary")

    approved_df = df[df['Request_Status'].str.lower() == 'approved'].copy()

    # Clean assistance types and city names
    approved_df['Type_of_Assistance_(CLASS)'] = approved_df['Type_of_Assistance_(CLASS)'].astype(str).str.strip().str.title()
    approved_df['Pt_City'] = approved_df['Pt_City'].replace('nan', pd.NA).dropna()

    # Key Stats
    total_patients = approved_df['Patient_ID#'].nunique()
    total_support = approved_df['Amount'].sum()
    avg_support = approved_df['Amount'].mean()
    common_assistance = approved_df['Type_of_Assistance_(CLASS)'].mode().iloc[0] if 'Type_of_Assistance_(CLASS)' in approved_df else 'N/A'
    avg_days = approved_df['Days_to_Payment'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Patients Helped", f"{total_patients}")
    col2.metric("💸 Total Support Given", f"${total_support:,.2f}")
    col3.metric("📊 Average Support", f"${avg_support:,.2f}")

    col4, col5 = st.columns(2)
    col4.metric("🎯 Most Common Assistance Type", common_assistance)
    col5.metric("⏱️ Avg Days to Payment", f"{avg_days:.1f} days")

    # Support Trend
    if 'App_Year' in approved_df.columns:
        trend_df = approved_df.groupby('App_Year').agg({
            'Amount': 'sum',
            'Patient_ID#': pd.Series.nunique
        }).reset_index().rename(columns={
            'App_Year': 'Year',
            'Amount': 'Total Support',
            'Patient_ID#': 'Unique Patients'
        })

        st.subheader("📈 Support Trend Over Years")
        fig = px.line(trend_df, x='Year', y='Total Support', markers=True, title="Total Support Given by Year")
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(family='Arial', color='white', size=16),
            xaxis=dict(title='Year', showline=True, linewidth=2, linecolor='white'),
            yaxis=dict(title='Total Support ($)', showline=True, linewidth=2, linecolor='white')
        )
        st.plotly_chart(fig)

    # Assistance Type Summary (Top 3)
    st.subheader("💼 Top 3 Supported Assistance Types")
    if 'Type_of_Assistance_(CLASS)' in approved_df.columns:
        top_assist_support = approved_df.groupby('Type_of_Assistance_(CLASS)').agg(
            Total_Support=('Amount', 'sum'),
            Patient_Count=('Patient_ID#', pd.Series.nunique)
        ).sort_values('Total_Support', ascending=False).head(3).reset_index()
        top_assist_support.columns = ['Assistance Type', 'Total Support ($)', 'Number of Patients']
        fig2 = px.bar(top_assist_support, x='Total Support ($)', y='Assistance Type', orientation='h',
                      color='Total Support ($)', color_continuous_scale='Inferno')
        fig2.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(family='Arial', color='white', size=16),
            xaxis=dict(title='Support Amount ($)', showline=True, linewidth=2, linecolor='white'),
            yaxis=dict(title='Assistance Type', showline=True, linewidth=2, linecolor='white')
        )
        st.plotly_chart(fig2)

    # Top 3 Cities

    st.subheader("📍 Top 3 Cities by Total Support")
    if 'Pt_City' in approved_df.columns:
        city_filtered_df = approved_df[
            approved_df['Pt_City'].notna() &
            (~approved_df['Pt_City'].str.lower().isin(['missing', 'nan']))
        ]
        top_cities = city_filtered_df.groupby('Pt_City')['Amount'].sum().sort_values(ascending=False).head(3).reset_index()
        top_cities.columns = ['City', 'Total Support ($)']
        fig3 = px.bar(
            top_cities,
            x='Total Support ($)',
            y='City',
            orientation='h',
            color='Total Support ($)',
            color_continuous_scale='Blues'
        )
        fig3.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(family='Arial', color='white', size=16),
            xaxis=dict(title='Support Amount ($)', showline=True, linewidth=2, linecolor='white'),
            yaxis=dict(title='City', showline=True, linewidth=2, linecolor='white')
        )
        st.plotly_chart(fig3)
        add_footer_logo()



