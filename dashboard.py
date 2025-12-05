import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ---------------------------------------------------
# Streamlit Page Config & Layout
# ---------------------------------------------------
st.set_page_config(
    page_title="Donation Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("💰 Donation Dashboard")
st.markdown("PayPal donations by program - **No donor information displayed**")

# ---------------------------------------------------
# Sidebar – only date + program controls
# ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("📅 Date Range")
    date_filter = st.selectbox(
        "Select Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time", "Custom Range"]
    )

    if date_filter == "Custom Range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End", datetime.now())
    else:
        end_date = datetime.now()
        if date_filter == "Last 7 Days":
            start_date = end_date - timedelta(days=7)
        elif date_filter == "Last 30 Days":
            start_date = end_date - timedelta(days=30)
        elif date_filter == "Last 90 Days":
            start_date = end_date - timedelta(days=90)
        else:
            start_date = datetime(2000, 1, 1)

    st.divider()

    st.subheader("🎯 Program Filter (applied below)")
# ---------------------------------------------------
# Data Loading – CSV ONLY (report.csv)
# ---------------------------------------------------
@st.cache_data(ttl=300)
def load_data(start, end, path: str = "report.csv") -> pd.DataFrame:
    """
    Load and clean donation data from a PayPal CSV export saved as report.csv
    in the same directory as this script.
    """
    try:
        df_upload = pd.read_csv(path)

        # Base renaming (don't touch amount/program here)
        column_mapping = {
            'Date': 'date',
            'Creation Date': 'date',
            'Transaction ID': 'transaction_id',
            'Status': 'status',
            'Currency': 'currency',
        }
        df_upload.rename(columns=column_mapping, inplace=True)

        # ----- Amount handling (choose ONE source column) -----
        amount_source_cols = ['Net', 'Gross', 'Amount', 'AMOUNT']
        amount_col = None
        for col in amount_source_cols:
            if col in df_upload.columns:
                amount_col = col
                break

        if amount_col is None:
            raise ValueError(
                "No amount column found. Expected one of: Net, Gross, Amount, AMOUNT."
            )

        # Create unified 'amount' column
        df_upload['amount'] = df_upload[amount_col]
        df_upload['amount'] = (
            df_upload['amount']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace('USD', '', regex=False)
            .str.strip()
        )
        df_upload['amount'] = pd.to_numeric(df_upload['amount'], errors='coerce').fillna(0.0)

        # ----- KEEP ONLY DONATION PAYMENTS -----
        if 'Type' in df_upload.columns:
            df_upload = df_upload[df_upload['Type'] == 'Donation Payment']

        # ----- Program handling -----
        # Collect potential program columns and then coalesce them
        prog_cols = []

        if 'program' in df_upload.columns:
            prog_cols.append('program')

        if 'Subject' in df_upload.columns:
            df_upload.rename(columns={'Subject': 'program_subject'}, inplace=True)
            prog_cols.append('program_subject')

        if 'Note' in df_upload.columns:
            df_upload.rename(columns={'Note': 'program_note'}, inplace=True)
            prog_cols.append('program_note')

        if 'Item Title' in df_upload.columns:
            df_upload.rename(columns={'Item Title': 'program_item_title'}, inplace=True)
            prog_cols.append('program_item_title')

        # 👈 Important for your export: Item ID carries the program name
        if 'Item ID' in df_upload.columns:
            df_upload.rename(columns={'Item ID': 'program_item_id'}, inplace=True)
            prog_cols.append('program_item_id')

        if prog_cols:
            df_upload['program'] = (
                df_upload[prog_cols]
                .bfill(axis=1)
                .iloc[:, 0]
                .fillna('General Fund')
            )
        else:
            df_upload['program'] = 'General Fund'

        # ----- Fill required columns if missing -----
        if 'transaction_id' not in df_upload.columns:
            df_upload['transaction_id'] = [f"TXN{i}" for i in range(len(df_upload))]
        if 'currency' not in df_upload.columns:
            df_upload['currency'] = 'USD'
        if 'status' not in df_upload.columns:
            df_upload['status'] = 'completed'

        # ----- Date parsing & filtering -----
        df_upload['date'] = pd.to_datetime(df_upload['date'], errors='coerce')
        df_upload = df_upload.dropna(subset=['date'])

        start_datetime = pd.to_datetime(start)
        end_datetime = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask = (df_upload['date'] >= start_datetime) & (df_upload['date'] <= end_datetime)
        df_upload = df_upload.loc[mask]

        return df_upload[['transaction_id', 'date', 'amount', 'currency', 'program', 'status']]

    except Exception as e:
        st.error(f"Error reading report.csv: {e}")
        return pd.DataFrame(columns=['transaction_id', 'date', 'amount', 'currency', 'program', 'status'])


df = load_data(start_date, end_date)

# ---------------------------------------------------
# Program filter
# ---------------------------------------------------
if not df.empty and 'program' in df.columns:
    programs = ["All Programs"] + sorted(df['program'].astype(str).unique().tolist())
else:
    programs = ["All Programs"]

selected_program = st.sidebar.selectbox("Filter by Program", programs)

if selected_program != "All Programs" and not df.empty:
    df = df[df['program'] == selected_program]

# ---------------------------------------------------
# Main content
# ---------------------------------------------------
if df.empty:
    st.warning("No donation data available for the selected period.")
else:
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    total_amount = df['amount'].sum()
    total_count = len(df)
    avg_donation = df['amount'].mean()
    unique_programs = df['program'].nunique()

    with col1:
        st.metric("💵 Total Donations", f"${total_amount:,.2f}")

    with col2:
        st.metric("📊 Number of Donations", f"{total_count:,}")

    with col3:
        st.metric("📈 Average Donation", f"${avg_donation:,.2f}")

    with col4:
        st.metric("🎯 Active Programs", unique_programs)

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Donations by Program")

        program_totals = (
            df.groupby('program', as_index=False)['amount']
            .sum()
            .rename(columns={'amount': 'total_amount'})
            .sort_values('total_amount', ascending=True)
        )

        fig_bar = px.bar(
            program_totals,
            x='total_amount',
            y='program',
            orientation='h',
            labels={'total_amount': 'Total Amount ($)', 'program': 'Program'},
            color='total_amount',
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("🥧 Distribution by Program")
        program_data = df.groupby('program')['amount'].sum().reset_index()

        fig_pie = px.pie(
            program_data,
            values='amount',
            names='program',
            hole=0.4
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Trend over time
    st.subheader("📈 Donation Trend Over Time")
    daily_donations = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()
    daily_donations.columns = ['date', 'amount']

    fig_line = px.line(
        daily_donations,
        x='date',
        y='amount',
        labels={'date': 'Date', 'amount': 'Total Amount ($)'},
        markers=True
    )
    fig_line.update_layout(height=300)
    st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # Program Summary Table
    st.subheader("📋 Program Summary")

    summary = df.groupby('program').agg({
        'amount': ['sum', 'count', 'mean'],
        'transaction_id': 'count'
    }).round(2)

    summary.columns = ['Total Amount', 'Donation Count', 'Average Donation', 'Transactions']
    summary['Total Amount'] = summary['Total Amount'].apply(lambda x: f"${x:,.2f}")
    summary['Average Donation'] = summary['Average Donation'].apply(lambda x: f"${x:,.2f}")
    summary = summary.sort_values('Donation Count', ascending=False)

    st.dataframe(summary, use_container_width=True)

    # Export options
    st.divider()
    _, col2 = st.columns([3, 1])

    with col2:
        st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False),
            file_name=f"donations_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🔒 Privacy Protected - No donor personal information is displayed</p>
    <p style='font-size: 0.9em;'>Data is loaded from <code>report.csv</code> in this app's directory.</p>
</div>
""", unsafe_allow_html=True)
