import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import requests
import base64

# ---------------------------------------------------
# PayPal API Configuration
# ---------------------------------------------------
try:
    PAYPAL_CLIENT_ID = st.secrets["PAYPAL_CLIENT_ID"]
    PAYPAL_SECRET = st.secrets["PAYPAL_SECRET"]
    PAYPAL_MODE = st.secrets.get("PAYPAL_MODE", "sandbox")  # "sandbox" or "live"
except KeyError:
    PAYPAL_CLIENT_ID = None
    PAYPAL_SECRET = None
    PAYPAL_MODE = "sandbox"

PAYPAL_API_BASE = {
    "sandbox": "https://api-m.sandbox.paypal.com",
    "live": "https://api-m.paypal.com"
}

# ---------------------------------------------------
# PayPal helpers
# ---------------------------------------------------
def get_paypal_access_token():
    """Get PayPal OAuth access token"""
    if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
        st.error("❌ PayPal credentials not configured. Please add them to .streamlit/secrets.toml")
        return None
    
    url = f"{PAYPAL_API_BASE[PAYPAL_MODE]}/v1/oauth2/token"
    
    auth = base64.b64encode(f"{PAYPAL_CLIENT_ID}:{PAYPAL_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error(f"""
            ❌ **Authentication Failed (401 Unauthorized)**
            
            **Current Mode**: {PAYPAL_MODE}
            **API URL**: {url}
            
            **Common fixes:**
            1. Make sure you're using **{PAYPAL_MODE.upper()}** credentials from PayPal Developer Dashboard
            2. Double-check your Client ID and Secret have no extra spaces
            3. If using LIVE mode, ensure your app is approved for production
            4. Try switching to 'sandbox' mode for testing
            
            **Your current credentials start with:**
            - Client ID: {PAYPAL_CLIENT_ID[:10]}...
            - Secret: {PAYPAL_SECRET[:5]}...
            """)
        else:
            st.error(f"PayPal API Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error getting PayPal access token: {e}")
        return None


def fetch_paypal_transactions(access_token, start_date, end_date):
    """Fetch transactions from PayPal API"""
    url = f"{PAYPAL_API_BASE[PAYPAL_MODE]}/v1/reporting/transactions"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    params = {
        "start_date": start_date.strftime("%Y-%m-%dT00:00:00-0000"),
        "end_date": end_date.strftime("%Y-%m-%dT23:59:59-0000"),
        "fields": "all",
        "page_size": "500"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            st.error("""
            ❌ **Transaction Search API Access Denied (403 Forbidden)**
            
            Your PayPal app doesn't have permission to use the Transaction Search API.
            
            **To fix this, you have 3 options:**
            
            **Option 1: Enable Transaction Search (Recommended)**
            1. Go to https://developer.paypal.com/dashboard
            2. Click on your app
            3. Scroll to "Features" section
            4. Enable "Transaction Search"
            5. Wait a few minutes for activation
            
            **Option 2: Use PayPal Webhooks (Real-time)**
            - Set up webhooks to receive donation events automatically
            - More reliable for real-time tracking
            
            **Option 3: Export from PayPal Dashboard**
            - Download transaction reports from PayPal.com
            - Upload CSV file to this dashboard
            
            **For now, using sample data instead.**
            """)
        else:
            st.error(f"PayPal API Error ({e.response.status_code}): {e}")
        return None
    except Exception as e:
        st.error(f"Error fetching PayPal transactions: {e}")
        return None


def parse_paypal_transactions(transactions_data) -> pd.DataFrame:
    """Parse PayPal transaction data into DataFrame"""
    donations = []
    
    if not transactions_data or "transaction_details" not in transactions_data:
        return pd.DataFrame()
    
    for txn in transactions_data["transaction_details"]:
        # Filter for completed donations only
        if txn.get("transaction_info", {}).get("transaction_status") == "S":
            transaction_info = txn.get("transaction_info", {})
            
            # Extract program from transaction note or custom field
            program = "General Fund"  # Default
            if "transaction_note" in transaction_info:
                program = transaction_info["transaction_note"]
            elif "custom_field" in transaction_info:
                program = transaction_info["custom_field"]
            
            donation = {
                "transaction_id": transaction_info.get("transaction_id", ""),
                "date": pd.to_datetime(transaction_info.get("transaction_initiation_date", "")),
                "amount": float(transaction_info.get("transaction_amount", {}).get("value", 0)),
                "currency": transaction_info.get("transaction_amount", {}).get("currency_code", "USD"),
                "program": program,
                "status": "completed"
            }
            
            donations.append(donation)
    
    return pd.DataFrame(donations)


def load_sample_data() -> pd.DataFrame:
    """Load sample data for testing"""
    data = {
        "transaction_id": [f"TXN{i}" for i in range(1, 51)],
        "date": pd.date_range(end=datetime.now(), periods=50, freq='D'),
        "amount": [100, 250, 500, 150, 300, 75, 450, 200, 600, 175,
                   125, 275, 525, 175, 325, 100, 475, 225, 625, 200,
                   150, 300, 550, 200, 350, 125, 500, 250, 650, 225,
                   175, 325, 575, 225, 375, 150, 525, 275, 675, 250,
                   200, 350, 600, 250, 400, 175, 550, 300, 700, 275],
        "currency": ["USD"] * 50,
        "program": ["Education Fund"] * 15
                   + ["Healthcare Initiative"] * 12
                   + ["Community Development"] * 11
                   + ["Emergency Relief"] * 12,
        "status": ["completed"] * 50
    }
    df = pd.DataFrame(data)
    return df

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
# Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    data_source = st.radio(
        "Data Source",
        ["Sample Data", "Live PayPal Data", "Upload CSV"],
        help="Use sample data for testing or connect to PayPal"
    )
    
    uploaded_file = None
    if data_source == "Upload CSV":
        st.info("📄 Upload a CSV exported from PayPal with columns: Date, Amount, Program")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    st.divider()
    
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
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# ---------------------------------------------------
# Data Loading
# ---------------------------------------------------
@st.cache_data(ttl=300)
def load_data(source, start, end, uploaded_file=None):
    # ---------- 1) CSV UPLOAD ----------
    if source == "Upload CSV" and uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)

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

            # 👈 This is the important one for your file
            if 'Item ID' in df_upload.columns:
                df_upload.rename(columns={'Item ID': 'program_item_id'}, inplace=True)
                prog_cols.append('program_item_id')

            if prog_cols:
                # First non-null of [Item ID, Subject, Note, Item Title, existing program]
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
            st.error(f"Error reading CSV: {e}")
            return load_sample_data()

    # ---------- 2) LIVE PAYPAL DATA ----------
    if source == "Live PayPal Data":
        with st.spinner("Fetching PayPal transactions..."):
            token = get_paypal_access_token()
            if token:
                txn_data = fetch_paypal_transactions(token, start, end)
                if txn_data:
                    return parse_paypal_transactions(txn_data)
        st.warning("Could not fetch PayPal data. Using sample data instead.")
        return load_sample_data()

    # ---------- 3) DEFAULT: SAMPLE DATA ----------
    return load_sample_data()


df = load_data(data_source, start_date, end_date, uploaded_file)

# ---------------------------------------------------
# Filter by date (for sample / live data without CSV date filtering)
# ---------------------------------------------------
if not df.empty:
    df['date'] = pd.to_datetime(df['date'])
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df['date'] >= start_datetime) & (df['date'] <= end_datetime)]

# ---------------------------------------------------
# Program filter
# ---------------------------------------------------
programs = ["All Programs"] + (sorted(df['program'].unique().tolist()) if not df.empty else [])
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

        # Build a clean DataFrame for Plotly
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
    <p style='font-size: 0.9em;'>Data updates every 5 minutes when using live PayPal connection</p>
</div>
""", unsafe_allow_html=True)
