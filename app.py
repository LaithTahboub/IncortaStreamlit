"""
Streamlit Data Analytics App with AI Chat
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from config import Config
from data_utils import load_sample_data
from incorta_connector import IncortaConnector, convert_incorta_to_dataframe
import plotly.express as px
import plotly.graph_objects as go

# PandasAI imports
try:
    import pandasai as pai
    from pandasai_openai.openai import OpenAI
    PANDASAI_AVAILABLE = True
except ImportError as e:
    PANDASAI_AVAILABLE = False
    print(f"PandasAI import error: {e}")

# Page configuration
st.set_page_config(
    page_title="Data Analytics with AI Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Light theme CSS - Force light mode
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Main content area */
    .main {
        background-color: #ffffff !important;
        # color: #000000 !important;
        padding-top: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    
    /* All text elements */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #000000 !important;
    }
    
    /* Streamlit specific text elements */
    # .stMarkdown, .stMarkdown p, .stMarkdown div {
    #     color: #000000 !important;
    # }
    
    /* Headers and subheaders */
    .stSubheader {
        color: #000000 !important;
    }
    
    /* Chat interface styles */
    # .chat-container {
    #     background-color: #f8f9fa;
    #     border-radius: 10px;
    #     padding: 1.5rem;
    #     margin-bottom: 2rem;
    #     border: 1px solid #e9ecef;
    #     box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    # }
            
    
    .chat-message-user {
        background-color: #007bff !important;
        color: white !important;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: left;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-message-user * {
        color: white !important;
        background-color: transparent !important;
    }
    
    .chat-message-user p, .chat-message-user div, .chat-message-user span {
        color: white !important;
        background-color: transparent !important;
    }
    
    .chat-message-assistant {
        background-color: white;
        color: #333;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        margin-right: 20%;
        text-align: left;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metrics styling */
    .stMetric {
        background-color: white !important;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Data containers */
    .data-info, .stExpander {
        background-color: #ffffff !important;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    /* Tables and dataframes */
    .stDataFrame {
        background-color: white !important;
        border: 1px solid #e9ecef;
        border-radius: 8px;
    }
    
    /* Force dataframe table styling */
    .stDataFrame > div {
        background-color: white !important;
    }
    
    .stDataFrame table {
        background-color: white !important;
        color: black !important;
    }
    
    .stDataFrame th {
        background-color: #f8f9fa !important;
        color: black !important;
        border: 1px solid #dee2e6 !important;
    }
    
    .stDataFrame td {
        background-color: white !important;
        color: black !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Override any dark theme for data display */
    [data-testid="stDataFrame"] {
        background-color: white !important;
    }
    
    [data-testid="stDataFrame"] div {
        background-color: white !important;
        color: black !important;
    }
    
    /* More specific dataframe styling */
    div[data-testid="dataframe"] {
        background-color: white !important;
        color: black !important;
    }
    
    div[data-testid="dataframe"] * {
        background-color: white !important;
        color: black !important;
    }
    
    /* Streamlit dataframe container */
    .element-container .stDataFrame {
        background-color: white !important;
    }
    
    /* All table elements */
    table, th, td, tr {
        background-color: white !important;
        color: black !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Data editor specific styling */
    .stDataEditor {
        background-color: white !important;
        color: black !important;
    }
    
    .stDataEditor * {
        background-color: white !important;
        color: black !important;
    }
    
    /* Fix data table header and search */
    .stDataFrame .dvn-scroller {
        background-color: white !important;
    }
    
    .stDataFrame .grid-container {
        background-color: white !important;
    }
    
    /* Data table toolbar/header */
    .stDataFrame .stDataFrameToolbar {
        background-color: white !important;
        color: black !important;
    }
    
    /* Search box in data table */
    .stDataFrame input[type="text"] {
        background-color: white !important;
        color: black !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Data table pagination */
    .stDataFrame .pagination {
        background-color: white !important;
        color: black !important;
    }
    
    /* Generic input styling */
    input {
        background-color: white !important;
        color: black !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Override any remaining dark elements */
    .element-container {
        background-color: white !important;
    }
    
    /* Top header and toolbar areas */
    .stApp > header {
        background-color: white !important;
    }
    
    /* Any remaining dark containers */
    div[class*="block-container"] {
        background-color: white !important;
    }
    
    /* Expander header fix */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: black !important;
    }
    
    /* Search and filter elements */
    .stSelectbox, .stMultiSelect, .stTextInput {
        background-color: white !important;
    }
    
    .stSelectbox *, .stMultiSelect *, .stTextInput * {
        background-color: white !important;
        color: black !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px !important;
        border: 1px solid #007bff !important;
        background-color: white !important;
        color: #007bff !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #007bff !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,123,255,0.3) !important;
    }
    
    /* Form submit button */
    button[kind="formSubmit"] {
        background-color: #007bff !important;
        color: white !important;
        border: 1px solid #007bff !important;
        border-radius: 8px !important;
    }
    
    /* All buttons light mode */
    button {
        background-color: white !important;
        color: #007bff !important;
        border: 1px solid #007bff !important;
    }
    
    button:hover {
        background-color: #007bff !important;
        color: white !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: #000000 !important;
        border: 1px solid #e9ecef;
        border-radius: 8px;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > select {
        background-color: white !important;
        color: #000000 !important;
    }
    
    /* Plotly charts light background */
    .plotly {
        color: #000000 !important;
        background-color: white !important;
    }
    
    /* Info/Success/Error boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-color: #c3e6cb !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border-color: #f5c6cb !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
        border-color: #bee5eb !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: #000000 !important;
        border: 1px solid #e9ecef;
        border-radius: 8px;
    }
    
    .streamlit-expanderContent {
        background-color: white !important;
        border: 1px solid #e9ecef;
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
</style>
""", unsafe_allow_html=True)

def setup_ai_chat(df):
    """Setup AI chat interface"""
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.subheader("AI Data Assistant")
    st.markdown("Ask questions about your data in natural language")
    
    if not PANDASAI_AVAILABLE:
        st.error("PandasAI is not available. Please install required dependencies.")
        st.code("pip install pandasai-openai")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # OpenAI configuration
    openai_api_key = Config.OPENAI_API_KEY
    
    if not openai_api_key:
        st.warning("Please configure OpenAI API key in your .env file to use the AI assistant.")
        st.info("Add this to your .env file:")
        st.code("OPENAI_API_KEY=your_openai_api_key_here")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    try:
        # Initialize PandasAI with OpenAI LLM
        if not OpenAI:
            st.error("OpenAI class not available")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Set OpenAI API key as environment variable
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Create OpenAI LLM instance
        llm = OpenAI(api_token=openai_api_key)
        
        # Configure PandasAI to use OpenAI LLM
        pai.config.set({
            "llm": llm,
            "temperature": 0.1,
            "verbose": False
        })
        
        # Create DataFrame using PandasAI's DataFrame
        pai_df = pai.DataFrame(df)
        
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-assistant">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input with form for Enter key support
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_question = st.text_input(
                    "Ask a question about your data:",
                    placeholder="e.g., 'What is the average value by category?'",
                    label_visibility="collapsed",
                    key="user_input"
                )
            with col2:
                send_button = st.form_submit_button("Send", type="primary")
        
        # Clear chat button only
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        if send_button and user_question:
            try:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Get AI response
                with st.spinner("Analyzing data..."):
                    if pai_df is None:
                        raise ValueError("PandasAI DataFrame is not initialized")
                    if not user_question or user_question.strip() == "":
                        raise ValueError("Question cannot be empty")
                    
                    response = pai_df.chat(user_question)
                    
                    # Validate response
                    if response is None:
                        response_text = "I couldn't generate a response for that question. Please try rephrasing."
                    else:
                        response_text = str(response).strip()
                        if not response_text:
                            response_text = "I received an empty response. Please try a different question."
                
                # Add AI response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
                # Rerun to update chat display
                st.rerun()
                
            except Exception as e:
                st.error(f"Error getting AI response: {str(e)}")
                st.info("Please check your OpenAI configuration and try again.")
        
    except Exception as e:
        st.error(f"Error setting up AI assistant: {str(e)}")
        st.info("Please verify your OpenAI configuration in the .env file")
    
    st.markdown('</div>', unsafe_allow_html=True)

def load_data():
    """Load data from specific Incorta dashboard insight"""
    # Hardcoded dashboard and insight IDs
    dashboard_id = "f4fe183f-bfc2-4ef8-8260-0bbffbfe7af3"
    insight_id = "fb0ed1e5-dfe6-4512-a4a5-bb57fbc2bd6c"
    
    df = None
    incorta = IncortaConnector()
    
    # Automatically load the data (no sidebar display)
    with st.spinner("Loading dashboard insight..."):
        result = incorta.query_dashboard_insight(dashboard_id, insight_id)
        if result:
            df = convert_incorta_to_dataframe(result)
            if df is None:
                df = load_sample_data()
        else:
            df = load_sample_data()
    
    return df

def create_supply_chain_visualizations(df):
    """Create specialized visualizations for supply chain data"""
    
    # Data overview metrics
    st.subheader("Supply Chain Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Detect column mapping - check for Sales/Profit (Incorta) or Amount (sample data)
    sales_col = 'Sales' if 'Sales' in df.columns else 'Amount' if 'Amount' in df.columns else None
    profit_col = 'Profit' if 'Profit' in df.columns else None
    region_col = 'Region' if 'Region' in df.columns else 'Site' if 'Site' in df.columns else None
    status_col = 'Status' if 'Status' in df.columns else 'Priority' if 'Priority' in df.columns else None
    category_col = 'Category' if 'Category' in df.columns else None
    
    # Total sales and metrics
    if sales_col:
        total_sales = df[sales_col].sum()
        avg_sales = df[sales_col].mean()
        with col1:
            st.metric("Total Sales", f"${total_sales:,.2f}")
        with col2:
            st.metric("Average Order Value", f"${avg_sales:,.2f}")
    
    if profit_col:
        total_profit = df[profit_col].sum()
        with col3:
            st.metric("Total Profit", f"${total_profit:,.2f}")
    else:
        with col3:
            st.metric("Total Orders", f"{len(df):,}")
    
    # Unique regions/sites metric
    if region_col:
        unique_regions = df[region_col].nunique()
        with col4:
            st.metric("Active Regions", f"{unique_regions:,}")
    
    # Data preview
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Create visualization columns
    col1, col2 = st.columns(2)
    
    # Sales by Region (if Sales and Region columns exist)
    if sales_col and region_col:
        with col1:
            st.subheader("Sales by Region")
            region_sales = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=region_sales.values, 
                y=region_sales.index,
                orientation='h',
                title="Sales Distribution by Region",
                labels={'x': 'Sales ($)', 'y': 'Region'},
                color=region_sales.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', family='Arial, sans-serif', size=12),
                showlegend=False,
                height=400,
                xaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
                yaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
                title=dict(font=dict(color='black'))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Priority/Status Distribution (if Priority or Status column exists)
    if status_col:
        with col2:
            st.subheader("Priority Distribution")
            status_counts = df[status_col].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Orders by Priority",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', family='Arial, sans-serif', size=12),
                height=400,
                title=dict(font=dict(color='black'))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Category Analysis (if Category and Sales columns exist)
    if category_col and sales_col:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales by Category")
            category_sales = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
            fig = px.bar(
                x=category_sales.index,
                y=category_sales.values,
                title="Category Performance",
                labels={'x': 'Category', 'y': 'Sales ($)'},
                color=category_sales.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', family='Arial, sans-serif', size=12),
                xaxis_tickangle=-45,
                showlegend=False,
                height=400,
                xaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
                yaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
                title=dict(font=dict(color='black'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Order Count by Category")
            category_counts = df[category_col].value_counts()
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Orders by Category",
                labels={'x': 'Category', 'y': 'Number of Orders'},
                color=category_counts.values,
                color_continuous_scale='Oranges'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', family='Arial, sans-serif', size=12),
                xaxis_tickangle=-45,
                showlegend=False,
                height=400,
                xaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
                yaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
                title=dict(font=dict(color='black'))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Customer Performance (if Customer and Sales columns exist)
    if 'Customer' in df.columns and sales_col:
        st.subheader("Top Customer Performance")
        customer_stats = df.groupby('Customer').agg({
            sales_col: ['sum', 'count', 'mean']
        }).round(2)
        customer_stats.columns = ['Total Sales', 'Order Count', 'Avg Order Value']
        customer_stats = customer_stats.sort_values('Total Sales', ascending=False).head(10)
        
        fig = px.scatter(
            customer_stats,
            x='Order Count',
            y='Total Sales',
            size='Avg Order Value',
            hover_name=customer_stats.index,
            title="Customer Performance: Sales vs Order Volume",
            labels={'x': 'Number of Orders', 'y': 'Total Sales ($)'},
            color='Total Sales',
            color_continuous_scale='RdYlBu'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', family='Arial, sans-serif', size=12),
            height=500,
            xaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
            yaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
            title=dict(font=dict(color='black'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sales distribution (if Sales column exists)
    if sales_col:
        st.subheader("Sales Value Distribution")
        fig = px.histogram(
            df,
            x=sales_col,
            nbins=30,
            title="Distribution of Sales Values",
            labels={'x': 'Sales Amount ($)', 'y': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', family='Arial, sans-serif', size=12),
            height=400,
            xaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
            yaxis=dict(tickfont=dict(color='black'), title=dict(font=dict(color='black'))),
            title=dict(font=dict(color='black'))
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    
    # Title
    st.title("Supply Chain Analytics with AI")
    st.markdown("Analyze your supply chain data and ask questions in natural language")
    
    # Load data
    df = load_data()
    
    # Store dataframe in session state for AI chat
    st.session_state.current_dataframe = df
    
    # AI Chat Interface (at the top)
    setup_ai_chat(df)
    
    # Data visualizations
    create_supply_chain_visualizations(df)

if __name__ == "__main__":
    main()