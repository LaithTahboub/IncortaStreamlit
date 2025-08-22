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
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.stats import pearsonr, f_oneway, chi2_contingency
import math
import warnings
warnings.filterwarnings('ignore')

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
        background-color: white !important;
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
    st.subheader("✨ AI Data Assistant")
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

def create_key_insights(df):
    """Generate AI-powered key insights with statistical analysis"""
    st.subheader("🔍 Key Insights - AI Statistical Analysis")
    st.markdown("Advanced analytics to identify trends, anomalies, and areas requiring attention")
    
    # Add specific CSS for Key Insights section
    st.markdown("""
    <style>
    /* Key Insights specific styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: black !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: black !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        color: black !important;
    }
    
    /* Ensure all markdown in Key Insights is black */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: black !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: black !important;
    }
    
    /* Tab content area */
    .stTabs > div > div > div > div {
        color: black !important;
    }
    
    /* All text in tabs */
    .stTabs * {
        color: black !important;
    }
    
    /* Expander headers in Key Insights */
    .stExpander > details > summary {
        background-color: white !important;
        color: black !important;
        border: 1px solid #e9ecef !important;
    }
    
    .stExpander > details > summary:hover {
        background-color: #f8f9fa !important;
    }
    
    /* All expander elements */
    [data-testid="stExpander"] {
        background-color: white !important;
    }
    
    [data-testid="stExpander"] > div {
        background-color: white !important;
        color: black !important;
    }
    
    /* Dropdown/selectbox styling */
    .stSelectbox > div > div {
        background-color: white !important;
    }
    
    .stSelectbox > div > div > div {
        background-color: white !important;
        color: black !important;
    }
    
    /* Dropdown menu when opened */
    [data-testid="stSelectbox"] > div > div > div {
        background-color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Identify numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Insufficient numeric data for comprehensive analysis")
        return
    
    # Create tabs for different types of insights
    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Correlations", "⚠️ Anomalies", "📈 Trends", "🎯 Recommendations"])
    

    with tab1:
        st.subheader("Correlation Analysis")

        # --- Setup & selection ---
        all_columns = df.columns.tolist()
        st.markdown("**Select a variable to analyze:**")
        selected_var = st.selectbox("Variable to analyze", all_columns, key="corr_primary_var")

        # --- Working copy: convert date-like columns to numeric days since min date ---
        df_work = df.copy()
        date_converted = []

        for col in df_work.columns:
            # Try parsing if dtype is object or column name suggests a date
            if (df_work[col].dtype == "object") or ("date" in col.lower()):
                try:
                    parsed = pd.to_datetime(df_work[col], errors="raise")
                    # If parsing succeeded for most rows, keep it
                    if parsed.notna().sum() >= max(3, int(0.5 * len(parsed))):
                        df_work[col] = (parsed - parsed.min()).dt.days
                        date_converted.append(col)
                except Exception:
                    # leave as-is if parsing fails
                    pass

        # Identify numeric and categorical columns based on df_work (post date conversion)

        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df_work.columns if c not in numeric_cols]

        # --- Associations: choose appropriate test per pair ---
        results = []  # dicts: {var, effect, raw, p, method, direction, note}

        target = selected_var
        target_is_num = target in numeric_cols

        for var in df_work.columns:
            if var == target:
                continue

            var_is_num = var in numeric_cols
            direction = None
            note = None

            try:
                # A) Numeric–Numeric → Pearson r
                if target_is_num and var_is_num:
                    s1 = df_work[target]
                    s2 = df_work[var]
                    mask = s1.notna() & s2.notna()
                    s1, s2 = s1[mask], s2[mask]
                    if len(s1) >= 3 and s1.nunique() > 1 and s2.nunique() > 1:
                        r, p = pearsonr(s1, s2)
                        effect = abs(r)
                        direction = "positively" if r > 0 else "negatively"
                        method = "Pearson r"
                        results.append({"var": var, "effect": effect, "raw": r, "p": p,
                                        "method": method, "direction": direction, "note": note})

                # B) Numeric–Categorical → One-way ANOVA (partial η²)
                elif target_is_num and not var_is_num:
                    # groups based on original categorical labels in df (not df_work)
                    cats = df[var].dropna().unique()
                    groups = [df_work[target][df[var] == cat].dropna() for cat in cats]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) > 1:
                        stat, p = f_oneway(*groups)
                        k = len(groups)
                        N = sum(len(g) for g in groups)
                        eta2_p = ((k - 1) * stat) / (((k - 1) * stat) + (N - k)) if (N - k) > 0 else 0.0
                        method = "ANOVA (η²)"
                        results.append({"var": var, "effect": eta2_p, "raw": eta2_p, "p": p,
                                        "method": method, "direction": None,
                                        "note": "Effect size is partial η²; higher = stronger group differences."})

                elif not target_is_num and var_is_num:
                    cats = df[target].dropna().unique()
                    groups = [df_work[var][df[target] == cat].dropna() for cat in cats]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) > 1:
                        stat, p = f_oneway(*groups)
                        k = len(groups)
                        N = sum(len(g) for g in groups)
                        eta2_p = ((k - 1) * stat) / (((k - 1) * stat) + (N - k)) if (N - k) > 0 else 0.0
                        method = "ANOVA (η²)"
                        results.append({"var": var, "effect": eta2_p, "raw": eta2_p, "p": p,
                                        "method": method, "direction": None,
                                        "note": "Effect size is partial η²; higher = stronger group differences."})

                # C) Categorical–Categorical → Chi-squared (Cramér's V)
                else:
                    # Use original df for categories to avoid any numeric conversions on dates
                    # Treat missing as a category so chisq doesn't drop rows inconsistently
                    ctab = pd.crosstab(df[target].fillna("Missing"), df[var].fillna("Missing"))
                    if ctab.shape[0] > 1 and ctab.shape[1] > 1:
                        chi2, p, dof, exp = chi2_contingency(ctab)
                        n = ctab.values.sum()
                        r_dim, c_dim = ctab.shape
                        denom = n * (min(r_dim - 1, c_dim - 1))
                        if denom > 0:
                            v = math.sqrt(chi2 / denom)  # Cramér's V ∈ [0,1]
                            method = "Chi-squared (Cramér's V)"
                            results.append({"var": var, "effect": v, "raw": v, "p": p,
                                            "method": method, "direction": None,
                                            "note": "Effect size is Cramér's V; higher = stronger association."})
            except Exception:
                # Skip problematic pairs quietly
                pass

        # Rank by effect size
        results = sorted(results, key=lambda d: d["effect"], reverse=True)
        top3 = results[:3]

        # Strength labels
        def strength_label(method, val):
            if method == "Pearson r":
                return "Strong" if val > 0.7 else "Moderate" if val > 0.3 else "Weak"
            if method == "ANOVA (η²)":
                # 0.01 small, 0.06 medium, 0.14 large (Cohen-ish)
                return "Strong" if val >= 0.14 else "Moderate" if val >= 0.06 else "Weak"
            if method == "Chi-squared (Cramér's V)":
                # 0.1 small, 0.3 medium, 0.5 large
                return "Strong" if val >= 0.5 else "Moderate" if val >= 0.3 else "Weak"
            return "—"

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Top 3 variables associated with {selected_var}:**")
            for i, item in enumerate(top3, 1):
                var = item["var"]
                method = item["method"]
                effect = item["effect"]
                raw = item["raw"]
                p = item["p"]
                direction = item["direction"]
                note = item["note"]
                label = strength_label(method, effect)

                if method == "Pearson r":
                    dir_txt = "positively" if direction == "positively" else "negatively"
                    st.markdown(f"**{i}. {var}** ({method})")
                    st.markdown(f"   Effect size |r|: {effect:.3f} ({label} {dir_txt}); r = {raw:.3f}, p = {p:.3g}")
                else:
                    st.markdown(f"**{i}. {var}** ({method})")
                    st.markdown(f"   Effect size: {raw:.3f} ({label}); p = {p:.3g}")
                    if note:
                        st.markdown(f"   *{note}*")

            # Visualization of strongest association
            if len(top3) > 0:
                strongest_var = top3[0]["var"]

                st.markdown(f"**Relationship: {selected_var} vs {strongest_var}**")

                sel_is_cat = selected_var in categorical_cols
                str_is_cat = strongest_var in categorical_cols

                if sel_is_cat or str_is_cat:
                    # Categorical involved
                    if sel_is_cat and not str_is_cat:
                        # selected_var categorical, strongest_var numeric → bar of means
                        grouped = df.groupby(selected_var)[strongest_var].mean().sort_values(ascending=False)
                        fig = px.bar(
                            x=grouped.index, y=grouped.values,
                            title=f"Average {strongest_var} by {selected_var}",
                            labels={'x': selected_var, 'y': f'Average {strongest_var}'}
                        )
                    elif (not sel_is_cat) and str_is_cat:
                        # strongest_var categorical, selected_var numeric → boxplot
                        fig = px.box(
                            df, x=strongest_var, y=selected_var,
                            title=f"{selected_var} distribution by {strongest_var}"
                        )
                        fig.update_xaxes(tickangle=45)
                    else:
                        # both categorical → heatmap of counts
                        cross_tab = pd.crosstab(df[selected_var], df[strongest_var])
                        fig = px.imshow(
                            cross_tab,
                            title=f"Cross-tabulation: {selected_var} vs {strongest_var}",
                            aspect='auto'
                        )
                else:
                    # Both numeric → scatter with trendline
                    fig = px.scatter(
                        df, x=selected_var, y=strongest_var,
                        title=f"Relationship: {selected_var} vs {strongest_var}",
                        trendline="ols"
                    )

                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Correlation heatmap for numeric variables only (post date conversion)
            if len(numeric_cols) > 1:
                st.markdown("**Numeric Variables Correlation Matrix:**")
                corr_matrix_numeric = df_work[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix_numeric,
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title="Numeric Correlations Heatmap"
                )
                fig_corr.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    height=400
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # Additional notes
            st.markdown("**Analysis Notes:**")
            if date_converted:
                st.markdown(f"• Converted {len(date_converted)} date-like column(s) to numeric days: {', '.join(date_converted[:5])}{'…' if len(date_converted)>5 else ''}")
            st.markdown(f"• {len(numeric_cols)} numeric variables and {len(categorical_cols)} categorical variables detected")
            st.markdown(f"• Associations use Pearson (num–num), ANOVA η² (num–cat), and Cramér’s V (cat–cat)")

            # Distribution / summary for selected variable
            if selected_var in categorical_cols:
                value_counts = df[selected_var].value_counts().head(5)
                st.markdown(f"**Top values in {selected_var}:**")
                for val, count in value_counts.items():
                    st.markdown(f"• {val}: {count} records")
            else:
                st.markdown(f"**{selected_var} statistics:**")
                st.markdown(f"• Mean: {df_work[selected_var].mean():.2f}")
                st.markdown(f"• Std Dev: {df_work[selected_var].std():.2f}")
                st.markdown(f"• Range: {df_work[selected_var].min():.2f} - {df_work[selected_var].max():.2f}")

    with tab2:
        st.subheader("Anomaly Detection")
        
        # Select primary metric for anomaly detection
        sales_col = 'Sales' if 'Sales' in df.columns else 'Amount' if 'Amount' in df.columns else numeric_cols[0]
        
        # Statistical anomaly detection using Z-score
        z_scores = np.abs(stats.zscore(df[sales_col].dropna()))
        threshold = 2.5
        anomalies_statistical = df[z_scores > threshold]
        
        # DBSCAN clustering for anomaly detection
        if len(numeric_cols) >= 2:
            features = df[numeric_cols[:3]].dropna()  # Use up to 3 features
            if len(features) > 10:  # Need sufficient data points
                scaler = StandardScaler()

                for col in features.columns:
                    if np.issubdtype(features[col].dtype, np.datetime64):
                        features[col] = (features[col] - features[col].min()).dt.days

                # Drop any non-numeric columns (still strings, objects, etc.)
                features = features.select_dtypes(include=[np.number])

                features_scaled = scaler.fit_transform(features)
                
                # DBSCAN clustering
                eps = 0.5
                min_samples = max(3, len(features) // 50)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(features_scaled)
                
                # Outliers are labeled as -1
                outlier_indices = features.index[clusters == -1]
                anomalies_clustering = df.loc[outlier_indices]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Statistical Anomalies (Z-score > 2.5)", len(anomalies_statistical))
            
            if len(anomalies_statistical) > 0:
                # Anomaly visualization
                fig_anom = px.scatter(
                    df, 
                    x=range(len(df)), 
                    y=sales_col,
                    title=f"Anomaly Detection - {sales_col}",
                    labels={'x': 'Data Point Index', 'y': sales_col}
                )
                
                # Highlight anomalies
                if len(anomalies_statistical) > 0:
                    fig_anom.add_scatter(
                        x=anomalies_statistical.index,
                        y=anomalies_statistical[sales_col],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='x'),
                        name='Anomalies'
                    )
                
                fig_anom.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    height=400
                )
                st.plotly_chart(fig_anom, use_container_width=True)
        
        with col2:
            if len(anomalies_statistical) > 0:
                st.markdown("**Anomalous Data Points:**")
                for idx, row in anomalies_statistical.head(5).iterrows():
                    value = row[sales_col]
                    st.markdown(f"• Record #{idx}: {sales_col} = {value:,.2f}")
                
                # Show anomaly statistics
                st.markdown("**Anomaly Statistics:**")
                st.markdown(f"• Mean anomaly value: {anomalies_statistical[sales_col].mean():,.2f}")
                st.markdown(f"• Normal range: {df[sales_col].mean() - 2*df[sales_col].std():,.2f} to {df[sales_col].mean() + 2*df[sales_col].std():,.2f}")
    
    with tab3:
        st.subheader("Trend Analysis")
        
        # Time-based analysis if date column exists
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_cols and len(date_cols) > 0:
            date_col = date_cols[0]
            try:
                # Support multiple date formats including "12/1/21"
                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors='coerce')
                # Remove any rows where date parsing failed
                df_clean = df.dropna(subset=[date_col])
                df_sorted = df_clean.sort_values(date_col)
                
                # Trend calculation
                x_numeric = np.arange(len(df_sorted))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, df_sorted[sales_col])
                
                st.markdown(f"**Trend Analysis for {sales_col}:**")
                trend_direction = "increasing" if slope > 0 else "decreasing"
                st.markdown(f"• Overall trend: **{trend_direction}** (slope: {slope:.4f})")
                st.markdown(f"• Trend strength: **{abs(r_value):.3f}** (R-squared: {r_value**2:.3f})")
                
                # Trend visualization
                fig_trend = px.scatter(
                    df_sorted,
                    x=date_col,
                    y=sales_col,
                    title=f"Trend Analysis - {sales_col} over Time",
                    trendline="ols"
                )
                fig_trend.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
            except:
                st.info("Date column found but could not be parsed for trend analysis")
        else:
            # Distribution and pattern analysis
            st.markdown("**Distribution Analysis:**")
            
            # Skewness and kurtosis
            skewness = stats.skew(df[sales_col].dropna())
            kurtosis = stats.kurtosis(df[sales_col].dropna())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Skewness", f"{skewness:.3f}")
                skew_interpretation = "right-skewed" if skewness > 0.5 else "left-skewed" if skewness < -0.5 else "normally distributed"
                st.markdown(f"Distribution is **{skew_interpretation}**")
            
            with col2:
                st.metric("Kurtosis", f"{kurtosis:.3f}")
                kurt_interpretation = "heavy-tailed" if kurtosis > 0.5 else "light-tailed" if kurtosis < -0.5 else "normal-tailed"
                st.markdown(f"Distribution has **{kurt_interpretation}**")
    
    with tab4:
        st.subheader("AI Recommendations")
        
        recommendations = []
        
        # Based on correlations - check numeric correlations
        if len(numeric_cols) >= 2:
            corr_matrix_numeric = df[numeric_cols].corr()
            strong_corrs = []
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicates
                        corr_val = corr_matrix_numeric.loc[col1, col2]
                        if abs(corr_val) > 0.7:  # Strong correlations
                            strong_corrs.append({
                                'Variable 1': col1,
                                'Variable 2': col2, 
                                'Correlation': corr_val
                            })
            
            if strong_corrs:
                recommendations.append({
                    'area': 'Strong Correlations',
                    'finding': f"Found {len(strong_corrs)} strong correlations between numeric variables",
                    'action': f"Monitor {strong_corrs[0]['Variable 1']} and {strong_corrs[0]['Variable 2']} together as they move in tandem",
                    'priority': 'High' if abs(strong_corrs[0]['Correlation']) > 0.8 else 'Medium'
                })
        
        # Based on anomalies
        if len(anomalies_statistical) > 0:
            anomaly_rate = len(anomalies_statistical) / len(df) * 100
            if anomaly_rate > 5:
                recommendations.append({
                    'area': 'Data Quality',
                    'finding': f"{anomaly_rate:.1f}% of data points are anomalous",
                    'action': "Investigate data collection processes and validate outlier transactions",
                    'priority': 'High'
                })
            else:
                recommendations.append({
                    'area': 'Outlier Management',
                    'finding': f"Found {len(anomalies_statistical)} potential outliers",
                    'action': "Review high-value transactions for accuracy and business insights",
                    'priority': 'Medium'
                })
        
        # Based on distribution analysis
        if 'sales_col' in locals():
            coefficient_of_variation = df[sales_col].std() / df[sales_col].mean()
            if coefficient_of_variation > 1:
                recommendations.append({
                    'area': 'Data Variability',
                    'finding': f"High variability in {sales_col} (CV: {coefficient_of_variation:.2f})",
                    'action': "Consider segmentation analysis to understand different customer/product groups",
                    'priority': 'Medium'
                })
        
        # Based on categorical analysis
        if categorical_cols:
            for cat_col in categorical_cols[:2]:  # Check top 2 categorical columns
                if df[cat_col].nunique() > len(df) * 0.8:  # Too many unique values
                    recommendations.append({
                        'area': 'Data Structure',
                        'finding': f"{cat_col} has too many unique values ({df[cat_col].nunique()})",
                        'action': f"Consider grouping or categorizing {cat_col} for better analysis",
                        'priority': 'Low'
                    })
        
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations):
                priority_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                with st.expander(f"{priority_color[rec['priority']]} {rec['area']} - {rec['priority']} Priority"):
                    st.markdown(f"**Finding:** {rec['finding']}")
                    st.markdown(f"**Recommended Action:** {rec['action']}")
        else:
            st.info("No specific recommendations generated. Your data appears to be well-structured.")
        
        # Summary insights
        st.markdown("---")
        st.markdown("**Summary Insights:**")
        
        # Data quality score
        quality_score = 100
        if len(anomalies_statistical) / len(df) > 0.05:
            quality_score -= 20
        if len([col for col in df.columns if df[col].isnull().sum() > 0]) > len(df.columns) * 0.3:
            quality_score -= 15
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Quality Score", f"{quality_score}%")
        with col2:
            st.metric("Anomaly Rate", f"{len(anomalies_statistical)/len(df)*100:.1f}%")
        with col3:
            insight_count = len(recommendations)
            st.metric("Insights Generated", insight_count)

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
    
    # Key Insights with Statistical Analysis
    create_key_insights(df)
    
    # Data visualizations
    create_supply_chain_visualizations(df)

if __name__ == "__main__":
    main()