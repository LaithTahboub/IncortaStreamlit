"""
Claude Data Assistant for advanced data analysis and insights
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import anthropic

class ClaudeDataAssistant:
    """
    Advanced data assistant using Claude for intelligent data analysis
    """
    
    def __init__(self, api_key: str):
        """Initialize Claude client"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        # Initialize persistent execution environment
        self.execution_globals = {}
        self.execution_locals = {}
    
    def get_data_summary(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive data summary for Claude context"""
        
        # Convert dtypes to strings for JSON serialization
        dtypes_dict = {}
        for col, dtype in df.dtypes.items():
            dtypes_dict[col] = str(dtype)
        
        # Convert missing values to regular integers
        missing_values_dict = {}
        for col, count in df.isnull().sum().items():
            missing_values_dict[col] = int(count)
        
        # Convert sample data, handling non-serializable types
        sample_data = []
        for _, row in df.head(3).iterrows():
            row_dict = {}
            for col, value in row.items():
                if pd.isna(value):
                    row_dict[col] = None
                elif isinstance(value, (np.integer, np.floating)):
                    row_dict[col] = float(value) if isinstance(value, np.floating) else int(value)
                else:
                    row_dict[col] = str(value)
            sample_data.append(row_dict)
        
        summary = {
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "dtypes": dtypes_dict,
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "missing_values": missing_values_dict,
            "sample_data": sample_data
        }
        
        # Add basic statistics for numeric columns
        numeric_stats = {}
        for col in summary["numeric_columns"]:
            try:
                stats_dict = {}
                for stat_name, stat_func in [("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max"), ("median", "median")]:
                    value = getattr(df[col], stat_func)()
                    if pd.isna(value):
                        stats_dict[stat_name] = None
                    else:
                        stats_dict[stat_name] = float(value)
                numeric_stats[col] = stats_dict
            except Exception:
                # Skip columns that can't compute statistics
                numeric_stats[col] = {"error": "Unable to compute statistics"}
        summary["numeric_statistics"] = numeric_stats
        
        # Add value counts for categorical columns (top 5)
        categorical_info = {}
        for col in summary["categorical_columns"]:
            try:
                top_values = df[col].value_counts().head(5)
                categorical_info[col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in top_values.items()}
                }
            except Exception:
                categorical_info[col] = {
                    "unique_count": 0,
                    "top_values": {},
                    "error": "Unable to compute value counts"
                }
        summary["categorical_info"] = categorical_info
        
        return json.dumps(summary, indent=2)
    
    def create_analysis_prompt(self, question: str, data_summary: str) -> str:
        """Create a comprehensive prompt for Claude to analyze data"""
        prompt = f"""You are an expert data analyst. You have access to a dataset with the following structure and summary:

{data_summary}

The user has asked: "{question}"

Your task is to provide a comprehensive analysis by writing Python code to answer this question. Follow these guidelines:

1. **Data Analysis Approach:**
   - Start with exploratory analysis to understand the data better
   - Use appropriate statistical methods and visualizations
   - Apply business intelligence thinking to provide actionable insights

2. **Code Structure:**
   - Write clean, well-commented Python code
   - Keep analysis concise - aim for ONE code block that answers the question completely
   - Use meaningful variable names
   - Include data validation and error handling where appropriate
   - **IMPORTANT: DO NOT include any import statements** - all libraries are already imported and available

3. **Analysis Depth:**
   - Go beyond basic descriptive statistics
   - Look for patterns, trends, and anomalies
   - Provide statistical significance testing when relevant
   - Consider business context and practical implications

4. **Output Format:**
   - Start each code block with ```python
   - End each code block with ```
   - After each code block, provide interpretation in natural language
   - Use visualizations to support your findings
   - Conclude with actionable recommendations

5. **Available Libraries and DataFrame:**
   - The data is available in a pandas DataFrame called `df`
   - All libraries are pre-imported: pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go, scipy.stats as stats, sklearn components, warnings, plotly.subplots.make_subplots, plotly.figure_factory as ff
   - Variables persist between code blocks - you can define variables in one block and use them in subsequent blocks
   - **Do not import anything - everything is already available**

Please provide a step-by-step analysis with code and explanations to thoroughly answer the user's question.
"""
        return prompt
    
    def execute_code_safely(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Safely execute Python code with the dataframe in context
        Returns: dict with 'success', 'result', 'error', 'figures', 'output'
        """
        # Import necessary modules for safe execution
        from sklearn.preprocessing import MinMaxScaler
        from plotly.subplots import make_subplots
        import plotly.figure_factory as ff
        import warnings
        import io
        import sys
        
        # Capture print output
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        # Create a safe execution environment with all necessary imports
        if not self.execution_globals:  # Initialize only once
            self.execution_globals = {
                'df': df,
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'stats': stats,
                'StandardScaler': StandardScaler,
                'MinMaxScaler': MinMaxScaler,
                'KMeans': KMeans,
                'make_subplots': make_subplots,
                'ff': ff,
                'warnings': warnings,
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'max': max,
                    'min': min,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted
                }
            }
        
        # Update df in case it has changed
        self.execution_globals['df'] = df
        
        figures = []
        
        try:
            # Execute the code with persistent locals
            exec(code, self.execution_globals, self.execution_locals)
            
            # Restore stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
            # Capture any figures created
            for var_name, var_value in self.execution_locals.items():
                if hasattr(var_value, 'show') and hasattr(var_value, 'update_layout'):
                    # This is likely a plotly figure
                    figures.append(var_value)
            
            # Also check globals for figures
            for var_name, var_value in self.execution_globals.items():
                if hasattr(var_value, 'show') and hasattr(var_value, 'update_layout'):
                    # This is likely a plotly figure
                    if var_value not in figures:
                        figures.append(var_value)
            
            return {
                'success': True,
                'result': self.execution_locals,
                'error': None,
                'figures': figures,
                'output': output
            }
            
        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = old_stdout
            return {
                'success': False,
                'result': None,
                'error': str(e),
                'figures': [],
                'output': captured_output.getvalue()
            }
    
    def parse_code_blocks(self, response_text: str) -> List[str]:
        """Extract Python code blocks from Claude's response"""
        code_blocks = []
        lines = response_text.split('\n')
        in_code_block = False
        current_block = []
        
        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                current_block = []
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                if current_block:
                    code_blocks.append('\n'.join(current_block))
            elif in_code_block:
                current_block.append(line)
        
        return code_blocks
    
    def get_complete_analysis(self, question: str, df: pd.DataFrame):
        """
        Get complete analysis from Claude (non-streaming for better reliability)
        """
        # Get data summary
        data_summary = self.get_data_summary(df)
        
        # Create the analysis prompt
        prompt = self.create_analysis_prompt(question, data_summary)
        
        # Get response from Claude
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return response.content[0].text
                
        except Exception as e:
            st.error(f"Error communicating with Claude: {str(e)}")
            return None
    
    def display_streaming_response(self, response_text: str, df: pd.DataFrame):
        """
        Display Claude's response as it streams, executing code blocks
        """
        # Split response into text and code sections
        parts = response_text.split('```python')
        
        # Display the first part (text before any code)
        if parts[0].strip():
            st.markdown(parts[0])
        
        # Process each code block
        for i, part in enumerate(parts[1:], 1):
            if '```' in part:
                # Split code and following text
                code_and_text = part.split('```', 1)
                code = code_and_text[0]
                following_text = code_and_text[1] if len(code_and_text) > 1 else ""
                
                # Display code in an expandable section
                with st.expander(f"📝 Code Block {i}", expanded=False):
                    st.code(code, language='python')
                
                # Execute the code
                if code.strip():
                    execution_result = self.execute_code_safely(code, df)
                    
                    if execution_result['success']:
                        # Display any printed output
                        if execution_result.get('output') and execution_result['output'].strip():
                            st.text(execution_result['output'])
                        
                        # Display any figures created
                        for fig in execution_result['figures']:
                            fig.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(color='black'),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.error(f"Code execution error: {execution_result['error']}")
                        # Also show any output that was captured before the error
                        if execution_result.get('output') and execution_result['output'].strip():
                            st.text("Output before error:")
                            st.text(execution_result['output'])
                
                # Display text following the code block
                if following_text.strip():
                    st.markdown(following_text)
            else:
                # Incomplete code block (still streaming)
                with st.expander(f"📝 Code Block {i} (Loading...)", expanded=False):
                    st.code(part, language='python')
    
    def analyze_question(self, question: str, df: pd.DataFrame):
        """
        Main method to analyze a user question
        """
        # Get complete analysis from Claude
        response = self.get_complete_analysis(question, df)
        
        if response:
            # Display the complete analysis
            self.display_streaming_response(response, df)
            return response
        else:
            st.error("❌ **Analysis failed. Please try again.**")
            return None


def setup_claude_assistant(df: pd.DataFrame, api_key: str):
    """
    Setup Claude data assistant interface
    """
    st.subheader("✨ Claude Data Assistant")
    st.markdown("Ask complex questions about your data and get comprehensive analysis with code")
    
    if not api_key:
        st.warning("Please configure your Anthropic API key in your .env file to use Claude assistant.")
        st.info("Add this to your .env file:")
        st.code("ANTHROPIC_API_KEY=your_anthropic_api_key_here")
        return
    
    # Initialize Claude assistant
    try:
        assistant = ClaudeDataAssistant(api_key)
        
        # Initialize chat history
        if "claude_chat_history" not in st.session_state:
            st.session_state.claude_chat_history = []
        
        # Display chat history
        if st.session_state.claude_chat_history:
            for message in st.session_state.claude_chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    with st.container():
                        st.markdown("**🤖 Claude's Analysis:**")
                        assistant.display_streaming_response(message["content"], df)
                        st.markdown("---")  # Add separator between analyses
        
        # Chat input with form
        with st.form(key="claude_chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_question = st.text_area(
                    "Ask a complex question about your data:",
                    placeholder="e.g., 'Who are our top 3 customers and what strategies can we use to acquire more customers like them?'",
                    height=100,
                    label_visibility="collapsed",
                    key="claude_user_input"
                )
            with col2:
                send_button = st.form_submit_button("Analyze", type="primary")
                clear_button = st.form_submit_button("Clear Chat")
        
        # Handle clear chat
        if clear_button:
            st.session_state.claude_chat_history = []
            st.rerun()
        
        # Handle new question
        if send_button and user_question.strip():
            try:
                # Add user message to history
                st.session_state.claude_chat_history.append({
                    "role": "user", 
                    "content": user_question
                })
                
                # Display the user's question immediately
                st.markdown(f'<div class="chat-message-user">{user_question}</div>', unsafe_allow_html=True)
                
                # Get Claude's analysis
                with st.spinner("🧠 Claude is analyzing your data..."):
                    response = assistant.analyze_question(user_question, df)
                
                if response:
                    # Add Claude's response to history
                    st.session_state.claude_chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Don't rerun - the response is already displayed by analyze_question
                
            except Exception as e:
                st.error(f"Error getting Claude analysis: {str(e)}")
                st.info("Please check your Anthropic API configuration and try again.")
    
    except Exception as e:
        st.error(f"Error setting up Claude assistant: {str(e)}")
        st.info("Please verify your Anthropic API configuration in the .env file")