"""
Controlled Claude Data Assistant - No code blocks, structured analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import anthropic
import io
import sys

class ControlledClaudeAssistant:
    """
    Controlled data assistant using Claude with structured analysis workflow
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
    
    def create_initial_prompt(self, question: str, data_summary: str) -> str:
        """Create initial prompt to get Claude's analysis plan"""
        prompt = f"""You are an expert data analyst. You have access to a dataset with the following structure:

{data_summary}

The user has asked: "{question}"

Please respond with EXACTLY this format:

ANALYSIS: [One sentence describing what analysis you'll perform]

CODE:
[Pure Python code using pandas, numpy, plotly, scipy.stats, sklearn - NO IMPORT STATEMENTS]

The code should:
- Use the dataframe 'df' that's already available
- Perform the necessary analysis to answer the question
- Store key results in variables for later explanation
- ALWAYS include print statements to show results
- Print the final answer/findings in a clear format
- Create visualizations if helpful (using px or go)

IMPORTANT: Make sure to print() the key results so they can be seen in the output!

IMPORTANT: 
- Do NOT include any import statements (pandas is pd, numpy is np, plotly.express is px, etc.)
- All libraries are already imported and available
- Keep code focused and efficient
- Store important results in clearly named variables"""
        
        return prompt
    
    def create_final_prompt(self, question: str, code_output: str, conversation_history: list = None) -> str:
        """Create final prompt to get Claude's conclusion"""
        
        # Add conversation context if available
        context = ""
        if conversation_history:
            context = "\n\nPrevious conversation context:\n"
            for exchange in conversation_history[-3:]:  # Last 3 exchanges for context
                context += f"Q: {exchange['question']}\nA: {exchange['result'].get('final_answer', 'No answer')}\n\n"
        
        prompt = f"""Based on the code execution results below, please answer the user's question: "{question}"

{context}
Code Output:
{code_output}

IMPORTANT INSTRUCTIONS:
- Be as brief as possible while still answering the question completely  
- Use ONLY plain text - absolutely no markdown, latex, special characters, or formatting
- Do not use dollar signs, asterisks, underscores, backticks, or any special symbols
- Write numbers and currency as simple text (example: "1000 dollars" not "$1,000")
- Use simple sentences with no special formatting whatsoever
- Focus on actionable insights and key findings
- Keep your response concise and direct"""
        
        return prompt
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract code section from Claude's response"""
        # Look for CODE: section
        code_match = re.search(r'CODE:\s*\n(.*?)(?=\n\n|$)', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            # Remove any accidental code block markers
            code = re.sub(r'```python\s*\n?', '', code)
            code = re.sub(r'```\s*$', '', code)
            # Remove any import statements just to be safe
            lines = code.split('\n')
            filtered_lines = []
            for line in lines:
                if not line.strip().startswith(('import ', 'from ')):
                    filtered_lines.append(line)
            return '\n'.join(filtered_lines)
        return ""
    
    def execute_code_safely(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute Python code safely and capture output"""
        # Import necessary modules for safe execution
        from sklearn.preprocessing import MinMaxScaler
        from plotly.subplots import make_subplots
        import plotly.figure_factory as ff
        import warnings
        
        # Capture print output
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        
        # Create execution environment if not exists
        if not self.execution_globals:
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
                'sorted': sorted,
                'any': any,
                'all': all
            }
        
        # Update df and redirect stdout
        self.execution_globals['df'] = df
        sys.stdout = captured_output
        
        figures = []
        
        try:
            # Execute code in combined namespace
            combined_namespace = {**self.execution_globals, **self.execution_locals}
            exec(code, combined_namespace)
            
            # Update the locals with any new variables created
            for key, value in combined_namespace.items():
                if key not in self.execution_globals:
                    self.execution_locals[key] = value
            
            # Restore stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
            # Capture figures from both namespaces
            for var_name, var_value in combined_namespace.items():
                if hasattr(var_value, 'show') and hasattr(var_value, 'update_layout'):
                    if var_value not in figures:
                        figures.append(var_value)
            
            return {
                'success': True,
                'output': output,
                'figures': figures,
                'error': None
            }
            
        except Exception as e:
            sys.stdout = old_stdout
            return {
                'success': False,
                'output': captured_output.getvalue(),
                'figures': [],
                'error': str(e)
            }
    
    def analyze_question_streaming(self, question: str, df: pd.DataFrame, progress_container, conversation_history=None):
        """Main controlled analysis workflow with streaming"""
        
        # Step 1: Get data summary
        data_summary = self.get_data_summary(df)
        
        # Step 2: Stream Claude's initial analysis plan
        initial_prompt = self.create_initial_prompt(question, data_summary)
        
        with progress_container.container():
            st.markdown("**AI is thinking...**")
        
        try:
            # Stream the initial response
            with self.client.messages.stream(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,
                messages=[{"role": "user", "content": initial_prompt}]
            ) as stream:
                
                initial_text = ""
                current_analysis = ""
                current_code = ""
                
                for text in stream.text_stream:
                    initial_text += text
                    
                    # Extract analysis description as it streams
                    analysis_match = re.search(r'ANALYSIS:\s*(.*?)(?=\n|CODE:|$)', initial_text, re.DOTALL)
                    if analysis_match:
                        current_analysis = analysis_match.group(1).strip()
                    
                    # Extract code as it streams
                    current_code = self.extract_code_from_response(initial_text)
                    
                    # Update display in real-time
                    with progress_container.container():
                        if current_analysis:
                            st.markdown(f"**Analysis Plan:** {current_analysis}")
                        
                        if current_code:
                            st.markdown("**Writing Code...**")
                            with st.expander("Live Code Generation", expanded=True):
                                st.code(current_code, language='python')
            
            # Extract final analysis and code
            analysis_match = re.search(r'ANALYSIS:\s*(.*?)(?=\n|CODE:)', initial_text, re.DOTALL)
            analysis_description = analysis_match.group(1).strip() if analysis_match else "Analyzing data..."
            
            code = self.extract_code_from_response(initial_text)
            
            if not code:
                with progress_container.container():
                    st.error("No code generated")
                return {
                    'success': False,
                    'error': 'No code found in Claude response',
                    'analysis_description': analysis_description
                }
            
            # Step 3: Execute the code
            with progress_container.container():
                st.markdown("**Executing code...**")
            
            execution_result = self.execute_code_safely(code, df)
            
            if not execution_result['success']:
                with progress_container.container():
                    st.error(f"Code execution failed: {execution_result['error']}")
                return {
                    'success': False,
                    'error': f"Code execution failed: {execution_result['error']}",
                    'analysis_description': analysis_description,
                    'code': code
                }
            
            # Show execution results immediately
            with progress_container.container():
                st.markdown("**Code executed successfully**")
                
                # Debug: Show what we captured
                if execution_result['output'].strip():
                    st.markdown("**Code Output:**")
                    st.text(execution_result['output'])
                else:
                    # Don't show analysis results section - it's not needed for the user
                    pass
                
                # Show any figures
                for fig in execution_result['figures']:
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='black'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Step 4: Stream Claude's final conclusion
            # Create comprehensive data summary including variables if print output is empty
            if execution_result['output'].strip():
                data_for_prompt = execution_result['output']
            else:
                # Use variable contents since print didn't work
                var_summary = []
                for var_name, var_value in list(self.execution_locals.items())[:5]:
                    if not var_name.startswith('_'):
                        try:
                            if hasattr(var_value, 'head'):  # DataFrame or Series
                                var_summary.append(f"{var_name}:\n{var_value.head(10).to_string()}")
                            elif isinstance(var_value, (list, tuple, dict)) and len(str(var_value)) < 500:
                                var_summary.append(f"{var_name}: {var_value}")
                            elif isinstance(var_value, (int, float, str)) and len(str(var_value)) < 200:
                                var_summary.append(f"{var_name}: {var_value}")
                        except Exception:
                            pass
                data_for_prompt = "\n".join(var_summary) if var_summary else "No data captured"
            
            final_prompt = self.create_final_prompt(question, data_for_prompt, conversation_history)
            
            with progress_container.container():
                st.markdown("**AI is formulating the final answer...**")
            
            with self.client.messages.stream(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": final_prompt}]
            ) as stream:
                
                final_answer = ""
                answer_container = progress_container.empty()
                
                for text in stream.text_stream:
                    final_answer += text
                    
                    # Update answer in real-time
                    with answer_container.container():
                        st.markdown("**Final Answer:**")
                        st.text(final_answer)
            
            return {
                'success': True,
                'analysis_description': analysis_description,
                'code': code,
                'code_output': execution_result['output'],
                'figures': execution_result['figures'],
                'final_answer': final_answer,
                'error': None
            }
            
        except Exception as e:
            with progress_container.container():
                st.error(f"Error: {str(e)}")
            return {
                'success': False,
                'error': f"Error communicating with Claude: {str(e)}",
                'analysis_description': "Analysis failed"
            }


def setup_controlled_claude_assistant(df: pd.DataFrame, api_key: str):
    """Setup controlled Claude assistant interface"""
    st.markdown("Ask questions about your data and get structured analysis")
    
    if not api_key:
        st.warning("Please configure your Anthropic API key in your .env file to use the assistant.")
        st.info("Add this to your .env file:")
        st.code("ANTHROPIC_API_KEY=your_anthropic_api_key_here")
        return
    
    try:
        assistant = ControlledClaudeAssistant(api_key)
        
        # Initialize chat history
        if "controlled_claude_history" not in st.session_state:
            st.session_state.controlled_claude_history = []
        
        # Display chat history
        if st.session_state.controlled_claude_history:
            for exchange in st.session_state.controlled_claude_history:
                # User question
                st.markdown(f'<div class="chat-message-user">{exchange["question"]}</div>', unsafe_allow_html=True)
                
                # AI analysis
                with st.container():
                    st.markdown("**AI Analysis:**")
                    
                    if exchange["result"]["success"]:
                        # Show analysis description
                        st.markdown(f"*{exchange['result']['analysis_description']}*")
                        
                        # Show code in expander
                        with st.expander("View Analysis Code", expanded=False):
                            st.code(exchange["result"]["code"], language='python')
                        
                        # Show code output if any
                        if exchange["result"]["code_output"].strip():
                            st.text(exchange["result"]["code_output"])
                        
                        # Show figures
                        for fig in exchange["result"]["figures"]:
                            fig.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(color='black'),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show final answer
                        st.markdown("**Answer:**")
                        st.text(exchange['result']['final_answer'])
                    
                    else:
                        st.error(f"Analysis failed: {exchange['result']['error']}")
                    
                    st.markdown("---")
        
        # Chat input
        with st.form(key="controlled_claude_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_question = st.text_area(
                    "Ask a question about your data:",
                    placeholder="e.g., 'Who are the top 3 customers by total sales and what makes them valuable?'",
                    height=80,
                    label_visibility="collapsed",
                    key="controlled_claude_input"
                )
            with col2:
                analyze_button = st.form_submit_button("Analyze", type="primary")
                clear_button = st.form_submit_button("Clear Chat")
        
        # Handle clear
        if clear_button:
            st.session_state.controlled_claude_history = []
            st.rerun()
        
        # Handle new question
        if analyze_button and user_question.strip():
            try:
                # Display user question immediately
                st.markdown(f'<div class="chat-message-user">{user_question}</div>', unsafe_allow_html=True)
                
                # Create progress container for streaming
                progress_container = st.empty()
                
                # Analyze with Claude using streaming
                result = assistant.analyze_question_streaming(user_question, df, progress_container, st.session_state.controlled_claude_history)
                
                # Add to history
                st.session_state.controlled_claude_history.append({
                    "question": user_question,
                    "result": result
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    except Exception as e:
        st.error(f"Error setting up Claude assistant: {str(e)}")
        st.info("Please verify your Anthropic API configuration in the .env file")