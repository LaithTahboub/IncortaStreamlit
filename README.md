# Data Analytics Dashboard

A Streamlit application for data analysis with AI-powered chat using PandasAI and OpenAI.

## Features

- **📊 Dashboard**: Interactive data visualization and analysis
- **🤖 AI Chat**: Ask questions about your data using PandasAI + OpenAI
- **🔗 Incorta Integration**: Connect to Incorta for enterprise data
- **📈 Analytics**: Statistical analysis, correlations, and data quality reports

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4
   
   # Optional: Incorta configuration
   INCORTA_INSTANCE_URL=your_incorta_instance
   INCORTA_TENANT=your_tenant
   INCORTA_PAT=your_personal_access_token
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Dashboard**: Load data (sample, SQL query, or Incorta insight)
2. **AI Chat**: Ask questions about your data in natural language

## Dependencies

- streamlit
- pandas
- plotly
- pandasai (latest)
- openai
- requests
- python-dotenv