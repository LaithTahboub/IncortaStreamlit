"""
Configuration settings for the Streamlit application
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Incorta Configuration
    INCORTA_INSTANCE_URL = os.getenv("INCORTA_INSTANCE_URL", "se-prod-demo.cloud4.incorta.com")
    INCORTA_TENANT = os.getenv("INCORTA_TENANT", "demo")
    INCORTA_PAT = os.getenv("INCORTA_PAT", "")
    INCORTA_DASHBOARD_ID = os.getenv("INCORTA_DASHBOARD_ID", "")
    INCORTA_INSIGHT_ID = os.getenv("INCORTA_INSIGHT_ID", "")
    
    # App Configuration
    PAGE_TITLE = "Data Analytics Dashboard"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    SAMPLE_DATA_PATH = ""