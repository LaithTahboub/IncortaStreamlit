"""
Data utilities and sample data loading
"""
import pandas as pd
import streamlit as st
from config import Config

@st.cache_data
def load_sample_data():
    """Load sample supply chain data"""
    try:
        if Config.SAMPLE_DATA_PATH and Config.SAMPLE_DATA_PATH.strip():
            df = pd.read_csv(Config.SAMPLE_DATA_PATH)
        else:
            raise FileNotFoundError("No sample data path configured")
    except:
        # Fallback sample data - expanded dataset
        data = {
            'Site': ['Chicago', 'Chicago', 'Phoenix', 'Chicago', 'Phoenix', 'Miami', 'Chicago', 'Phoenix', 'Miami', 'Phoenix', 'Chicago', 'Miami'],
            'Vendor': ['FiberOptic Solutions 2', 'FiberOptic Solutions 2', 'PowerGrid Supplies 17', 'FiberOptic Solutions 2', 'PowerGrid Supplies 17', 'Unknown', 'TechCorp Inc', 'PowerGrid Supplies 17', 'FiberOptic Solutions 2', 'TechCorp Inc', 'Unknown', 'PowerGrid Supplies 17'],
            'Order_Number': ['PO000296', 'PO000296', 'PO000275', 'PO000296', 'PO000275', 'TO000213', 'PO000298', 'PO000299', 'PO000300', 'PO000301', 'TO000214', 'PO000302'],
            'Status': ['Picked', 'Backordered', 'Picked', 'Picked', 'Open', 'Backordered', 'Picked', 'Open', 'Picked', 'Backordered', 'Picked', 'Open'],
            'Category': ['Tools', 'Network Equipment', 'Tools', 'Tools', 'Construction Materials', 'Safety Equipment', 'Network Equipment', 'Construction Materials', 'Tools', 'Safety Equipment', 'Network Equipment', 'Tools'],
            'Item': ['Safety Helmet 88', 'Keyboard 12', 'MacBook Pro 131', 'Concrete Mix 119', 'USB Drive 125', 'Steel Rebar 137', 'Router 45', 'Cement Bags 200', 'Drill Set 67', 'Hard Hat 23', 'Switch 78', 'Hammer 34'],
            'Amount': [702.24, 13485.52, 4719.72, 482.38, 2828.25, 574.64, 8950.30, 1245.67, 890.45, 156.78, 5670.89, 234.56]
        }
        df = pd.DataFrame(data)
    return df