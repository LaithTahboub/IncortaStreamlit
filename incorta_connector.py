"""
Incorta API connector for data retrieval
"""
import requests
import streamlit as st
import pandas as pd
from config import Config

class IncortaConnector:
    def __init__(self):
        self.instance = Config.INCORTA_INSTANCE_URL
        self.tenant = Config.INCORTA_TENANT
        self.pat = Config.INCORTA_PAT
        self.dashboard_id = Config.INCORTA_DASHBOARD_ID
        self.insight_id = Config.INCORTA_INSIGHT_ID
        
        # Different URL patterns to try
        self.url_patterns = [
            "https://{instance}/incorta/api/v2/{tenant}",  # Pattern from working sample
            "https://{instance}/api/v2/{tenant}",           # Alternative pattern
        ]
        
    def test_connection(self):
        """Test connection with different URL patterns and provide diagnostics"""
        if not self.pat:
            return {"success": False, "error": "PAT not configured"}
        
        results = []
        
        # Test basic connectivity first
        for i, pattern in enumerate(self.url_patterns):
            base_url = pattern.format(instance=self.instance, tenant=self.tenant)
            test_url = f"{base_url}/test"  # Simple endpoint to test
            
            try:
                headers = {
                    "Authorization": f"Bearer {self.pat}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                response = requests.get(test_url, headers=headers, timeout=30)
                results.append({
                    "pattern": i + 1,
                    "url": test_url,
                    "status_code": response.status_code,
                    "success": response.status_code < 400,
                    "response_length": len(response.text),
                    "error": None
                })
                
            except Exception as e:
                results.append({
                    "pattern": i + 1,
                    "url": test_url,
                    "status_code": None,
                    "success": False,
                    "response_length": 0,
                    "error": str(e)
                })
        
        return {"success": True, "results": results}

    def get_sqlx_query_response(self, sql_query):
        """Execute SQL query against Incorta using SQLx API"""
        if not self.pat:
            st.error("Incorta PAT (Personal Access Token) not configured")
            return None
            
        try:
            # Use the working pattern from sample code
            url = f"https://{self.instance}/incorta/api/v2/{self.tenant}/sqlxquery"
            headers = {
                "Authorization": f"Bearer {self.pat}",
                "Content-Type": "application/json"
            }
            data = {"sql": sql_query}
            
            response = requests.post(url, headers=headers, json=data, timeout=180)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error executing SQL query: {str(e)}")
            return None
    
    def get_dashboard_prompts(self, dashboard_id=None):
        """Get available prompts/insights for a dashboard"""
        dashboard_id = dashboard_id or self.dashboard_id
        if not all([self.pat, dashboard_id]):
            st.warning("Dashboard ID and PAT required")
            return None
            
        try:
            # Correct URL format: remove /incorta from the path
            url = f"https://{self.instance}/api/v2/{self.tenant}/dashboards/{dashboard_id}/prompts"
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.pat}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error getting dashboard prompts: {str(e)}")
            return None

    def query_dashboard_insight(self, dashboard_id=None, insight_id=None):
        """Query Incorta dashboard insight - using exact pattern from working sample"""
        dashboard_id = dashboard_id or self.dashboard_id
        insight_id = insight_id or self.insight_id
        
        if not all([self.pat, dashboard_id, insight_id]):
            st.warning("Dashboard insight configuration incomplete")
            return None
            
        try:
            # Use EXACT pattern from working sample code
            url = f"https://{self.instance}/incorta/api/v2/{self.tenant}/dashboards/{dashboard_id}/insights/{insight_id}/query"
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.pat}",
                "Content-Type": "application/json"
            }
            payload = {
                "prompts": [],
                "pagination": {}
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error querying dashboard insight: {str(e)}")
            return None
    
    def get_catalog_root(self):
        """Get root catalog with all folders and dashboards - try multiple URL patterns"""
        if not self.pat:
            st.warning("PAT required to browse catalog")
            return None
        
        # Try different URL patterns
        url_attempts = [
            f"https://{self.instance}/incorta/api/v2/{self.tenant}/catalog",  # Pattern from working sample
            f"https://{self.instance}/api/v2/{self.tenant}/catalog",          # Alternative pattern
        ]
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.pat}",
            "Content-Type": "application/json"
        }
        
        for i, url in enumerate(url_attempts):
            try:
                st.info(f"Trying URL pattern {i+1}: {url}")
                response = requests.get(url, headers=headers, timeout=30)
                
                st.info(f"Status Code: {response.status_code}")
                st.info(f"Response Headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    st.success(f"✅ Success with pattern {i+1}!")
                    return response.json()
                elif response.status_code == 401:
                    st.error(f"❌ Pattern {i+1}: Authentication failed - check PAT token")
                elif response.status_code == 403:
                    st.error(f"❌ Pattern {i+1}: Access forbidden - check permissions")
                elif response.status_code == 404:
                    st.warning(f"⚠️ Pattern {i+1}: Endpoint not found")
                elif response.status_code == 503:
                    st.error(f"❌ Pattern {i+1}: Service unavailable")
                else:
                    st.warning(f"⚠️ Pattern {i+1}: HTTP {response.status_code}")
                    
                # Try to get response text for debugging
                try:
                    response_text = response.text[:500]  # First 500 chars
                    if response_text:
                        st.code(f"Response preview: {response_text}")
                except:
                    pass
                    
            except requests.exceptions.Timeout:
                st.error(f"❌ Pattern {i+1}: Request timeout")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Pattern {i+1}: Connection error")
            except Exception as e:
                st.error(f"❌ Pattern {i+1}: {str(e)}")
        
        st.error("All URL patterns failed")
        return None
    
    def search_catalog(self, search_term):
        """Search for dashboards or folders"""
        if not self.pat:
            st.warning("PAT required to search catalog")
            return None
            
        try:
            url = f"https://{self.instance}/api/v2/{self.tenant}/catalog/search"
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.pat}",
                "Content-Type": "application/json"
            }
            params = {"q": search_term}
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error searching catalog: {str(e)}")
            return None

def convert_incorta_to_dataframe(incorta_response):
    """Convert Incorta API response to pandas DataFrame"""
    if not incorta_response:
        return None
        
    try:
        headers = incorta_response.get('headers', {})
        data = incorta_response.get('data', [])
        
        # Extract column names
        dimensions = headers.get('dimensions', [])
        measures = headers.get('measures', [])
        
        columns = []
        for dim in dimensions:
            columns.append(dim.get('label', f"Dimension_{dim.get('index', 0)}"))
        for measure in measures:
            columns.append(measure.get('label', f"Measure_{measure.get('index', 0)}"))
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        return df
    except Exception as e:
        st.error(f"Error converting Incorta data: {str(e)}")
        return None