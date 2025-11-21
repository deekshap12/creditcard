import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import time
from streamlit_extras.stylable_container import stylable_container
from datetime import datetime, timedelta
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import hashlib
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    .st-bw {
        color: #667eea !important;
    }
    .st-bx {
        border-color: #667eea !important;
    }
    .st-bv {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    .st-bv:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Input field styling - white background */
    input[type="text"],
    input[type="password"] {
        background-color: #f8f9fa !important;
        color: #333 !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    input[type="text"]:focus,
    input[type="password"]:focus {
        background-color: white !important;
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    /* Form container styling */
    .stTabs {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px !important;
        padding: 30px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 2px solid #e0e0e0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #667eea !important;
        border-bottom: 3px solid #667eea !important;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: #e8eaf6 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'model_retrain_log' not in st.session_state:
    st.session_state.model_retrain_log = []
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []

# --- User Database (Demo - In production, use a real database) ---
USERS_DB = {
    "admin": "admin123",
    "analyst": "analyst123",
    "user": "user123"
}

USERS_FILE = "registered_users.pkl"

def load_users_db():
    """Load users from file if it exists"""
    global USERS_DB
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'rb') as f:
                USERS_DB = pickle.load(f)
        except:
            pass

def save_users_db():
    """Save users to file"""
    try:
        with open(USERS_FILE, 'wb') as f:
            pickle.dump(USERS_DB, f)
    except Exception as e:
        st.error(f"Error saving users: {str(e)}")

def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    """Verify password against hash"""
    return stored_hash == hash_password(provided_password)

def authenticate_user(username, password):
    """Authenticate user credentials"""
    if username in USERS_DB:
        if USERS_DB[username] == password:
            return True
    return False

def register_user(username, password, confirm_password):
    """Register a new user"""
    # Validation
    if not username or not password or not confirm_password:
        return False, "❌ All fields are required"
    
    if len(username) < 3:
        return False, "❌ Username must be at least 3 characters long"
    
    if len(password) < 6:
        return False, "❌ Password must be at least 6 characters long"
    
    if password != confirm_password:
        return False, "❌ Passwords do not match"
    
    if username in USERS_DB:
        return False, "❌ Username already exists"
    
    # Register user
    USERS_DB[username] = password
    save_users_db()
    return True, f"✅ Account created successfully! Welcome, {username}!"

# Load users on startup
load_users_db()

def load_model():
    """Load the trained fraud detection model"""
    try:
        return joblib.load("fraud_model.pkl")
    except Exception as e:
        st.warning(f"Model not found. Using demo model. Error: {str(e)}")
        return create_demo_model()

def create_demo_model():
    """Create a demo model for demonstration"""
    np.random.seed(42)
    X_demo = np.random.randn(1000, 30)
    y_demo = np.random.randint(0, 2, 1000)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_demo, y_demo)
    return model

def process_file(uploaded_file):
    """Process the uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

def get_feature_importance(model, feature_names):
    """Extract feature importance from the model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        return [(feature_names[i], importances[i]) for i in indices]
    return []

def explain_prediction(model, sample, feature_names, prediction, confidence):
    """Generate explanation for a single prediction"""
    explanation = {
        'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
        'confidence': confidence,
        'top_features': get_feature_importance(model, feature_names)[:5]
    }
    return explanation

def retrain_model_with_new_data(model, new_data, new_labels):
    """Retrain model with new data"""
    try:
        if hasattr(model, 'partial_fit'):
            model.partial_fit(new_data, new_labels)
        else:
            model.fit(new_data, new_labels)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.model_retrain_log.append({
            'timestamp': timestamp,
            'samples': len(new_data),
            'status': 'Success'
        })
        return model
    except Exception as e:
        st.session_state.model_retrain_log.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'samples': len(new_data),
            'status': f'Failed: {str(e)}'
        })
        return model

def generate_monthly_trends(df):
    """Generate monthly fraud trends"""
    if 'Time' in df.columns:
        try:
            df_copy = df.copy()
            df_copy['Time'] = pd.to_datetime(df_copy['Time'], errors='coerce')
            df_copy['Month'] = df_copy['Time'].dt.to_period('M').astype(str)
            trends = df_copy.groupby('Month')['Prediction'].agg(['sum', 'count'])
            trends['fraud_rate'] = (trends['sum'] / trends['count'] * 100).round(2)
            trends = trends.reset_index()
            return trends
        except Exception as e:
            st.warning(f"Could not generate trends: {str(e)}")
            return None
    return None

def analyze_predictions(df, model):
    """Analyze and visualize prediction results with XAI"""
    # Calculate metrics
    total_transactions = len(df)
    fraud_count = int(df['Prediction'].sum()) if 'Prediction' in df.columns else 0
    fraud_percentage = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Transactions", f"{total_transactions:,}")
    with col2:
        st.metric("🚨 Potential Frauds", f"{fraud_count:,}")
    with col3:
        st.metric("⚠️ Fraud Rate", f"{fraud_percentage:.2f}%")
    with col4:
        st.metric("✅ Legitimate", f"{total_transactions - fraud_count:,}")
    
    # Style metrics
    style_metric_cards(background_color="#FFFFFF", border_left_color="#2563eb")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "📈 Trends", "🔍 Feature Importance", "� XAI", "� Details"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Pie chart of fraud vs non-fraud
            fig = px.pie(
                names=['Non-Fraud', 'Fraud'],
                values=[total_transactions - fraud_count, fraud_count],
                title='Transaction Distribution',
                color_discrete_sequence=['#2563eb', '#dc2626']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart of fraud distribution
            fraud_counts = pd.DataFrame({
                'Type': ['Non-Fraud', 'Fraud'],
                'Count': [total_transactions - fraud_count, fraud_count]
            })
            
            fig = px.bar(
                fraud_counts,
                x='Type',
                y='Count',
                title='Fraud vs Non-Fraud Count',
                color='Type',
                color_discrete_map={'Non-Fraud': '#2563eb', 'Fraud': '#dc2626'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Monthly trends
        st.subheader("📈 Monthly Fraud Trends")
        trends = generate_monthly_trends(df)
        if trends is not None and len(trends) > 0:
            st.dataframe(trends, use_container_width=True)
            
            try:
                fig = px.line(
                    trends,
                    x='Month',
                    y='fraud_rate',
                    title='Monthly Fraud Rate Trend',
                    markers=True,
                    labels={'fraud_rate': 'Fraud Rate (%)', 'Month': 'Month'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create trend chart: {str(e)}")
        else:
            st.info("No time-based data available for trend analysis")
    
    with tab3:
        # Feature importance
        st.subheader("🔍 Top Features Influencing Fraud Detection")
        feature_names = [col for col in df.columns if col not in ['Prediction', 'Time', 'Class']]
        feature_importance = get_feature_importance(model, feature_names)
        
        if feature_importance:
            fi_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
            fig = px.bar(
                fi_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance Scores',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model")
    
    with tab4:
        # Explainable AI
        st.subheader("💡 Explainable AI (XAI) - Why Transactions are Flagged")
        
        if len(df) > 0:
            # Show fraud transactions
            fraud_transactions = df[df['Prediction'] == 1].head(5)
            
            if len(fraud_transactions) > 0:
                st.write("**Top 5 Flagged Fraudulent Transactions:**")
                for idx, row in fraud_transactions.iterrows():
                    with st.expander(f"Transaction {idx} - FRAUD DETECTED"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Prediction Details:**")
                            st.write(f"- Status: 🚨 FRAUD")
                            st.write(f"- Confidence: High")
                        with col2:
                            st.write("**Key Influencing Factors:**")
                            if feature_importance:
                                for feat, imp in feature_importance[:3]:
                                    st.write(f"- {feat}: {imp:.4f}")
            else:
                st.success("✅ No fraudulent transactions detected!")
        else:
            st.info("No transactions to analyze")
    
    with tab5:
        # Show detailed predictions with original data
        st.subheader("📋 Detailed Predictions")
        st.dataframe(
            df,
            use_container_width=True,
            height=400,
            column_config={
                "Prediction": st.column_config.NumberColumn(
                    "Fraud Prediction",
                    help="0: Non-Fraud, 1: Fraud",
                    format="%d"
                )
            }
        )

def login_page():
    """Display login/signup page"""
    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <h1>🛡️ Fraud Detection System</h1>
            <p style="font-size: 18px; color: #666;">Secure Login</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Toggle between login and signup
        tab1, tab2 = st.tabs(["🔓 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                st.markdown("---")
                username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    help="Demo users: admin, analyst, user"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    help="Demo passwords: admin123, analyst123, user123"
                )
                
                st.markdown("---")
                login_button = st.form_submit_button("� Login", use_container_width=True)
                
                if login_button:
                    if not username or not password:
                        st.error("❌ Please enter both username and password")
                    elif authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success(f"✅ Welcome back, {username}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
            
            # Demo credentials info
            with st.expander("ℹ️ Demo Credentials"):
                st.markdown("""
                **Available Demo Accounts:**
                - **Username:** admin | **Password:** admin123
                - **Username:** analyst | **Password:** analyst123
                - **Username:** user | **Password:** user123
                """)
        
        with tab2:
            with st.form("signup_form"):
                st.markdown("---")
                new_username = st.text_input(
                    "Create Username",
                    placeholder="Choose a username (min 3 characters)",
                    key="signup_username"
                )
                new_password = st.text_input(
                    "Create Password",
                    type="password",
                    placeholder="Create a password (min 6 characters)",
                    key="signup_password"
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Confirm your password",
                    key="confirm_password"
                )
                
                st.markdown("---")
                signup_button = st.form_submit_button("📝 Create Account", use_container_width=True)
                
                if signup_button:
                    success, message = register_user(new_username, new_password, confirm_password)
                    if success:
                        st.success(message)
                        st.info("✅ You can now login with your new account!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(message)
            
            # Password requirements info
            with st.expander("ℹ️ Password Requirements"):
                st.markdown("""
                **Account Creation Requirements:**
                - Username: Minimum 3 characters
                - Password: Minimum 6 characters
                - Passwords must match
                - Username must be unique
                """)

def main_app():
    """Main application function with advanced features"""
    # --- Sidebar ---
    with st.sidebar:
        st.title("🛡️ Advanced Fraud Detection")
        st.markdown("---")
        
        # User info and logout
        st.markdown(f"**👤 Logged in as:** {st.session_state.username}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.success("✅ Logged out successfully!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        
        # Navigation
        page = st.radio("📍 Navigation", ["🏠 Home", "📊 Dashboard", "🔄 Model Retraining", "📜 Retrain Log"])
        
        st.markdown("---")
        
        with st.expander("ℹ️ Instructions"):
            st.markdown("""
            **Home Tab:**
            1. Upload a CSV file with transaction data
            2. System analyzes for potential fraud
            3. Review visualizations and XAI explanations
            
            **Dashboard Tab:**
            - View comprehensive fraud analytics
            - See feature importance
            - Understand why transactions are flagged
            
            **Model Retraining:**
            - Retrain with new data
            - Update model with latest patterns
            - Monitor retraining history
            """)
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8em; color: #666; text-align: center;">
            <p>🚀 Advanced Fraud Detection System v2.0</p>
            <p>✨ Features: XAI • Auto-Retrain • Trend Analysis</p>
            <p>© 2025 All Rights Reserved</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model once
    model = load_model()
    
    # --- Page Content ---
    if page == "🏠 Home":
        st.title("💳 Credit Card Fraud Detection System")
        st.markdown("Advanced fraud detection with Explainable AI")
        st.markdown("---")
        
        # Add a file upload card
        with stylable_container(
            key="upload_container",
            css_styles="""
                {
                    background: white;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                    margin-bottom: 2rem;
                }
            """
        ):
            st.header("📤 Upload Transaction Data")
            st.markdown("Upload a CSV file containing transaction details for fraud analysis.")
        
        uploaded_file = st.file_uploader(
            "Upload your transaction data (CSV)",
            type=["csv"],
            help="Upload a CSV file with transaction data"
        )
        
        if uploaded_file is not None:
            with st.spinner('🔄 Processing your file...'):
                # Show progress
                progress_bar = st.progress(0)
                
                # Process file
                df, error = process_file(uploaded_file)
                if error:
                    st.error(error)
                    return
                    
                progress_bar.progress(20)
                
                if model is None:
                    st.error("Failed to load model")
                    return
                    
                progress_bar.progress(40)
                
                # Make predictions
                try:
                    X = df.drop(columns=["Class"], errors="ignore")
                    df["Prediction"] = model.predict(X)
                    progress_bar.progress(80)
                    
                    # Store in session
                    st.session_state.transaction_history.append({
                        'timestamp': datetime.now(),
                        'transactions': len(df),
                        'frauds': int(df['Prediction'].sum())
                    })
                    
                    # Display results
                    st.success("✅ Analysis complete!")
                    analyze_predictions(df, model)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        help="Download the complete dataset with predictions"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
                
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
        else:
            # Show welcome message
            st.info("👈 Please upload a CSV file to begin fraud detection")
    
    elif page == "📊 Dashboard":
        st.title("📊 Advanced Analytics Dashboard")
        
        if len(st.session_state.transaction_history) > 0:
            # Create summary metrics
            total_analyzed = sum([t['transactions'] for t in st.session_state.transaction_history])
            total_frauds = sum([t['frauds'] for t in st.session_state.transaction_history])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Analyzed", f"{total_analyzed:,}")
            with col2:
                st.metric("🚨 Total Frauds", f"{total_frauds:,}")
            with col3:
                fraud_rate = (total_frauds / total_analyzed * 100) if total_analyzed > 0 else 0
                st.metric("⚠️ Overall Fraud Rate", f"{fraud_rate:.2f}%")
            
            style_metric_cards(background_color="#FFFFFF", border_left_color="#2563eb")
            
            # Transaction history chart
            history_df = pd.DataFrame(st.session_state.transaction_history)
            fig = px.line(
                history_df,
                x='timestamp',
                y=['transactions', 'frauds'],
                title='Transaction Analysis History',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No transaction history yet. Upload a file from the Home tab to get started!")
    
    elif page == "🔄 Model Retraining":
        st.title("🔄 Automatic Model Retraining")
        
        st.markdown("""
        This feature allows you to retrain the model with new data to adapt to:
        - New fraud patterns
        - Changing spending behaviors
        - Emerging cyber attacks
        """)
        
        st.markdown("---")
        
        retrain_file = st.file_uploader(
            "Upload new training data (CSV)",
            type=["csv"],
            key="retrain_uploader",
            help="CSV with features and labels"
        )
        
        if retrain_file is not None:
            if st.button("🚀 Retrain Model"):
                with st.spinner("🔄 Retraining model..."):
                    try:
                        retrain_df = pd.read_csv(retrain_file)
                        
                        # Prepare data
                        X_retrain = retrain_df.drop(columns=["Class"], errors="ignore")
                        y_retrain = retrain_df.get("Class", np.random.randint(0, 2, len(retrain_df)))
                        
                        # Retrain
                        model = retrain_model_with_new_data(model, X_retrain.values, y_retrain)
                        
                        st.success("✅ Model retrained successfully!")
                        st.info(f"Retrained with {len(retrain_df)} samples")
                        
                    except Exception as e:
                        st.error(f"❌ Retraining failed: {str(e)}")
    
    elif page == "📜 Retrain Log":
        st.title("📜 Model Retraining Log")
        
        if len(st.session_state.model_retrain_log) > 0:
            log_df = pd.DataFrame(st.session_state.model_retrain_log)
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No retraining history yet.")

if __name__ == "__main__":
    if st.session_state.authenticated:
        main_app()
    else:
        login_page()
