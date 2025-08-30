import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .section-header {
            font-size: 2rem;
            color: #ff7f0e;
            margin: 2rem 0 1rem 0;
            border-bottom: 3px solid #ff7f0e;
            padding-bottom: 0.5rem;
        }
        .subsection-header {
            font-size: 1.5rem;
            color: #2ca02c;
            margin: 1.5rem 0 1rem 0;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #d4edda;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #f8d7da;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #dc3545;
            margin: 1rem 0;
        }
        .metric-container {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .sidebar-header {
            color: #1f77b4;
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        /* Custom button styling */
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border-radius: 20px;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0d47a1;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        /* Hide Streamlit footer */
        .css-1rs6os {
            display: none;
        }
        /* Page title styling */
        .page-title {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
            padding: 1rem;
            background: linear-gradient(90deg, #f0f2f6, #ffffff, #f0f2f6);
            border-radius: 10px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

def create_info_box(title, content, box_type="info"):
    """Create styled info boxes"""
    box_class = f"{box_type}-box"
    st.markdown(f"""
    <div class="{box_class}">
    <h3>{title}</h3>
    {content}
    </div>
    """, unsafe_allow_html=True)

def create_metric_box(title, value, description=""):
    """Create styled metric boxes"""
    st.markdown(f"""
    <div class="metric-container">
    <h4 style="color: #1f77b4; margin: 0 0 0.5rem 0;">{title}</h4>
    <h2 style="color: #2ca02c; margin: 0 0 0.5rem 0;">{value}</h2>
    <p style="color: #666; margin: 0; font-size: 0.9rem;">{description}</p>
    </div>
    """, unsafe_allow_html=True)