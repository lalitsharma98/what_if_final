import streamlit as st
from data_extractor import run_data_extractor
from fte_analysis import run_fte_analysis
from default import main
from DatamartDaywise1 import daywise


# App title
# st.set_page_config(page_title="What-if Analysis", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["What-if analysis","Data Extractor", "FTE Req Analysis","Datamart-Daywise"])

# Main content based on selection
if page == "What-if analysis":
    st.title("What-If Analysis Tool")
    main()

elif page == "Data Extractor":
    st.title("Data Extractor")
    run_data_extractor()

elif page == "FTE Req Analysis":
    st.title("FTE Requirement Analysis")
    run_fte_analysis()

elif page == "Datamart-Daywise":
    st.title("Datamart Daywise")
    daywise()