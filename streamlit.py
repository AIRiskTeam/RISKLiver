import streamlit as st
from Estimator.streamlit import calculate_descriptors


# Init
if 'is_busy' not in st.session_state:
    st.session_state.is_busy = False


# Title
st.title("RISKLiver for Streamlit")
st.text("RiskLiver is a ML-based tool to predict hazard levels and potency for drug-induced liver injury(DILI).")
st.divider()


# Input
@st.cache_data
def _calculate_descriptors(input_str:str):
    return calculate_descriptors(input_str)

def submit_click():
    # click on submit button
    st.session_state.is_busy = True

    desc_df = _calculate_descriptors(input_str=st.session_state['smiles_input'])
    with st.container():
        if len(desc_df) > 0:
        st.text("The following SMILES are ready for prediction.")
        st.dataframe(desc_df, key="descriptors_data")

with st.container():
    st.header("Input smiles")
    smiles_input = st.text_area(
        placeholder="One SMILES per line", 
        key="smiles_input"
    )
    st.text("Warning: invalid SMILES would be ignored, please check out the ouputs.")
    st.button(
        label="Please wait..." if st.session_state.is_busy else "Calculate descriptors",
        width="stretch",
        key="submit_button",
        on_click=submit_click,
        disabled=st.session_state.is_busy,
    )


# Hazard Identification
with st.container():