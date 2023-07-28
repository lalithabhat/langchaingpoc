import streamlit as st
import pandas as pd
import indexes

col1, col2 = st.columns([3,2])


with col1:
    st.title("Lead routing engine")
    lead_form = st.text_area("describe your requirements here", height=300)
    if st.button("Search Offering"):
        response = indexes.searchServices(lead_form)

with col2:
    st.markdown("<br/>" * 5, unsafe_allow_html=True)  # Creates 5 lines of vertical space
    st.write(response)