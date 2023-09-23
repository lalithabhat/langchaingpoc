import streamlit as st
import pandas as pd
import indexes

col1, col2 = st.columns([3,2])

offering_data_df = pd.DataFrame({
        "Offering": ["Service Offerings", "Source"],
        "Value": ["", ""]
})

with col1:
    st.title("Lead routing engine")
    lead_form = st.text_area("describe your requirements here", height=300)
    if st.button("Search Offering"):
        if not lead_form:
            st.error('fill the lead form and then click search')
            st.stop()
        else:
            offering_data_df = indexes.searchServices(lead_form)

with col2:
    st.markdown("<br/>" * 5, unsafe_allow_html=True)  # Creates 5 lines of vertical space
    st.dataframe(
        offering_data_df,
        column_config={
            "Offering": st.column_config.Column(width=150),
            "Value": st.column_config.Column(width=150)
        },
        hide_index=True
    )