import streamlit as st
import pandas as pd
import indexes_rca

col1, col2 = st.columns([3,2])

rca_data_df = pd.DataFrame({
        "Cause": ["Cause Analysis", "Source"],
        "Value": ["", ""]
})

with col1:
    st.title("RCA Tool")
    rca_form = st.text_area("describe your issue here", height=300)
    if st.button("Identity Root Cause and Suggest Preventive Measures"):
        if not rca_form:
            st.error('fill the  form and then click search')
            st.stop()
        else:
            rca_data_df = indexes_rca.searchServices(rca_form)

with col2:
    st.markdown("<br/>" * 5, unsafe_allow_html=True)  # Creates 5 lines of vertical space
    st.dataframe(
        rca_data_df,
        column_config={
            "Cause": st.column_config.Column(width=500),
            "Value": st.column_config.Column(width=500)
        },
        hide_index=True
    )