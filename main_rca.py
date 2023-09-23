import streamlit as st
import pandas as pd
import indexes_rca
st.set_page_config(layout="wide")

col1, col2 = st.columns([2,1])

rca_data_df = pd.DataFrame({
        "Cause Analyis and Source": ["", ""]
})

with col2:
    st.title("RCA Tool")
    rca_form = st.text_area("describe your issue here", height=300)
    if st.button("Identity Root Cause and Suggest Preventive Measures"):
        if not rca_form:
            st.error('fill the  form and then click search')
            st.stop()
        else:
            rca_data_df = indexes_rca.searchServices(rca_form)

with col1:
    st.markdown("<br/>" * 5, unsafe_allow_html=True)  # Creates 5 lines of vertical space
    st.dataframe(
        rca_data_df,
        use_container_width= True,
        hide_index=True
    )
