import streamlit as st

st.set_page_config(page_title="FellX4 — XRD Toolkit", layout="wide")

home = st.Page("pages/home.py", title="Startseite", icon="🏠", default=True)
analysis = st.Page("pages/1_Analyse.py", title="Analyse", icon="🔬")

pg = st.navigation([home, analysis])
pg.run()
