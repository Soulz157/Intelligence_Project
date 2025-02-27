import streamlit as st
import numpy as np
import pandas as pd


df = pd.read_csv('data/MobilesDataset2025.csv' ,encoding='ISO-8859-1')

st.title('Machine Learning')
st.write('Welcome to the Machine Learning page!')
st.write(df)