import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Text to poem generation')

text = st.text_area('Enter text',None)

if st.button('Generate poem'):
    res = predict([text])
    st.text(res[0])
