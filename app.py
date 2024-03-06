import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Text to Poem Generator')

text = st.text_area('Enter text',None)

if st.button('Generate Poem'):
    res = predict([text])
    st.text(res[0])
