import streamlit as st
import numpy as np
import pandas as pd
import pathlib 

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("css/style.css")
load_css(css_path)


df = pd.read_csv('data/MobilesDataset2025.csv' ,encoding='ISO-8859-1')

st.title('Classification of Mobile Phones')
st.markdown('''<div class="container">
            <div class="row">
                <div class="col-contentML">
                    <p>ผมได้นำDataset มา จาก
                    <a href="https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025"> Kaggle </a> 
                    ซึ่งเป็น Dataset ของ Moblieใน ปี2024 เพื่อมาจำแนกและแบ่งประเภทของ Mobile ออกเป็น 3 ระดับ: Budget (ราคาประหยัด), Mid-range (ระดับกลาง), Premium (ระดับพรีเมียม) โดยใช้ข้อมูล เช่น RAM, แบตเตอรี่, ขนาดจอ, ฯลฯ
                    โดย Algorithm ที่ผมเลือกที่จะนำมาใช้คือ K-Nearest Neighbors (KNN) และ Support Vector Machine (SVM)</p>
                </div>
            </div>
        </div>''' , unsafe_allow_html=True)

st.divider()
st.markdown("<h3 style='text-align: center; color: #00000;'>Raw Data</h3>", unsafe_allow_html=True)
st.write(df)

st.divider()
st.subheader('Company Name Filter From Raw Data')
st.markdown(
    f"""
    <div style="display: flex; flex-wrap: nowrap; overflow-x: auto;">
        {' '.join([f'<div style="padding: 10px; margin-right: 20px;">{name}</div>' for name in df['Company Name'].unique()])}
    </div>
    """, unsafe_allow_html=True
)
