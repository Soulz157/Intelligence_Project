import streamlit as st
import pathlib 

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("css/style.css")
load_css(css_path)


st.title('Intelligence Project')
st.markdown('''<div class="container">
    <div class="row">
        <div class="col-content">
            <p>ผมได้นำDataset มา จาก
            <a href="https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025"> Kaggle </a> 
             ซึ่งเป็น Dataset ของ Moblieในปี2024เพื่อมาจำแนกและแบ่งประเถทของMobileออกเป็น 3 ระดับ: Budget (ราคาประหยัด), Mid-range (ระดับกลาง), Premium (ระดับพรีเมียม) โดยใช้ข้อมูล เช่น RAM, แบตเตอรี่, ขนาดจอ, ฯลฯ</p>
            <a href="/Machine Learning" class="btn btn-primary">Go to Machine Learning</a>
        </div>
    </div>
            </div>''', unsafe_allow_html=True)