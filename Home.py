import streamlit as st
import pathlib 

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("css/style.css")
load_css(css_path)


st.markdown("""
            <div class="container" style="text-align: center;">
                <h1>Intelligence Project</h1>
            </div>
""", unsafe_allow_html=True)
st.markdown("""
            <div class="container" style="text-align: center; margin-top: 20px;">
                <h4>Select a page to continue</h4>
            </div>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.card-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 20px;
}
.card {
    width: 350px;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    text-align: center;
    transition: transform 0.3s ease-in-out;
}
.card:hover {
    transform: scale(1.05);
}
.card img {
    width: 100%;
    border-radius: 8px;
}
.card a {
    display: block;
    text-decoration: none;
    font-weight: bold;
    color: #FFFFFF;
    background: #007BFF;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
.card a:hover {
    background: #0056b3;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-container">
    <div class="card">
        <h3>🧠 Machine Learning</h3>
        <a href="https://intelligence-project-phoorich.streamlit.app/machine_learning target="_blank">Continue</a>
    </div>
    <div class="card">
        <h3>🪴 Neural Network</h3>
        <a href="https://intelligence-project-phoorich.streamlit.app/neural_network" target="_blank">Continue</a>
    </div>
</div>
""", unsafe_allow_html=True)