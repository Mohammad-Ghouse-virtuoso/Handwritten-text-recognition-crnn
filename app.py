import streamlit as st
from Text_recognition_Hand_W2 import predict_text


st.title("Handwriting Recognition")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    result_text = predict_text("temp_image.png")
    st.write(f"Predicted Text: {result_text}")
