import streamlit as st
import tempfile
import os
from nst import run_nst

st.title('Style your Content using NST')
st.warning("⚠️ This app runs on CPU ( if GPU isn't available on your device ) — generation takes 10-12 minutes. Best experienced locally with a GPU.")

style_option = st.radio("Style source", ["Use a preset", "Upload your own"])

uploaded_style = True
uploaded_content = True
content = ''
style = ''
style_path = ''
content_path = ''

if style_option == "Use a preset" :
    uploaded_style = False
    preset = st.selectbox("Choose a preset style", ["Starry Night", "Water Lillies", "The Scream", "Composition VII"])
    style_paths = {
        "Starry Night": "style/starry_night.jpg",
        "Water Lillies": "style/lillies.jpg",
        "The Scream": "style/scream.jpg",
        "Composition VII": "style/kandinsky.jpg"
        }
        
    style_path = style_paths[preset]
    
else:
    style_file = st.file_uploader('Choose a style image from the presets - or upload your own image:', type = ['webp', 'jpg', 'png'])
        
content_option = st.radio("Content source", ["Use a preset", "Upload your own"])

if content_option == "Use a preset" :
    uploaded_content = False
    preset = st.selectbox("Choose a preset stock image", ["Africa", "Mountains", "Tokyo"])
    content_paths = {
        "Africa": "content/africa.jpg",
        "Mountains": "content/mountains.jpg",
        "Tokyo": "content/tokyo.jpg",
        }
        
    content_path = content_paths[preset]
    
else:
    content_file = st.file_uploader('Choose a content image from the stock - or upload your own image:', type = ['webp', 'jpg', 'png'])
    
    
    
strength = st.slider('Style Strength', 1, 10, 5)
beta = 10 ** strength
alpha = 0.001

if uploaded_style and style_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
        f.write(style_file.read())
        style = f.name
else:
    style = style_path
    
if uploaded_content and content_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
        f.write(content_file.read())
        content = f.name
else:
    content = content_path
    
    

if st.button("Generate Image"):
    with st.spinner("Generating... this may take a few minutes"):
        img = run_nst(content, style, alpha, beta, steps=300)
    st.image(img)
