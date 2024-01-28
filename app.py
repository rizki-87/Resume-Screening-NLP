
import streamlit as st
import pandas as pd
import PyPDF2
import dill
import pickle

# Fungsi untuk mengekstrak teks dari PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfFileReader(file)
    text = ''
    for page in range(reader.numPages):
        text += reader.getPage(page).extractText()
    return text

# Load fungsi, model, dan vectorizer dari file
with open('cleanResume_function.dill', 'rb') as f:
    cleanResume_function = dill.load(f)

with open('tfidfd.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('clf.pkl', 'rb') as f:
    classification_model = pickle.load(f)

st.title("Resume Screening with NLP")

uploaded_file = st.file_uploader("Upload your resume in PDF format:", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    cleaned_text = cleanResume_function(text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    prediction = classification_model.predict(vectorized_text)
    
    st.write(f"Predicted Category: {prediction[0]}")
    # Tambahkan lebih banyak feedback instan dan analisis di sini
