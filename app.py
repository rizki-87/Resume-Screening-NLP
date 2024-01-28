
import streamlit as st
import pandas as pd
import PyPDF2
# import dill
import pickle
import re

# Fungsi untuk membersihkan teks resume
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)  # Menghapus URL
    cleanText = re.sub('RT|cc', ' ', cleanText)  # Menghapus RT dan cc
    cleanText = re.sub('@\S+', ' ', cleanText)  # Menghapus mentions
    cleanText = re.sub('#\S+', ' ', cleanText)  # Menghapus hashtags
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~"""), ' ', cleanText)  # Menghapus punctuations
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)  # Menghapus karakter non-ASCII
    cleanText = re.sub('\s+', ' ', cleanText)  # Menghapus spasi ekstra

    return cleanText

# Fungsi untuk mengekstrak teks dari PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfFileReader(file)
    text = ''
    for page in range(reader.numPages):
        text += reader.getPage(page).extractText()
    return text

# # Load fungsi, model, dan vectorizer dari file
# with open('cleanResume_function.dill', 'rb') as f:
#     cleanResume_function = dill.load(f)

with open('tfidfd.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('clf.pkl', 'rb') as f:
    classification_model = pickle.load(f)

category_mapping = {
    0: "Advocate",
    1: "Arts",
    2: "Automation Testing",
    3: "Blockchain",
    4: "Business Analyst",
    5: "Civil Engineer",
    6: "Data Science",
    7: "Database",
    8: "DevOps Engineer",
    9: "DotNet Developer",
    10: "ETL Developer",
    11: "Electrical Engineering",
    12: "HR",
    13: "Hadoop",
    14: "Health and Fitness",
    15: "Java Developer",
    16: "Mechanical Engineer",
    17: "Network Security Engineer",
    18: "Operations Manager",
    19: "PMO",
    20: "Python Developer",
    21: "SAP Developer",
    22: "Sales",
    23: "Testing",
    24: "Web Designing",
    # Jika kategori prediksi tidak ada dalam kamus ini, akan dikembalikan "Unknown"
    'outside the data above': "Unknown"
}

st.title("Resume Screening with NLP")

uploaded_file = st.file_uploader("Upload your resume in PDF format:", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    cleaned_text = cleanResume(text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    prediction = classification_model.predict(vectorized_text)
    
    # Mendapatkan kategori berdasarkan prediksi
    predicted_category = category_mapping.get(prediction[0], "Unknown")
    
    st.write(f"Predicted Category: {predicted_category}")  # Menampilkan kategori bukan angka
    # Tambahkan lebih banyak feedback instan dan analisis di sini
