
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

st.sidebar.title('Navigation')
options = ["Home", "Data Storytelling dan Visualization", "Model"]
selection = st.sidebar.radio("Go to", options)


# Konten untuk halaman Home
if selection == "Home":
    st.markdown("""
        <h2 style="font-weight:bold;">Simplify Recruit: An NLP Model for Resume Screening</h2>
        <h3 style="font-weight:bold;">by Rizki Pria Aditama</h3>
    """, unsafe_allow_html=True)
    st.markdown("Welcome to SimplifyRecruit, the cutting-edge platform designed to streamline and enhance the resume screening process. With our advanced NLP model, we've created a powerful tool to help you save time and effort in candidate selection. SimplifyRecruit leverages natural language understanding to swiftly analyze and filter through hundreds or even thousands of resumes within seconds, allowing you to focus on the most promising candidates. Our platform offers automated sorting, predictive ranking, and insightful data visualization, empowering you to make informed recruitment decisions efficiently. Seamlessly integrate SimplifyRecruit with your existing Applicant Tracking System (ATS) and elevate your recruitment process to new heights. Explore our Model Page in the navigation menu to perform instant screening or access visualizations and storytelling to gain deeper insights into your recruitment data. Let's embark on this journey together and discover the future of recruitment efficiency with SimplifyRecruit!")

# Konten untuk halaman Data Storytelling dan Visualization
elif selection == "Data Storytelling dan Visualization":
    st.title("Data Storytelling dan Visualization")
    st.write("Halaman ini akan menampilkan visualisasi data.")
    # Tempat untuk visualisasi data dan storytelling Anda

# Konten untuk halaman Model (yang sudah kita selesaikan)
elif selection == "Model":
    st.title("Resume Screening with NLP")
    uploaded_file = st.file_uploader("Upload your resume in PDF format:", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        cleaned_text = cleanResume(text)
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        prediction = classification_model.predict(vectorized_text)
        
        # Mendapatkan kategori berdasarkan prediksi
        predicted_category = category_mapping.get(prediction[0], "Unknown")
        st.write(f"Predicted Category: {predicted_category}")
        # Tambahkan lebih banyak feedback instan dan analisis di sini
