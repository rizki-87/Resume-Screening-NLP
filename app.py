
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import dill
import re
import PyPDF2
import io
from EDA import load_data, plot_resumes_per_category, create_wordcloud_for_category, plot_resume_lengths_by_category

# Load the necessary files for the model
# (These paths need to be updated with the correct paths where the files are stored)
clf_path = 'clf.pkl'
tfidf_vectorizer_path = 'tfidfd.pkl'
# cleaning_function_path = 'cleanResume_function.dill'

with open('clf.pkl', 'rb') as file:
    clf = pickle.load(file)

with open('tfidfd.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# with open(r'D:\PORTOFOLIO\Resume-Screening-NLP\cleanResume_function.dill', 'rb') as file:
#     clean_resume = dill.load(file)
    
def cleanResume(txt):
    cleanText = re.sub('http\S+\s',' ',txt)
    cleanText = re.sub('RT|cc',' ',cleanText)
    cleanText = re.sub('@\S+',' ',cleanText)
    cleanText = re.sub('#\S+',' ',cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~"""),' ',cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]',' ',cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)

    return cleanText

def predict_category(cv_text):
    # Preprocess and predict the category of the resume
    cleaned_text = cleanResume(cv_text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    category = clf.predict(vectorized_text)
    return category[0]

def extract_text_from_pdf(pdf_file):
    # Membuat objek PDF reader
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    
    # Menginisialisasi teks yang akan dihasilkan
    text = ""
    
    # Loop melalui setiap halaman dan ekstrak teks
    for page in range(pdf_reader.numPages):
        # Mendapatkan halaman
        pdf_page = pdf_reader.getPage(page)
        
        # Ekstrak teks dari halaman dan tambahkan ke variabel text
        text += pdf_page.extractText()
    
    return text


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

# Streamlit app layout
st.title('Resume Screening with NLP')

data = load_data('UpdatedResumeDataSet.csv')

# Navigation
page = st.sidebar.selectbox('Choose a page', ['Home', 'Data Storytelling & Visualization', 'Model'])

if page == 'Home':
    st.header('Welcome to the Resume Screening Application')
    st.write('This application uses NLP to screen resumes.')

elif page == 'Data Storytelling & Visualization':
    st.header('Data Storytelling & Visualization')
    st.write('Here you can see visualizations of the data.')

    if st.button('Show Number of Resumes per Job Category'):
        plot_path = plot_resumes_per_category(data)
        st.image(plot_path, caption='Number of Resumes per Job Category')

# elif page == 'Model':
#     st.header('Try the Model')
#     st.write('Upload a CV in PDF format to see its predicted category.')
#     uploaded_file = st.file_uploader('Upload your CV', type='pdf')
#     if uploaded_file is not None:
#         # Process the file here and predict
#         st.write('Processing...')
#         # TODO: Add code to read PDF and extract text
#         cv_text = 'extracted text from the PDF'  # Placeholder
#         predicted_category = predict_category(cv_text)
#         st.write('Predicted Category:', predicted_category)
#         if cv_text:  # Ganti 'if cv_text:' dengan kondisi yang sesuai setelah ekstraksi teks
#             predicted_category_num = predict_category(cv_text)
#             predicted_category_name = category_mapping.get(predicted_category_num, "Unknown")
#             st.write('Predicted Category:', predicted_category_name)
        
elif page == 'Model':
    st.header('Try the Model')
    st.write('Upload a CV in PDF format to see its predicted category.')
    uploaded_file = st.file_uploader('Upload your CV', type='pdf')
    if uploaded_file is not None:
        # Read the PDF file
        pdf_file = io.BytesIO(uploaded_file.read())
        cv_text = extract_text_from_pdf(pdf_file)  # You need to implement this function
        
        if cv_text:
            st.write('Processing...')
            predicted_category_num = predict_category(cv_text)
            predicted_category_name = category_mapping.get(predicted_category_num, "Unknown")
            st.write('Predicted Category:', predicted_category_name)
