import streamlit as st
import pandas as pd
import PyPDF2
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk

df = pd.read_csv('UpdatedResumeDataSet.csv')

nltk.download('punkt')
nltk.download('stopwords')

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)  # Menghapus URL
    cleanText = re.sub('RT|cc', ' ', cleanText)  # Menghapus RT dan cc
    cleanText = re.sub('@\S+', ' ', cleanText)  # Menghapus mentions
    cleanText = re.sub('#\S+', ' ', cleanText)  # Menghapus hashtags
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~"""), ' ', cleanText)  # Menghapus punctuations
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)  # Menghapus karakter non-ASCII
    cleanText = re.sub('\s+', ' ', cleanText)  # Menghapus spasi ekstra

    return cleanText

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

st.sidebar.title('Navigation')
options = ["Home", "Data Storytelling dan Visualization", "Model"]
selection = st.sidebar.radio("Go to", options)


# Content for Home page
if selection == "Home":
    st.markdown("""
        <h2 style="font-weight:bold;">Simplify Recruit: An NLP Model for Resume Screening</h2>
        <h3 style="font-weight:bold;">by Rizki Pria Aditama</h3>
    """, unsafe_allow_html=True)
    st.markdown("""
    Welcome to SimplifyRecruit, the cutting-edge platform designed to streamline and enhance the resume screening process. 
    With our advanced NLP model, we've created a powerful tool to help you save time and effort  in  candidate  selection. 
    SimplifyRecruit leverages natural language understanding to swiftly  analyze  and  filter  through  hundreds  or  even 
    thousands of resumes within seconds, allowing you to focus on the  most  promising  candidates.  Our  platform  offers 
    automated sorting, predictive ranking, and insightful data visualization, empowering you to make informed  recruitment 
    decisions efficiently. Seamlessly integrate SimplifyRecruit with your existing Applicant  Tracking  System  (ATS)  and 
    elevate your recruitment process to new heights. Explore our Model Page in the  navigation  menu  to  perform  instant 
    screening or access visualizations and storytelling to gain deeper insights into your recruitment data.  Let's  embark 
    on this journey together and discover the future of recruitment efficiency with SimplifyRecruit!
    """)

# Content for Data Storytelling and Visualization pages
elif selection == "Data Storytelling dan Visualization":
    st.title("Data Storytelling dan Visualization")
    st.write("This page will showcase visualizations and storytelling based on a dataset obtained from an IT company in London, which serves as the foundation for the development of this NLP model..")
    # Tempat untuk visualisasi data dan storytelling Anda

    # Visualization 1: Number of Resumes per Job Category
    plt.figure(figsize=(15,5))
    sns.countplot(y='Category', data=df, order = df['Category'].value_counts().index)
    st.pyplot(plt)
    st.markdown("""
    #### Insights on Number of Resumes per Job Category
    - **Dominant Categories:** The "Java Developer" category has the highest number of resumes, indicating a strong market presence.
    - **Moderate Representation:** "Web Designing", "HR", and "Hadoop" show balanced demand and supply.
    - **Development Areas:** Fewer resumes for "Advocate" and "Civil Engineer" suggest potential focus areas for HR.
    - **Recruitment Balance:** Equal numbers in certain categories suggest a talent availability balance.

    **HR Implications:**
    - Assess and adapt recruitment strategies to match organizational needs.
    - Identify and address skill gaps through recruitment or training.
    - Utilize insights to balance recruitment and workforce development efforts.
    """)

    # Visualization 2: Word Cloud for a Specific Category (e.g., Java Developer)
    selected_category = 'Java Developer'
    resumes_in_category = df[df['Category'] == selected_category]['Resume']
    combined_text = ' '.join(resumes_in_category)
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(combined_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {selected_category} Resumes')
    st.pyplot(plt)
    st.markdown("""
    #### Insights from Word Cloud for 'Java Developer' Position
    - **Keyword Frequency:** Common terms reflect the role's associated skills and experiences.
    - **Technical Proficiencies:** Frequent mentions of specific technologies highlight key candidate qualifications.
    - **Experience Emphasis:** The focus on experience duration can inform critical hiring factors.
    - **Educational Background:** Academic qualifications play a significant role in candidate profiles.

    **HR Utility:**
    - Aid in efficient resume screening and job description tailoring.
    - Benchmark candidates against common standards.
    - Develop training programs to address skill gaps.
    - Make strategic hiring decisions based on skill representation in the market.
    """)


    # Visualization 3: Resume Lengths by Category
    df['Resume_Length'] = df['Resume'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(25, 8))
    sns.boxplot(x='Category', y='Resume_Length', data=df)
    plt.xticks(rotation=45)
    plt.title('Distribution of Resume Lengths by Category')
    plt.ylabel('Length of Resume (Number of Words)')
    plt.xlabel('Category')
    st.pyplot(plt)
    st.markdown("""
    #### Insights from Resume Lengths Distribution
    - **Content Variability:** Different categories show varying resume lengths, indicating diverse presentation styles.
    - **Category Norms:** Certain fields like 'HR' and 'Advocate' tend to have shorter resumes, suggesting a preference for conciseness.
    - **Over-Elaboration Indicators:** Longer resumes in some categories may suggest unnecessary elaboration.

    **HR Department Applications:**
    - Standardize screening processes and provide application guidance based on industry norms.
    - Focus on key information for efficiency in review processes.
    - Use outliers as indicators for additional scrutiny to maintain quality control.
    """)

# Content for the Model page
elif selection == "Model":
    st.title("Resume Screening with NLP")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

    
