# Resume-Screening-NLP
## Introduction
The "Resume-Screening-NLP" project leverages advanced Natural Language Processing (NLP) techniques to automate the resume screening process, addressing the challenge HR departments face when filtering through numerous job applications. By analyzing resumes against job descriptions, this tool streamlines the initial stages of recruitment, ensuring a more efficient and effective candidate selection process.

## Deployment
This application is deployed using Streamlit Share, allowing users to access it directly through a web interface without the need for local installation. Experience the live application here: Resume Screening Application.

## Features
- Automated Resume Screening: Quickly compare multiple resumes against a job description to identify the most suitable candidates.
- NLP-Powered Analysis: Utilize state-of-the-art NLP techniques for deep textual analysis and matching.
- User-Friendly Interface: Easy-to-navigate web interface for uploading documents and viewing screening results.

## Getting Started
### Prerequisites
Internet connection to access the deployed web application.
For local development or testing: Python 3.6+, Git.
### Accessing the Deployed Application
1. Go to Resume Screening Application.
2. Upload Job Description: Drag and drop the job description file or use the file selector.
3. Upload Resumes: Upload the resumes you wish to screen against the job description.
4. Screening Results: View the ranked list of resumes based on their relevance to the job description.
### Running Locally (Optional)
To run the application locally for development or testing purposes:

![Screenshot 2024-01-29 064248](https://github.com/rizki-87/Resume-Screening-NLP/assets/140106584/a87e4e9a-a796-4f86-92ec-50583226c23e)

## Files Description
- Resume-Screening-NLP.ipynb: Jupyter notebook with detailed analysis, model training, and evaluation steps.
- Model_Inference-Resume-Screening-NLP.ipynb: Notebook showcasing how to perform inference with the trained model.
- app.py: Streamlit web application script for the user interface.
- clf.pkl: Serialized model file used for predictions.
- cleanResume_function.dill: Serialized preprocessing function for cleaning resume data.
- tfidfd.pkl: Serialized TF-IDF vectorizer for text feature extraction.
- UpdatedResumeDataSet.csv: Dataset used for training the model, comprising various resume samples.
- requirements.txt: List of Python packages required to run the project.
- Contributing
- We welcome contributions to enhance the "Resume-Screening-NLP" project. Please review the CONTRIBUTING.md file for guidelines on contributing to this repository.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
Thanks to Streamlit for providing the framework to deploy this interactive web application.
Appreciation for the open-source community for the tools and libraries that made this project possible.

