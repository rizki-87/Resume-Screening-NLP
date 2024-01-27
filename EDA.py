
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Fungsi untuk memuat data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Fungsi untuk menghitung jumlah resume per kategori pekerjaan
def plot_resumes_per_category(data):
    plt.figure(figsize=(15, 5))
    sns.countplot(data['Category'])
    plt.xticks(rotation=45)
    plt.title('Number of Resumes per Job Category')
    plt.ylabel('Number of Resumes')
    plt.xlabel('Job Category')
    plt.tight_layout()
    plot_path = "/mnt/data/resumes_per_category.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Fungsi untuk membuat word cloud dari resume untuk kategori 'Java Developer'
def create_wordcloud_for_category(data, category='Java Developer'):
    resumes_in_category = data[data['Category'] == category]['Resume']
    combined_text = ' '.join(resumes_in_category)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {category} Resumes')
    plt.tight_layout()
    wordcloud_path = f"/mnt/data/wordcloud_{category.replace(' ', '_')}.png"
    plt.savefig(wordcloud_path)
    plt.close()
    return wordcloud_path

# Fungsi untuk memplot distribusi panjang resume berdasarkan kategori
def plot_resume_lengths_by_category(data):
    data['Resume_Length'] = data['Resume'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(25, 8))
    sns.boxplot(x='Category', y='Resume_Length', data=data)
    plt.xticks(rotation=45)
    plt.title('Distribution of Resume Lengths by Category')
    plt.ylabel('Length of Resume (Number of Words)')
    plt.xlabel('Category')
    plt.tight_layout()
    lengths_plot_path = "/mnt/data/resume_lengths_by_category.png"
    plt.savefig(lengths_plot_path)
    plt.close()
    return lengths_plot_path
