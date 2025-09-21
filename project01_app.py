# Project01_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords

# Seting page configuration
st.set_page_config(
    page_title="CORD-19 Research Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Downloading stopwords
try:
    nltk.download('stopwords')
except:
    pass

# App title and description
st.title("📊 CORD-19 Research Paper Analysis")
st.markdown("""
This interactive dashboard explores the COVID-19 Open Research Dataset (CORD-19), 
containing metadata for thousands of scientific papers related to COVID-19.
""")

# Sidebar for controls
st.sidebar.header("Controls")
st.sidebar.info("Adjust these parameters to customize the visualizations")

# Loading data (with caching to avoid reloading on every interaction)
@st.cache_data
def load_data():
    # Loading cleaned dataset here
        df = pd.read_csv('cleaned_metadata.csv')

df = load_data()

# Sampling of the data
if st.sidebar.checkbox("Show raw data sample"):
    st.subheader("Sample of the Data")
    num_rows = st.slider("Number of rows to show", 5, 50, 10)
    st.dataframe(df.head(num_rows))

# Interactive controls
top_n_journals = st.sidebar.slider("Number of top journals to show", 5, 25, 15)
top_n_words = st.sidebar.slider("Number of top words to show", 10, 50, 25)
min_year = int(df['publication_year'].min()) if 'publication_year' in df.columns else 2019
max_year = int(df['publication_year'].max()) if 'publication_year' in df.columns else 2023
selected_years = st.sidebar.slider(
    "Select year range",
    min_value=min_year,
    max_value=max_year,
    value=(2019, max_year)
)

# Filtering data based on selected years
if 'publication_year' in df.columns:
    filtered_df = df[(df['publication_year'] >= selected_years[0]) & 
                    (df['publication_year'] <= selected_years[1])]
else:
    filtered_df = df

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Publications Over Time")
    if 'publication_year' in filtered_df.columns:
        yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(yearly_counts.index, yearly_counts.values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Publication Year')
        ax.set_ylabel('Number of Publications')
        ax.set_title('Publications by Year')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Publication year data not available")

with col2:
    st.subheader("🏥 Top Publishing Journals")
    if 'journal' in filtered_df.columns:
        journal_counts = filtered_df['journal'].value_counts().head(top_n_journals)
        fig, ax = plt.subplots(figsize=(10, 6))
        journal_counts.sort_values(ascending=True).plot(kind='barh', ax=ax, color='lightcoral')
        ax.set_xlabel('Number of Publications')
        ax.set_title(f'Top {top_n_journals} Journals')
        st.pyplot(fig)
    else:
        st.warning("Journal data not available")

# Word cloud
st.subheader("☁️ Word Cloud of Paper Titles")
if 'title' in filtered_df.columns and len(filtered_df) > 0:
    # Generate word cloud
    text = ' '.join(filtered_df['title'].dropna().astype(str))
    
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'covid', '19', 'sars', 'cov', '2', 'coronavirus', 'study', 'using'}
    stop_words.update(custom_stop_words)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stop_words,
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Frequent Words in Titles')
    st.pyplot(fig)
else:
    st.warning("Title data not available")

# Top words list
st.subheader("📋 Top Words in Titles")
if 'title' in filtered_df.columns and len(filtered_df) > 0:
    all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
    translator = str.maketrans('', '', string.punctuation)
    all_titles_clean = all_titles.lower().translate(translator)
    words = all_titles_clean.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_freq = Counter(filtered_words)
    
    top_words = word_freq.most_common(top_n_words)
    word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    st.dataframe(word_df)
else:
    st.warning("Title data not available")

# Statistics section
st.subheader("📊 Dataset Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Papers", len(filtered_df))
    
with col2:
    if 'publication_year' in filtered_df.columns:
        st.metric("Time Span", f"{int(filtered_df['publication_year'].min())}-{int(filtered_df['publication_year'].max())}")
    
with col3:
    if 'journal' in filtered_df.columns:
        unique_journals = filtered_df['journal'].nunique()
        st.metric("Unique Journals", unique_journals)

        #Footer
# Footer
st.markdown("---")
st.markdown("""
**Note:** I have created this application using the CORD-19 dataset. 
Specifically for Research Purposes.
            George Silvar 
""")
