#1: Downloading the data from the link provided
#pip install kaggle
#kaggle datasets download -d allen-institute-for-ai/CORD-19-research-challenge
#After downloading the data, I have extracted metadata.csv 

# 2: Loadin the Metadata.csv into DataFrame
import pandas as pd
df = pd.read_csv('metadata.csv')

# Displaying the basic information about the DataFrame to confirm it loaded correctly
print("DataFrame loaded successfully!")
print(f"Number of rows and columns: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# 3-4: Examining the First Few Rows and Data Structure

# Displaying the first few rows to see a sample of the data
print("First 6 rows of the DataFrame:")
print(df.head())
print("\n") 

# Displaying the last few rows to see the end of the dataset
print("Last 6 rows of the DataFrame:")
print(df.tail())
print("\n")

# Getting a concise summary of the DataFrame, including the number of non-null values and data types
print("DataFrame Info (structure, dtypes, non-null counts):")
print(df.info())
print("\n")

# Geting the dimensions of the DataFrame (rows, columns)
print(f"Shape of the DataFrame: {df.shape}")
print("\n")

# Displaying the column names
print("Column names:")
print(df.columns.tolist())
print("\n")

# Generating descriptive statistics for numerical columns
print("Descriptive statistics for numerical columns:")
print(df.describe())

# 5: Identifying Columns with Many Missing Values

# Calculating the number of missing values per column
missing_values = df.isnull().sum()

# Calculating the percentage of missing values per column
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Creating a new DataFrame to clearly display this information
missing_info = pd.DataFrame({
    'Column_Name': missing_values.index,
    'Missing_Values': missing_values.values,
    'Missing_Percentage': missing_percentage.values
})

# Sorting the DataFrame by the percentage of missing values in descending order
missing_info = missing_info.sort_values('Missing_Percentage', ascending=False)

# Displaying the outcome
print("Missing Values Information for Each Column:")
print(missing_info)
print("\n")

#6: Handling Missing Values

# I decideded to remove that are missing rows and columns and not critical
# Making a copy of the original DataFrame to preserve the raw data
df_clean = df.copy()
print(f"Original shape: {df_clean.shape}")

#DROPING ENTIRE COLUMNS that are mostly missing and not important
# Identify columns with more than a threshold (e.g., 95%) of values missing
cols_to_drop = missing_info[missing_info['Missing_Percentage'] > 95]['Column_Name'].tolist()
print(f"\nDropping columns with >95% missing values: {cols_to_drop}")
df_clean = df_clean.drop(columns=cols_to_drop)

print(f"Shape after dropping high-missing columns: {df_clean.shape}")

# DROPING ROWS missing critical information: ABSTRACT
# This is a common, crucial step for analyzing texts
print("\nDropping rows missing an abstract...")
initial_count = len(df_clean)
df_clean = df_clean.dropna(subset=['abstract'])
new_count = len(df_clean)
print(f"Dropped {initial_count - new_count} rows missing an abstract. New shape: {df_clean.shape}")

#7: Presenting Clean Dataset
#Creating a clean copy of the original DataFrame to preserve the raw data
df_clean = df.copy()
# Printing the initial state
print("=== DATA CLEANING PROCESS STARTED ===")
print(f"Original dataset shape: {df_clean.shape}\n")

#8: Converting Date Columns to Datetime Format

#Before conversion, I have to check the data type and sample of the publish_time column
print("Step 7: Converting 'publish_time' to datetime format.")
print(f"Current data type of 'publish_time': {df_clean['publish_time'].dtype}")
print("\nFirst 10 values before conversion:")
print(df_clean['publish_time'].head(10))
print("\n")

# Converting the column to datetime, forcing errors to become NaT (Not a Time)
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce', format='mixed')

# After conversion, I have to check the results
print(f"New data type of 'publish_time': {df_clean['publish_time'].dtype}")
print("\nFirst 10 values after conversion:")
print(df_clean['publish_time'].head(10))
print("\n")

# Checking how many values could not be converted (became NaT)
nat_count = df_clean['publish_time'].isna().sum()
total_count = len(df_clean)
print(f"Number of entries that could not be converted (NaT): {nat_count}")
print(f"Percentage of invalid/unsupported date formats: {(nat_count / total_count * 100):.2f}%")
print("\n")

#9: Extract Year from Publication Date for Time-based Analysis

#Extracting the year from the datetime column using .dt.year
print("Extracting year from 'publish_time'...")
df_clean['publication_year'] = df_clean['publish_time'].dt.year

# Displaying the results
print("First 5 rows with the new 'publication_year' column:")
print(df_clean[['title', 'publish_time', 'publication_year']].head())
print("\n")

# Checking for any entries where the year could not be extracted (will be NaN)
missing_year_count = df_clean['publication_year'].isna().sum()
total_count = len(df_clean)
print(f"Number of entries without a valid year: {missing_year_count}")
print(f"Percentage of dataset without a year: {(missing_year_count / total_count * 100):.2f}%")
print("\n")

# DATA ANALYSIS AND VISUALIZATION 

# 10: Counting Papers by Publication Year

# Counting the number of papers for each unique year
papers_per_year = df_clean['publication_year'].value_counts()

# The result is a Series sorted by count (descending). For a timeline, I often want it sorted by year.
# Let's create two views: one sorted by year and one sorted by count.

# Sorting by Year (Ascending) - best for seeing the trend over time
papers_per_year_sorted_by_year = papers_per_year.sort_index(ascending=True)

# Sorting by Count (Descending) - best for seeing the most prolific years
papers_per_year_sorted_by_count = papers_per_year.sort_values(ascending=False)

# Displaying the results
print("NUMBER OF PAPERS PUBLISHED PER YEAR")
print("===================================")
print(f"{'Year':<10} {'Count':<10}") # Formatting the header
print("-" * 20)
for year, count in papers_per_year_sorted_by_year.items():
    print(f"{int(year):<10} {count:<10}") # Converting year to int for cleaner printing
print("\n")


# 11: Identifying Top Journals Publishing COVID-19 Research

# Counting the number of papers for each journal
papers_per_journal = df_clean['journal'].value_counts()

# Getting the total number of papers for percentage calculation
total_papers = len(df_clean)

# Displaying the top N journals
top_n = 20
print(f"TOP {top_n} JOURNALS PUBLISHING COVID-19/CORONAVIRUS RESEARCH")
print("=============================================================")
print(f"{'Rank':<5} {'Journal':<60} {'Count':<10} {'Percentage':<10}")
print("-" * 85)

# Iterating through the top N journals and print them
for rank, (journal, count) in enumerate(papers_per_journal.head(top_n).items(), 1):
    percentage = (count / total_papers) * 100
    # Truncating very long journal names for cleaner display
    display_journal = (journal[:57] + '...') if len(journal) > 60 else journal
    print(f"{rank:<5} {display_journal:<60} {count:<10} {percentage:.2f}%")

print("\n")

# Calculating and printing the coverage of the top journals
top_journals_total_papers = papers_per_journal.head(top_n).sum()
top_journals_coverage = (top_journals_total_papers / total_papers) * 100
print(f"The top {top_n} journals account for {top_journals_total_papers} papers, which is {top_journals_coverage:.1f}% of the dataset.")
print("\n")

# 12: Finding most frequent words

import nltk
from nltk.corpus import stopwords
from collections import Counter
import string

# Downloading the stopwords list (only need to do this once)
nltk.download('stopwords')

# Geting the English stopwords
stop_words = set(stopwords.words('english'))

# Combining all titles into one big string, ensuring we handle NaN values (though we cleaned them)
all_titles = ' '.join(df_clean['title'].dropna().astype(str))

# Converting to lowercase and removing punctuation

# This translation table maps all punctuation to None
translator = str.maketrans('', '', string.punctuation)
all_titles_clean = all_titles.lower().translate(translator)

# Splitting the text into words (tokenize)
words = all_titles_clean.split()

# Removing stopwords
filtered_words = [word for word in words if word not in stop_words and len(word) > 2] # Also filter out very short words

# Counting the frequency of each word
word_freq = Counter(filtered_words)

# Finding the most common words
top_n_words = 30
most_common_words = word_freq.most_common(top_n_words)

# Displaying the results
print(f"MOST FREQUENT {top_n_words} WORDS IN PAPER TITLES")
print("==============================================")
print(f"{'Rank':<5} {'Word':<15} {'Frequency':<10}")
print("-" * 35)
for rank, (word, count) in enumerate(most_common_words, 1):
    print(f"{rank:<5} {word:<15} {count:<10}")

print("\n")

import matplotlib.pyplot as plt

# Extracting data for plotting
words_list, counts_list = zip(*most_common_words) # This "unzips" the list of tuples

# Creating a horizontal bar chart for better readability
plt.figure(figsize=(10, 8))
plt.barh(range(len(words_list)), counts_list, color='skyblue')
plt.yticks(range(len(words_list)), words_list)
plt.xlabel('Frequency')
plt.title(f'Top {top_n_words} Most Frequent Words in Research Paper Titles')
plt.gca().invert_yaxis() # Inverting y-axis to show the highest frequency at the top
plt.tight_layout()
plt.show()

#13: Plotting Number of Pubications overtim
 
import seaborn as sns

# Setting the style for the plot
sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))

# Counting publications per year
yearly_counts = df_clean['publication_year'].value_counts().sort_index()

# Creating the plot
# Using a bar plot to clearly show the count for each discrete year
bars = plt.bar(yearly_counts.index, yearly_counts.values, color='steelblue', edgecolor='black', alpha=0.7)

# Annotating the bars with the exact count (optional but helpful)
for bar in bars:
    height = bar.get_height()
    if height > 1000: # Only label bars taller than this to avoid clutter
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05*max(yearly_counts.values),
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9, rotation=0)

# Formating the plot
plt.xlabel('Publication Year', fontsize=14)
plt.ylabel('Number of Publications', fontsize=14)
plt.title('Number of COVID-19 and Coronavirus Research Publications Over Time', fontsize=16, fontweight='bold')
plt.xticks(yearly_counts.index, rotation=45) # Ensure every year gets a tick label

# Formating y-axis to use commas for thousands
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

# Adjusting layout to prevent cutting off labels
plt.tight_layout()

# Showing the plot
plt.show()

# Printing the data used for the plot for reference
print("Data used for the publication trend plot:")
print(yearly_counts)

#14: Creating a bar chart of top publishing journals

# Getting the top N journals
top_n = 15
top_journals = df_clean['journal'].value_counts().head(top_n)

# Reversing the order for the horizontal bar chart (highest bar at the top)
top_journals = top_journals.sort_values(ascending=True)

# Creating the plot
plt.figure(figsize=(12, 10))
bars = plt.barh(top_journals.index, top_journals.values, color='lightcoral', edgecolor='black', alpha=0.8)

# Adding value labels on the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + (max(top_journals.values) * 0.01),  # Positioning text just to the right of the bar
             bar.get_y() + bar.get_height()/2,
             f'{int(width):,}',
             ha='left', va='center',
             fontsize=10)

# Formating the chart
plt.xlabel('Number of Publications', fontsize=14)
plt.ylabel('Journal', fontsize=14)
plt.title(f'Top {top_n} Journals Publishing COVID-19 and Coronavirus Research', fontsize=16, fontweight='bold')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}')) # Adding commas to x-axis

# Adjusting layout and display
plt.tight_layout()
plt.show()

# Printing the exact data for reference
print(f"Exact counts for the top {top_n} journals:")
print(top_journals.sort_values(ascending=False))

#15: Generating a Word Cloud of Paper Titles

from wordcloud import WordCloud, STOPWORDS

# Combining  all titles into one string, handling any potential NaNs
text = ' '.join(df_clean['title'].dropna().astype(str))

# Defining stopwords - combining NLTK stopwords, my custom list, and the WordCloud's built-in ones
stop_words = set(stopwords.words('english'))
custom_stop_words = {'covid', '19', 'sars', 'cov', '2', 'coronavirus', 'study', 'using', 'based', 'results', 'method', 'potential', 'new', 'review', 'case', 'analysis', 'clinical', 'health'}
stop_words.update(custom_stop_words)
all_stopwords = STOPWORDS.union(stop_words)

# Generating the word cloud object
wordcloud = WordCloud(
    width=1200,
    height=800,
    background_color='white',
    stopwords=all_stopwords,
    colormap='viridis', # This can be changed: 'plasma', 'inferno', 'magma', 'cool', 'winter'
    max_words=150,      # Maximum number of words to display
    collocations=False  # No need of including bigrams (e.g., "new_york")
).generate(text)

# Creating the plot
plt.figure(figsize=(14, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.title('Word Cloud of COVID-19 and Coronavirus Research Paper Titles', fontsize=20, pad=20, fontweight='bold')
plt.tight_layout(pad=0)
plt.show()

#16: Ploting Distribution of Paper Counts by Source

# Checking for common column names related to source.
possible_source_columns = ['source', 'source_x', 'source_y', 'database', 'repository']
source_column = None

for col in possible_source_columns:
    if col in df_clean.columns:
        source_column = col
        break

if source_column:
    print(f"Using column '{source_column}' for source analysis.")
    # Counting the number of papers per source
    source_counts = df_clean[source_column].value_counts()


    # Due to the availability of many small sources, I have grouped them into "Other"
    threshold = 0.01 * len(df_clean)  # Threshold of 1% of total papers
    main_sources = source_counts[source_counts >= threshold]
    other_count = source_counts[source_counts < threshold].sum()
    
    # Creating a new series for the plot, including "Other"
    plot_data = main_sources.copy()
    if other_count > 0:
        plot_data['Other'] = other_count

    # Create a pie chart to show the proportion from each major source
    plt.figure(figsize=(12, 10))
    colors = plt.cm.Set3(range(len(plot_data)))  # Generate a color map
    
    wedges, texts, autotexts = plt.pie(plot_data.values, 
                                       labels=plot_data.index, 
                                       colors=colors,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       textprops={'fontsize': 12})
    # Improving the appearance of the autopct labels
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_weight('bold')
    
    plt.title('Distribution of Research Papers by Source/Database', fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')  # Ensuring the pie is drawn as a circle
    plt.tight_layout()
    plt.show()

    # Printing the data in a table for clarity
    print("\nNumber of papers per source:")
    for source, count in source_counts.head(10).items(): # Show top 10
        print(f"{source}: {count:,}")

else:
    print("No clear 'source' column found in the DataFrame.")
    print("Available columns are:")
    print(df_clean.columns.tolist())


    # This step completes our comprehensive exploratory data analysis of the CORD-19 metadata;
    #  covering temporal trends, journal outlets, title content, and data sources.








