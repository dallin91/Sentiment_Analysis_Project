import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import seaborn as sns
import nltk

# Read data using pandas
data = pd.read_excel("SampleReviews.xlsx")
print(data['Score'])

# Create and show histogram showing counts of review stars
# score_plot = data['Score'].value_counts().sort_index().plot(kind='bar',
#                                                             title='Count of Reviews', figsize=(10, 5),
#                                                             color='gray', edgecolor='black')
# score_plot.set_xlabel('Review Stars')
# plt.show()

# NLTK practice DO NOT USE IN FINAL PRODUCT
example = data['Text'][26]
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
chunked = nltk.chunk.ne_chunk(tagged)
print(example)

# VADER (Valence Aware Dictionary and sEntiment Reasoner) Sentiment Scoring
# DO NOT USE IN FINAL PRODUCT
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()
results = {}
for i, row in tqdm(data.iterrows(), total=len(data)):
    text = row['Text']
    myid = row['Id']
    # Converts non string variables to strings
    if type(text) is not str:
        text = str(text)
    polarity = sia.polarity_scores(text)
    results[myid] = polarity

# Take results and put into pandas dataframe
vaders = pd.DataFrame(results).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(data, how='left')

# Check compound score from vaders against star score review
tester_plot = sns.barplot(data=vaders, x='Score', y='compound')
tester_plot.set_title('Compound Score by Amazon Star Review')
plt.show()
