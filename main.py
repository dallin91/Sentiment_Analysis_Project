import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Packages for file opening
import tkinter as tk
from tkinter import filedialog

plt.style.use('ggplot')
import seaborn as sns
import nltk

# Packages for RoBERTa model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Tells PyTorch to use first available GPU
# device = torch.device(“cuda: 0”)

# Create Tkinter root window
root = tk.Tk()
root.withdraw()

# Prompt the user to select an Excel file using a file dialog
file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls *.xlsx")])

# Read selected excel file using pandas
data = pd.read_excel(file_path)
print(data['Score'])

# # Read data using pandas
# data = pd.read_excel("SampleReviews.xlsx")
# print(data['Score'])

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
# for i, row in tqdm(data.iterrows(), total=len(data)):
#     text = row['Text']
#     myid = row['Id']
#     # Converts non string variables to strings
#     if type(text) is not str:
#         text = str(text)
#     polarity = sia.polarity_scores(text)
#     results[myid] = polarity
#
# # Take results and put into pandas dataframe
# vaders = pd.DataFrame(results).T
# vaders = vaders.reset_index().rename(columns={'index': 'Id'})
# vaders = vaders.merge(data, how='left')

# # Check compound score from vaders against star score review
# tester_plot = sns.barplot(data=vaders, x='Score', y='compound')
# tester_plot.set_title('Compound Score by Amazon Star Review')
# plt.show()

#RoBERTa Model practice DO NOT USE IN FINAL PRODUCT
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

encoded_text = tokenizer(example, return_tensors='pt', max_length=512, truncation=True)
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    "roberta_neg" : scores[0],
    "roberta_neu" : scores[1],
    "roberta_pos" : scores[2]
}

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt', max_length=512, truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        "roberta_neg": scores[0],
        "roberta_neu": scores[1],
        "roberta_pos": scores[2]
    }
    return scores_dict


def get_sentiment(row):
    if row['roberta_neg'] > row['roberta_pos'] and row['roberta_neg'] > row['roberta_neu']:
        return 'Negative'
    if row['roberta_pos'] > row['roberta_neg'] and row['roberta_pos'] > row['roberta_neu']:
        return 'Positive'
    if row['roberta_neu'] > row['roberta_pos'] and row['roberta_neu'] > row['roberta_neg']:
        return 'Neutral'
    else:
        return 'Uh oh'


for i, row in tqdm(data.iterrows(), total=len(data)):
    try:
        text = row['Text']
        myid = row['Id']
        # Converts non string variables to strings
        if type(text) is not str:
            text = str(text)
        roberta_results = polarity_scores_roberta(text)

        results[myid] = roberta_results
    except RuntimeError:
        print(f'Broke for id {myid}')

roberta = pd.DataFrame(results).T
roberta = roberta.reset_index().rename(columns={'index': 'Id'})
roberta = roberta.merge(data, how='left')
roberta['sentiment'] = roberta.apply(get_sentiment, axis=1)

# Check compound score from roberta model against star score review
tester_plot = sns.barplot(data=roberta, x='Score', y='sentiment')
tester_plot.set_title('Sentiment Classification by Amazon Star Review')
plt.show()

# Choose file name and location
export_path = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel Files', '*.xlsx')])

print(roberta.head(5))
roberta.to_excel(export_path, index=False)
