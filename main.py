import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pandasai import PandasAI

#Instantiate a LLM
from pandasai.llm.openai import OpenAI
llm = OpenAI(api_token="OPENAI_API_KEY")

# Packages for file opening
import tkinter as tk
from tkinter import filedialog, simpledialog

plt.style.use('ggplot')
import seaborn as sns
import nltk

# Packages for RoBERTa model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Create Tkinter root window
root = tk.Tk()
root.withdraw()

# Prompt the user to select an Excel file using a file dialog
file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls *.xlsx")])

# Close program if no file opened
if not file_path:
    print("No file selected. Ending program now.")
    sys.exit()

# Read selected excel file using pandas
data = pd.read_excel(file_path)
print(data['Score'])

# Create and show histogram showing counts of review stars
score_plot = data['Score'].value_counts().sort_index().plot(kind='bar',
                                                            title='Count of Reviews', figsize=(10, 5),
                                                            color='gray', edgecolor='black')
score_plot.set_xlabel('Review Stars')
plt.show()

# NLTK initialization
example = data['Text'][26]
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
chunked = nltk.chunk.ne_chunk(tagged)
print(example)


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()
results = {}

#RoBERTa (Robustly Optimized Bidirectional Encoder Representations from Transformers) Model
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

# show_prompt = input("Would you like to ask questions about the data? "
#                                                    "Type 'Yes' or 'No'")
# # Use PandasAI to show other visualizations
# while show_prompt == 'Yes':
#     user_input = input("What data would you like to see?")
#     pandas_ai = PandasAI(llm)
#     pandas_ai.run(roberta, prompt=user_input)
#     show_prompt = input("Would you like to ask more questions about the data? "
#                                                        "Type 'Yes' or 'No'")

# Choose file name and location
export_path = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel Files', '*.xlsx')])

print(roberta.head(5))

# End program if no export path chosen
if not export_path:
    print("No file saved. Program will exit.")
    sys.exit()

roberta.to_excel(export_path, index=False)
