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
score_plot = data['Score'].value_counts().sort_index().plot(kind='bar',
                                                            title='Count of Reviews', figsize=(10, 5),
                                                            color='blue', edgecolor='black')
score_plot.set_xlabel('Review Stars')
plt.show()
