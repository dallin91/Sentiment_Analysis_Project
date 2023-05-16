import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Read data using pandas
data = pd.read_excel("SampleReviews.xlsx")
print(data['Text'])