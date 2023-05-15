import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation

from keras.preprocessing import sequence

from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords

np.random.seed(1)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def cut_stopwords(input):
    stopwords_list = stopwords.words('english')
    whitelist = ["n't", "not", "no"]
    words = input.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)


data = pd.read_excel("SampleReviews.xlsx", usecols=["Id", "Score", "Text"])
print(data['Text'])