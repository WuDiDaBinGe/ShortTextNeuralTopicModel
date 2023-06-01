import scipy.sparse
import numpy as np
import torch

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words

class DataProcessor(object):
    def __init__(self, english):
        self.english = english
        

    def process(self, texts):
        vector = CountVectorizer()
        bow_matrix = vector.fit_transform(texts).toarray()
        vocab = vector.get_feature_names()
        return bow_matrix, vocab
