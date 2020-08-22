
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

import os,re,nltk

import pandas as pd
import numpy as np
import gensim.models.keyedvectors as word2vec

import matplotlib.pyplot as plt
model=Word2Vec.load('./model/word2vec_Covid19.model')
# model = word2vec.KeyedVectors.load('word2vec_skipgram.model')
# for word in model.wv.most_similar(u"chính_trị",topn=50):

def flatten(lst):
    return [y for x in lst for y in x]
# pathfile = './words'
# with open(pathfile, 'r') as f:
#     words = f.readlines()
#     words = [word.strip() for word in words]


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=250, random_state=5)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

if __name__ == '__main__':
    print(model.wv.most_similar('giải'))
    tsne_plot(model)