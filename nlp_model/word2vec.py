import os
import pandas as pd
import string
from pyvi import ViTokenizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from deepai_nlp.tokenization.crf_tokenizer import CrfTokenizer
from deepai_nlp.tokenization.utils import preprocess_text

from readData import readExcelFile
from preProcessData import *

#List of categories
category_list = {
    '0': [],
    '1': [],
    '2': [],
    '3': [],
    '4': [],
    '5': [],
    '6': [],
    '7': [],
    '8': [],
}

my_sheet ='zingArticles'
file_name ='zingArticles.xlsx'
category_list_name = ['Politic','Sport','Education','Entertainment','Technology','Economy','Law','Film','Covid19']
tokenizer = CrfTokenizer()


if __name__ == "__main__":
    train_data = readExcelFile(file_name,my_sheet,category_list)
    data = []  
    stopWordLst = stopWordsLst()

    for i in list(category_list.keys()):
        data = []
        if len(train_data[i]) == 0:
            print(category_list_name[int(i)])
            continue
        for y in train_data[i]:
            documents = preprocess_text([y], tokenizer=tokenizer) # Tách từ và clean
            if len(documents) == 0:
                continue
            sents = removeStopWord(documents,stopWordLst)
            data.append(sents)

        
        model = Word2Vec(data, size=100, window=20, min_count=2, workers=4, sg=0)
        # if int(i) == 8 or int(i) == 7:
        #     i = '7'
        try:
            model.save("./model/word2vec_{}.model".format(category_list_name[int(i)]))
        except Exception as ex:
            print('Error:',ex)
            print('Position:',i)