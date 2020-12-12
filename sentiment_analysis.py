"""
Sentiment analysis
Ref:
https://github.com/GoatWang/IIIMaterial/blob/master/08_InformationRetreival/main08.ipynb
"""


import config

import numpy as np
import pandas as pd

from datetime import datetime

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation

from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import csr_matrix, hstack

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers

def load_data(p):
    """
    讀取資料集
    """
    file_names = [f for f in listdir(p) if isfile(join(p, f))]

    # 初始化 df
    df = pd.DataFrame({'text': pd.Series([], dtype='object'), 'label': pd.Series([], dtype='int64')})

    # 將 txt 讀入並放入 df
    for n in file_names:
        tmp_df = pd.read_table( p + n ,header=None,sep='\t', names=["text", "label"])
        df = df.append(tmp_df,ignore_index=True)
    
    return train_test_split(df['text'], df['label'], test_size=config.TEST_SIZE, random_state=123)

def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

def main():
    """
    英文 NLP - Sentiment analysis
    """
    # 初始化分詞、停用字
    porter_stemmer = PorterStemmer()
    stops = stopwords.words('english')
    
    # lancaster_stemmer = LancasterStemmer()
    # snowball_stemmer = SnowballStemmer('english')
    # wordnet_lemmatizer = WordNetLemmatizer()

    X_train, X_test, y_train, y_test = load_data(config.INPUT_PATH)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


    vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\;|\:')

    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    X_train_chars = X_train.str.len()
    X_test_chars = X_test.str.len()

    X_train_punc = X_train.apply(lambda x: len([c for c in str(x) if c in punctuation]))
    X_test_punc = X_test.apply(lambda x: len([c for c in str(x) if c in punctuation]))

    X_train_dtm = add_feature(X_train_dtm, [X_train_chars, X_train_punc])
    X_test_dtm = add_feature(X_test_dtm, [X_test_chars, X_test_punc])

    X_train_dense = X_train_dtm.todense()
    print(X_train_dense.shape)


    model=Sequential()
    model.add(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.01)))  
    model.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1))

    model.compile(optimizer='adam',loss="mse")
    earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=0)
    checkpoint =ModelCheckpoint(str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+".hdf5",save_best_only=True)
    callback_list=[earlystopper,checkpoint]
    model.fit(X_train_dense, y_train.values, epochs=20, batch_size=config.BATCH_SIZE,validation_split=config.VALIDATION_SIZE,callbacks=callback_list)

    pred_train = model.predict(X_train_dense,batch_size=config.BATCH_SIZE)
    pred_test = model.predict(X_test_dtm,batch_size=config.BATCH_SIZE)

    pred_test_round = np.around(pred_test)
    pred_test_int = pred_test_round.astype(int)
    correct_prediction = np.equal(pred_test_int, y_test.values)
    print(np.mean(correct_prediction))


    # testStr = "This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts."
    # # 請使用nltk.word_tokenize及nltk.wordpunct_tokenize進行分詞，並比較其中差異。
    # #=============your works starts===============#
    # word_tokenize_tokens = nltk.word_tokenize(testStr)
    # wordpunct_tokenize_tokens = nltk.wordpunct_tokenize(testStr)
    # #==============your works ends================#

    # print("/".join(word_tokenize_tokens))
    # print("/".join(wordpunct_tokenize_tokens))


if __name__ == "__main__":
    main()