"""
Sentiment analysis
Ref:
https://github.com/GoatWang/IIIMaterial/blob/master/08_InformationRetreival/main08.ipynb
https://www.kaggle.com/anirudha16101/sentiment-analysis
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
from string import punctuation

from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import csr_matrix, hstack

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers

import matplotlib.pyplot as plt

def load_data(p):
    """
    讀取資料集
    """
    file_names = [f for f in listdir(p) if isfile(join(p, f))]

    # # 初始化 df
    # df = pd.DataFrame({'text': pd.Series([], dtype='object'), 'label': pd.Series([], dtype='int64')})

    # # 將 txt 讀入並放入 df
    # for n in file_names:
    #     tmp_df = pd.read_table( p + n ,header=None,sep='\t', names=["text", "label"])
    #     df = df.append(tmp_df,ignore_index=True)
    
    # 嘗試切割子資料集
    df_yelp = pd.read_table( p + file_names[0] ,header=None,sep='\t', names=["text", "label"])
    df_imdb = pd.read_table( p + file_names[1] ,header=None,sep='\t', names=["text", "label"])
    df_amazon = pd.read_table( p + file_names[2] ,header=None,sep='\t', names=["text", "label"])

    df = pd.concat([df_amazon,df_yelp,df_imdb])

    return df

def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

def plot_graphs(history, string, method):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    # plt.show()
    plt.savefig(method + '_' + string + '.png')

def method_1_ann(X_train, X_test, y_train, y_test):
    """
    嘗試使用ANN做情感預測
    """
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
    model.add(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.1)))  
    model.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.1)))
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


def method_2_LSTM(df):
    """
    嘗試做 word to vector 之後使用 LSTM
    """
    text = df['text'].tolist()
    label = df['label'].tolist()

    # word to vector
    vocab_size=1000
    print("vocab size is", vocab_size)
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(text,vocab_size,max_subword_length=5)
    
    for i,sent in enumerate(text):
        text[i] = tokenizer.encode(sent)

    max_length = 50

    text_added = pad_sequences(text,maxlen=max_length,padding ='post',truncating='post')

    # 切分訓練集與測試集
    training_size=int(len(text)*0.8)
    train_seq=text_added[:training_size]
    train_labels=label[:training_size]

    test_seq=text_added[training_size:]
    test_labels=label[training_size:]

    train_labels=np.array(train_labels)
    test_labels=np.array(test_labels)
    print("Total no of Training Sequence are",len(train_seq))
    print("Total no of Test Sequence are",len(test_seq))

    # Create model
    embedding_dim = 16

    model=tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(6,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    #fit a model
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
    model.summary()

    history = model.fit(train_seq,train_labels,epochs=20,validation_data=(test_seq,test_labels))

    plot_graphs(history, "accuracy", 'LSTM')
    plot_graphs(history, "loss", 'LSTM')

def main():
    """
    英文 NLP - Sentiment analysis
    """
    df = load_data(config.INPUT_PATH)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=config.TEST_SIZE, random_state=123)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # method_1_ann(X_train, X_test, y_train, y_test)

    method_2_LSTM(df.reset_index(drop=True))
    

if __name__ == "__main__":
    main()