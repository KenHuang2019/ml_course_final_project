"""
Sentiment analysis
Ref:
https://github.com/GoatWang/IIIMaterial/blob/master/08_InformationRetreival/main08.ipynb
https://www.kaggle.com/anirudha16101/sentiment-analysis

Attention:
https://keras.io/examples/nlp/text_classification_with_transformer/
https://keras.io/api/optimizers/adam/
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

from tensorflow import keras
from tensorflow.keras import layers

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers

import matplotlib.pyplot as plt

# Attention
from multi_head_self_attention import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding

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
    # df_yelp = pd.read_table( p + file_names[0] ,header=None,sep='\t', names=["text", "label"])
    # df_imdb = pd.read_table( p + file_names[1] ,header=None,sep='\t', names=["text", "label"])
    df_amazon = pd.read_table( p + file_names[2] ,header=None,sep='\t', names=["text", "label"])

    # df = pd.concat([df_amazon,df_yelp,df_imdb])
    df = df_amazon

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
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=0)
    checkpoint =ModelCheckpoint(str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+".hdf5",save_best_only=True)
    callback_list=[earlystopper,checkpoint]
    model.fit(X_train_dense, y_train.values, epochs=40, batch_size=config.BATCH_SIZE,validation_split=config.VALIDATION_SIZE,callbacks=callback_list)

    pred_train = model.predict(X_train_dense,batch_size=config.BATCH_SIZE)
    pred_test = model.predict(X_test_dtm,batch_size=config.BATCH_SIZE)

    pred_test_round = np.around(pred_test)
    pred_test_int = pred_test_round.astype(int)
    correct_prediction = np.equal(pred_test_int, y_test.values)
    print(np.mean(correct_prediction))

def preprocessing(df, vocab_size, max_length):
    """
    docstring
    """
    text = df['text'].tolist()
    label = df['label'].tolist()

    # word to vector
    print("vocab size is", vocab_size)
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(text,vocab_size,max_subword_length=5)
    
    for i,sent in enumerate(text):
        text[i] = tokenizer.encode(sent)

    text_added = pad_sequences(text,maxlen=max_length,padding ='post',truncating='post')

    # 切分訓練集與測試集
    training_size=int(len(text)*0.8)

    train_seq=text_added[:training_size]
    test_seq=text_added[training_size:]

    train_labels=np.array(label[:training_size])
    test_labels=np.array(label[training_size:])
    print("Total no of Training Sequence are",len(train_seq))
    print("Total no of Test Sequence are",len(test_seq))

    return train_seq, train_labels, test_seq, test_labels

def method_2_LSTM(df):
    """
    嘗試做 word to vector 之後使用 LSTM
    """
    vocab_size = 1000
    max_length = 50
    train_seq, train_labels, test_seq, test_labels = preprocessing(df, vocab_size, max_length)

    # Create model
    embedding_dim = 16

    model=tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(6,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss',mode='auto',save_best_only=True),
    ]

    #fit a model
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
    model.summary()

    history = model.fit(train_seq,train_labels,epochs=40,validation_data=(test_seq,test_labels),callbacks=my_callbacks)

    plot_graphs(history, "accuracy", 'LSTM')
    plot_graphs(history, "loss", 'LSTM')

def method_3_Attention(df):
    """
    Attention
    """
    vocab_size = 1000
    max_length = 50
    x_train, y_train, x_val, y_val = preprocessing(df, vocab_size, max_length)

    embed_dim = 40  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 40  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(max_length,))
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=2,
    #     decay_rate=0.9)
    # optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    

    # opt = keras.optimizers.Adam(learning_rate=0.001)
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True,
        name="Adam"
    )

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    # my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=6),
    # tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss',mode='auto',save_best_only=True),
    # ]
    
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val)#,callbacks=my_callbacks
    )

    plot_graphs(history, "accuracy", 'Attention')
    plot_graphs(history, "loss", 'Attention')


def main():
    """
    英文 NLP - Sentiment analysis
    """
    df = load_data(config.INPUT_PATH)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=config.TEST_SIZE, random_state=123)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # method_1_ann(X_train, X_test, y_train, y_test)

    # method_2_LSTM(df.reset_index(drop=True))

    method_3_Attention(df.reset_index(drop=True))
    

if __name__ == "__main__":
    main()