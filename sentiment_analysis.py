"""
Sentiment analysis
Ref:
https://github.com/GoatWang/IIIMaterial/blob/master/08_InformationRetreival/main08.ipynb
https://www.kaggle.com/anirudha16101/sentiment-analysis

Attention:
https://keras.io/examples/nlp/text_classification_with_transformer/
https://keras.io/api/optimizers/adam/

在資料分析上嘗試更多種方法，再藉由資料分析結果重新調整參數設定

嘗試將資料集切分成更小的單位，並在不同資料量上觀察不同演算法的成效

在 Transformer 訓練上設定 EarlyStopping

嘗試 dimentionality reduction 看看分佈狀態
"""
import re
import config
import argparse
from os import listdir
from os.path import isfile, join
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing, tree, svm # Decision_Tree and Support_Vector_Machine

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical

import joblib

# Attention
from multi_head_self_attention import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding

# data analysis
from data_analysis_func import *

#### 以下為 preprocessing 函式 ################################################################################
eng_stopwords = nltk.corpus.stopwords.words("english")

def load_data(p, sub_set_num=None):
    """
    讀取資料集
    """
    file_names = [f for f in listdir(p) if isfile(join(p, f))]

    df_yelp = pd.read_table( p + file_names[0] ,header=None,sep='\t', names=["text", "label"], quoting=3)
    df_imdb = pd.read_table( p + file_names[1] ,header=None,sep='\t', names=["text", "label"], quoting=3)
    df_amazon = pd.read_table( p + file_names[2] ,header=None,sep='\t', names=["text", "label"], quoting=3)

    df = pd.DataFrame(columns = ['text' , 'label'])
    index = 0

    for n in range(1000):
        df.loc[index] = df_amazon.iloc[n]
        df.loc[index+1] = df_yelp.iloc[n]
        df.loc[index+2] = df_imdb.iloc[n]
        index += 3
    
    if sub_set_num == None:
        return df
    else:
        return df[:sub_set_num] # 切割子資料集

def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

def tokenize_split(df, target_column, vocab_size, max_length):
    t = df[target_column].values

    label = df['label'].values

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(t)
    vocab_size = len(tokenizer.word_index) + 1
    encoded_docs = tokenizer.texts_to_sequences(t)
    padded_sequence = pad_sequences(encoded_docs, maxlen=max_length)
    # 切分訓練集與測試集
    training_val_size=int(padded_sequence.shape[0]*0.8)

    train_val_seq = padded_sequence[:training_val_size]
    test_seq = padded_sequence[training_val_size:]

    train_val_labels = label[:training_val_size]
    test_labels = label[training_val_size:]

    print("Total no of Training Sequence are",len(train_val_seq))
    print("Total no of Test Sequence are",len(test_seq))

    train_size=int(len(train_val_seq)*0.8)

    train_seq = train_val_seq[:train_size]
    val_seq = train_val_seq[train_size:]

    train_labels = train_val_labels[:train_size]
    val_labels = train_val_labels[train_size:]

    return train_seq, train_labels, val_seq, val_labels, test_seq, test_labels

def preprocessing_split(df, target_column, vocab_size, max_length):
    """
    切分訓練集、測試集
    """
    text = df[target_column].tolist()
    label = df['label'].tolist()

    # word to vector
    print("vocab size : ", vocab_size)
    print("max length : ", max_length)
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(text,vocab_size,max_subword_length=10)
    
    for i,sent in enumerate(text):
        text[i] = tokenizer.encode(sent)

    text_added = pad_sequences(text,maxlen=max_length,padding ='post',truncating='post')

    # 切分訓練集與測試集
    training_val_size=int(len(text)*0.8)

    train_val_seq=text_added[:training_val_size]
    test_seq=text_added[training_val_size:]

    train_val_labels=np.array(label[:training_val_size])
    test_labels=np.array(label[training_val_size:])
    print("Total no of Training Sequence are",len(train_val_seq))
    print("Total no of Test Sequence are",len(test_seq))

    train_size=int(len(train_val_seq)*0.8)

    train_seq = train_val_seq[:train_size]
    val_seq = train_val_seq[train_size:]

    train_labels = train_val_labels[:train_size]
    val_labels = train_val_labels[train_size:]

    return train_seq, train_labels, val_seq, val_labels, test_seq, test_labels

def tokenize(Doc):
    stops = set(stopwords.words('english'))
    puns = string.punctuation
    if pd.notnull(Doc):
        # 使用nltk.wordpunct_tokenize將Doc切開, 去掉停用字與標點符號，並轉小寫
        tokens = nltk.wordpunct_tokenize(Doc)
        words = [w.lower() for w in tokens if w not in stops and w not in puns]
        return words
    else:
        return None

def doc2vec(doc, word2vec):
    if pd.notnull(doc):
        # 使用剛剛定義好的tokenize函式tokenize doc，並指派到terms
        # 找出每一個詞彙的代表向量(word2vec)
        # 並平均(element-wise)所有出現的詞彙向量(注意axis=0)，作為doc的代表向量
        terms = tokenize(doc)  # 把類別tokenize成一個個的詞彙
        termvecs = [word2vec.get(term) for term in terms if term in word2vec.keys()]
        docvec = np.average(np.array(termvecs), axis=0)
    
    if np.sum(np.isnan(docvec)) > 0:
        # 若找不到對應的詞向量，則給一條全部為零的向量，長度為原詞彙代表向量的長度(vec_dimensions)
        docvec=np.zeros(100, )  ## 先初始化一條向量，如果某個類別裡面的字都沒有在字典裡，那麼會回傳這條向量
    return docvec

def preprocessing_glove(df, vocab_size, max_length):
    """
    CountVectorizer
    """
    text = df['text'].tolist()
    label = df['label'].tolist()
    word_vec_mapping = {}
    path = "./glove_model/glove.twitter.27B.100d.txt"
    # 打開上述檔案，並將每一行中的第一個詞作為key，後面的數字做為向量，加入到word_vec_mapping
    with open(path, 'r', encoding='utf8') as f:  # 這個文檔的格式是一行一個字並配上他的向量，以空白鍵分隔
        for line in f:  
            tokens = line.split()
            token = tokens[0]  # 第一個token就是詞彙
            vec = tokens[1:]  # 後面的token向量
            word_vec_mapping[token] = np.array(vec, dtype=np.float32) # 把整個model做成一個字典，以利查找字對應的向量
    
    text_vecs = np.array([doc2vec(t, word_vec_mapping) for t in text])

    data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 100))
    text_vecs = data_scaler_minmax.fit_transform(text_vecs).astype(int)
    print(text_vecs[1])

    # 切分訓練集與測試集
    training_size=int(len(text)*0.8)

    train_seq=text_vecs[:training_size]
    test_seq=text_vecs[training_size:]

    train_labels=np.array(label[:training_size])
    test_labels=np.array(label[training_size:])
    print("Total no of Training Sequence are",len(train_seq))
    print("Total no of Test Sequence are",len(test_seq))

    return train_seq, train_labels, test_seq, test_labels

def preprocessing_no_split(df, vocab_size, max_length):
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

    X = pad_sequences(text,maxlen=max_length,padding ='post',truncating='post')

    Y = np.array(label)

    return X, Y

def select_rows(df):
    """
    選取 500, 1000, ... 2500 筆資料，測試不同資料量在相同演算法的差異
    """
    return df[:500], df[:1000], df[:1500], df[:2000], df[:2500], df
#### 以上為 preprocessing 函式 ################################################################################
# ----------------------------------------------------------------------------------------------------------#
#### 以下為 plot 相關設定與函式 ################################################################################
sns.set_style("darkgrid", {"axes.facecolor": ".7"})
sns.set_context(rc = {'patch.linewidth': 0.1})
cmap = sns.light_palette("#434343") # light:b #69d ch:start=.2,rot=-.3, ch:s=.25,rot=-.25 , dark:salmon_r

def plot_graphs(history, string, method, rows_num, target_column):
    plt.plot(history.history[string], color='white')
    plt.plot(history.history['val_'+string], color='black')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.savefig('./plot/training/' + method.lower() + '/' + rows_num + '_' +  string + '.png')
    plt.clf()
#### 以上為 plot 相關設定與函式 #################################################################################
# ----------------------------------------------------------------------------------------------------------#
#### 以下為 method 函式 #######################################################################################

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
    model.fit(X_train_dense, y_train.values, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,validation_split=config.VALIDATION_SIZE,callbacks=callback_list)

    pred_train = model.predict(X_train_dense,batch_size=config.BATCH_SIZE)
    pred_test = model.predict(X_test_dtm,batch_size=config.BATCH_SIZE)

    pred_test_round = np.around(pred_test)
    pred_test_int = pred_test_round.astype(int)
    correct_prediction = np.equal(pred_test_int, y_test.values)
    print(np.mean(correct_prediction))

def method_2_LSTM(df, target_column, max_length, vocab_size):
    """
    做 word to vector 之後使用 LSTM
    """
    rows_num = str(len(df.index))
    df = df.dropna()

    # train_x, train_y, val_x, val_y, test_x, test_y = preprocessing_split(df, target_column, vocab_size, max_length)
    train_x, train_y, val_x, val_y, test_x, test_y = tokenize_split(df, target_column, vocab_size, max_length)

    # Create model
    embedding_dim = max_length

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,return_sequences=True)),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10,activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./lstm_model/'  + rows_num + '_epoch-{epoch:02d}-val_accuracy-{val_accuracy:.2f}.h5',
            monitor='val_accuracy',
            mode='auto',
            save_best_only=True
            ),
    ]

    # opt_adam = tf.keras.optimizers.Adam(
    #     learning_rate=3e-4, # 3e-4
    #     beta_1=0.85,
    #     beta_2=0.99,
    #     epsilon=1e-07,
    #     amsgrad=True,
    #     name="Adam"
    # )

    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
    # model.summary()

    history = model.fit(
        train_x,
        train_y,
        epochs=config.EPOCHS,
        validation_data=(val_x, val_y),
        callbacks=my_callbacks
    )

    print("Evaluate on test data")
    test_results = model.evaluate(test_x, test_y, batch_size=32)
    print("test loss, test acc:", test_results)
    with open('./plot/training/lstm/test_result/'+rows_num+'_test_result.txt', 'w') as f:
        f.write("Test loss: {}\n".format(test_results[0]))
        f.write("Test acc: {}\n".format(test_results[1]))

    joblib.dump(history.history, './plot/training/lstm/hist_joblib/'+rows_num+'_hist.joblib')
    
    plot_graphs(history, "accuracy", 'LSTM', rows_num, target_column)
    plot_graphs(history, "loss", 'LSTM', rows_num, target_column)

class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))

def method_3_Attention(df, target_column, max_length, vocab_size):
    """
    Attention
    """
    rows_num = str(len(df.index))
    df = df.dropna()
    train_x, train_y, val_x, val_y, test_x, test_y = tokenize_split(df, target_column, vocab_size, max_length)

    # print("train[0]", x_train[0])

    # embed_dim = 12  # Embedding size for each token
    # num_heads = 6  # Number of attention heads
    # ff_dim = 8  # Hidden layer size in feed forward network inside transformer

    # inputs = layers.Input(shape=(max_length,))
    # embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
    # x = embedding_layer(inputs)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(0.95)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Dense(8)(x)
    # # x = layers.LeakyReLU()(x)
    # # x = layers.Dropout(0.6)(x)
    # # x = layers.BatchNormalization()(x)
    # # x = layers.Dense(4)(x)
    # # x = layers.LeakyReLU()(x)
    # # x = layers.Dropout(0.6)(x)
    # # x = layers.BatchNormalization()(x)
    # outputs = layers.Dense(2, activation="softmax")(x)

    # model = keras.Model(inputs=inputs, outputs=outputs)
    embed_dim = 128 # Embedding size for each token # 348
    num_heads = 4 # Number of attention heads
    ff_dim = 128 # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(max_length,))
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Dense(32)(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Dense(16)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    # x = layers.BatchNormalization()(x)

    outputs = layers.Dense(2, activation="softmax")(x)
    # outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    opt_adam = tf.keras.optimizers.Adam(
        learning_rate=3e-4, # 3e-4
        beta_1=0.85,
        beta_2=0.99,
        epsilon=1e-07,
        amsgrad=True,
        name="Adam"
    )

    # loss: categorical_crossentropy, sparse_categorical_crossentropy
    model.compile(optimizer=opt_adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Number of warmup epochs.
    warmup_epoch = 5
    sample_count = int(rows_num)
    b_s = 32
    # Compute the number of warmup batches.
    warmup_batches = warmup_epoch * sample_count / b_s

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./attention_model/'  + rows_num + '_epoch-{epoch:02d}-val_accuracy-{val_accuracy:.2f}.h5',
            monitor='val_accuracy',
            mode='auto',
            save_best_only=True
            ),
        WarmUpLearningRateScheduler(warmup_batches, init_lr=0.001)
    ]

    history = model.fit(
        train_x, train_y,
        batch_size=32,
        epochs=config.EPOCHS,
        validation_data=(val_x, val_y),
        callbacks=my_callbacks
    )

    print("Evaluate on test data")
    test_results = model.evaluate(test_x, test_y, batch_size=32)
    print("test loss, test acc:", test_results)
    with open('./plot/training/attention/test_result/'+rows_num+'_test_result.txt', 'w') as f:
        f.write("Test loss: {}\n".format(test_results[0]))
        f.write("Test acc: {}\n".format(test_results[1]))

    joblib.dump(history.history, './plot/training/attention/hist_joblib/'+rows_num+'_hist.joblib')

    # model.save('./model.h5')

    plot_graphs(history, "accuracy", 'Attention', rows_num, target_column)
    plot_graphs(history, "loss", 'Attention', rows_num, target_column)

def method_4_decisiontree(df):
    """
    決策樹
    """
    vocab_size = 1000
    max_length = 50
    train_seq, train_labels, test_seq, test_labels = preprocessing_split(df, vocab_size, max_length)
    
    classifier = tree.DecisionTreeClassifier(criterion='gini',max_depth=9,min_samples_split=2,min_impurity_decrease=0.0,ccp_alpha=0.0 )
    classifier = classifier.fit(train_seq, train_labels)
    tree.plot_tree(classifier)
    print(classifier.score(test_seq, test_labels))
    
def method_5_SVM(df):
    """
    支持向量機
    """
    vocab_size = 1000
    max_length = 50
    train_seq, train_labels, test_seq, test_labels = preprocessing_split(df, vocab_size, max_length)
    
    clf=svm.SVC(kernel='poly',C=1,gamma='auto')
    clf.fit(train_seq, train_labels)
    print(clf.score(test_seq, test_labels))
    #print(clf) 
    #print(clf.support_vectors_) #支援向量點 
    #print(clf.support_) #支援向量點的索引 
    #print(clf.n_support_) #每個class有幾個支援向量點 

def method_6_kag_ANN(df, target_column, max_length, vocab_size):
    """
    試試看kag的模型，莫名跑出89.5%
    """
    rows_num = str(len(df.index))
    df = df.dropna()

    # 先移除標點符號
    df['text_remove_puncs'] = df.text.apply(lambda x : punctuation_removal(x))

    # 再移除停用字
    df['text_remove_puncs_remove_stopwords'] = df.text_remove_puncs.apply(lambda x : remove_eng_stopwords(x))

    X_train, X_test, y_train, y_test = train_test_split(df['text_remove_puncs_remove_stopwords'], df['label'], test_size=config.TEST_SIZE, random_state=123)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    vect = TfidfVectorizer()
    
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    X_train_dense = X_train_dtm.todense()
    #X_test_dense = X_test_dtm.todense()
    print(X_train_dense.shape)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    #沒有加入batchnormalization也可以到87%
    model=Sequential([
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.25),
        layers.BatchNormalization(),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.25),
        layers.BatchNormalization(),
        layers.Dense(2,activation='sigmoid')
    ])

    model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

    callback_list = [
        EarlyStopping(patience=15, verbose=0),
        ModelCheckpoint(
            filepath='./ann_model/'  + rows_num + '_epoch-{epoch:02d}-val_accuracy-{val_accuracy:.2f}.h5',
            monitor='val_accuracy',
            mode='auto',
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train_dense, 
        y_train, 
        epochs=config.EPOCHS, 
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SIZE,
        callbacks=callback_list
    )

    joblib.dump(history.history, './plot/training/ann/hist_joblib/'+rows_num+'_hist.joblib')

    plot_graphs(history, "accuracy", 'ANN', rows_num, target_column)
    plot_graphs(history, "loss", 'ANN', rows_num, target_column)

#### 以上為 method 函式 #######################################################################################
# ----------------------------------------------------------------------------------------------------------#
#### 以下為資料分析函式 ########################################################################################

def data_analysis(df):
    """
    資料分析
    """
    volume_analysis(df)

    sent_length, sent_freq = length_analysis(df)

    length_cdf_analysis(df, sent_length, sent_freq)

    # 先移除標點符號
    df['text_remove_puncs'] = df.text.apply(lambda x : punctuation_removal(x))
    # 再移除停用字
    df['text_remove_puncs_remove_stopwords'] = df.text_remove_puncs.apply(lambda x : remove_eng_stopwords(x))
    puncs_stopword_removal_sent_length, puncs_stopword_removal_sent_freq = puncs_stopword_removal_length_analysis(df)
    puncs_stopword_removal_length_cdf_analysis(df, puncs_stopword_removal_sent_length, puncs_stopword_removal_sent_freq)

    top_n_common_words = find_top_n_common_words(df, 50)

    new_stop_words = generate_new_stop_words(top_n_common_words)

    df['text_remove_new_stopwords'] = df.text_remove_puncs_remove_stopwords.apply(lambda x : remove_new_stopwords(x, new_stop_words))

    generate_word_cloud(df)

    # 計算不同詞性的數量
    part_of_speech_analysis(df)

    # 使用 text_remove_new_stopwords column 做 word to vector

    # 方法一 GloVe pre-train model
    # 先用 glove model 生成對應每個 word 的 vector dict
    word_vec_mapping = generate_glove_word_vec_mapping_dict()
    df['w2v_glove'] = df.text_remove_new_stopwords.apply(lambda x : w2v_glove(x, word_vec_mapping))
    pca_2d_w2v_glove_result, pca_3d_w2v_glove_result, tsne_2d_w2v_glove_result, tsne_3d_w2v_glove_result = w2v_dim_reduction(df['w2v_glove'])
    df['pca_2d_w2v_glove'] = pca_2d_w2v_glove_result.tolist()
    df['pca_3d_w2v_glove'] = pca_3d_w2v_glove_result.tolist()
    df['tsne_2d_w2v_glove'] = tsne_2d_w2v_glove_result.tolist()
    df['tsne_3d_w2v_glove'] = tsne_3d_w2v_glove_result.tolist()
    visualize_vectors(df[['label', 'pca_2d_w2v_glove']], "2d", "pca", "glove")
    visualize_vectors(df[['label', 'pca_3d_w2v_glove']], "3d", "pca", "glove")
    visualize_vectors(df[['label', 'tsne_2d_w2v_glove']], "2d", "tsne", "glove")
    visualize_vectors(df[['label', 'tsne_3d_w2v_glove']], "3d", "tsne", "glove")
    
    # 方法二 TF-IDF
    v = TfidfVectorizer()
    tfidf_result = v.fit_transform(df['text_remove_new_stopwords'])
    df['w2v_tfidf'] = tfidf_result.toarray().tolist()
    pca_2d_w2v_tfidf_result, pca_3d_w2v_tfidf_result, tsne_2d_w2v_tfidf_result, tsne_3d_w2v_tfidf_result = w2v_dim_reduction(df['w2v_tfidf'])
    df['pca_2d_w2v_tfidf'] = pca_2d_w2v_tfidf_result.tolist()
    df['pca_3d_w2v_tfidf'] = pca_3d_w2v_tfidf_result.tolist()
    df['tsne_2d_w2v_tfidf'] = tsne_2d_w2v_tfidf_result.tolist()
    df['tsne_3d_w2v_tfidf'] = tsne_3d_w2v_tfidf_result.tolist()
    visualize_vectors(df[['label', 'pca_2d_w2v_tfidf']], "2d", "pca", "tfidf")
    visualize_vectors(df[['label', 'pca_3d_w2v_tfidf']], "3d", "pca", "tfidf")
    visualize_vectors(df[['label', 'tsne_2d_w2v_tfidf']], "2d", "tsne", "tfidf")
    visualize_vectors(df[['label', 'tsne_3d_w2v_tfidf']], "3d", "tsne", "tfidf")

    return df

#### 以上為資料分析函式 ########################################################################################
# ----------------------------------------------------------------------------------------------------------#
#### 以下為訓練不同資料量的函式 #################################################################################
def train_LSTM_models(df_500, df_1000, df_1500, df_2000, df_2500, df_3000, max_length, vocab_size):
    """
    一次訓練不同資料量的LSTM模型
    config.TARGET_COLUMNS[0] = "text"
    config.TARGET_COLUMNS[1] = "text_remove_puncs_remove_stopwords"
    config.TARGET_COLUMNS[2] = "text_remove_new_stopwords"
    """

    # 500 rows
    method_2_LSTM(df_500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 1000 rows
    method_2_LSTM(df_1000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 1500 rows
    method_2_LSTM(df_1500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 2000 rows
    method_2_LSTM(df_2000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 2500 rows
    method_2_LSTM(df_2500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 3000 rows
    method_2_LSTM(df_3000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

def train_Transformer_models(df_500, df_1000, df_1500, df_2000, df_2500, df_3000, max_length, vocab_size):
    """
    一次訓練不同資料量的 Transformer 模型
    config.TARGET_COLUMNS[1] = "text_remove_puncs_remove_stopwords"
    """
    # 500 rows
    method_3_Attention(df_500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 1000 rows
    method_3_Attention(df_1000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 1500 rows
    method_3_Attention(df_1500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 2000 rows
    method_3_Attention(df_2000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 2500 rows
    method_3_Attention(df_2500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 3000 rows
    method_3_Attention(df_3000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

def train_ANN_models(df_500, df_1000, df_1500, df_2000, df_2500, df_3000, max_length, vocab_size):
    """
    一次訓練不同資料量的 ANN 模型
    config.TARGET_COLUMNS[1] = "text_remove_puncs_remove_stopwords"
    """
    # 500 rows
    method_6_kag_ANN(df_500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 1000 rows
    method_6_kag_ANN(df_1000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 1500 rows
    method_6_kag_ANN(df_1500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 2000 rows
    method_6_kag_ANN(df_2000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 2500 rows
    method_6_kag_ANN(df_2500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

    # 3000 rows
    method_6_kag_ANN(df_3000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size)

#### 以上為訓練不同資料量的函式 #################################################################################
# ----------------------------------------------------------------------------------------------------------#
#### 以下為程式主函式 #########################################################################################
def main():
    """
    英文 NLP - Sentiment analysis
    hidden layer nodes number 
    Ref:https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", required=False, default="n")
    args = parser.parse_args()

    if args.preprocess.lower() == 'y':
        df = load_data(config.INPUT_PATH)

        processed_df = data_analysis(df)

        processed_df.to_csv ('./processed_data.csv', index = False, header=True)
    else:
        print("read processed data CSV")
        processed_df = pd.read_csv("./processed_data.csv")

    df_500, df_1000, df_1500, df_2000, df_2500, df_3000 = select_rows(processed_df)

    # 確認第一筆資料
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(processed_df.head(1))

    # X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=config.TEST_SIZE, random_state=123)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # method_1_ann(X_train, X_test, y_train, y_test)

    trainig_set = df_3000[:2400]
    # max_length = int(trainig_set['text_remove_puncs_remove_stopwords'].str.encode(encoding='utf-8').str.len().max())
    max_length = 92
    print("Max length", max_length)

    unique_words = set()
    trainig_set['text_remove_puncs_remove_stopwords'].dropna().str.lower().str.split().apply(unique_words.update)
    vocab_size = len(unique_words)
    print("Vocab size: ", vocab_size)

    # train_LSTM_models(df_500, df_1000, df_1500, df_2000, df_2500, df_3000, max_length, vocab_size)

    # train_Transformer_models(df_500, df_1000, df_1500, df_2000, df_2500, df_3000, max_length, vocab_size)

    # train_ANN_models(df_500, df_1000, df_1500, df_2000, df_2500, df_3000, max_length, vocab_size)

if __name__ == "__main__":
    main()