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

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Attention
from multi_head_self_attention import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding

# data analysis
from data_analysis_func import *

eng_stopwords = nltk.corpus.stopwords.words("english")
sns.set_style("darkgrid", {"axes.facecolor": ".7"})
sns.set_context(rc = {'patch.linewidth': 0.1})
cmap = sns.light_palette("#434343") # light:b #69d ch:start=.2,rot=-.3, ch:s=.25,rot=-.25 , dark:salmon_r

def load_data(p, sub_set_num=None):
    """
    讀取資料集
    """
    file_names = [f for f in listdir(p) if isfile(join(p, f))]

    df_yelp = pd.read_table( p + file_names[0] ,header=None,sep='\t', names=["text", "label"], quoting=3)
    df_imdb = pd.read_table( p + file_names[1] ,header=None,sep='\t', names=["text", "label"], quoting=3)
    df_amazon = pd.read_table( p + file_names[0] ,header=None,sep='\t', names=["text", "label"], quoting=3)
    df = pd.concat([df_amazon,df_yelp,df_imdb])

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

def plot_graphs(history, string, method):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    # plt.show()
    plt.savefig(method + '_' + string + '.png')
    plt.clf()

#########################################################################################

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

def preprocessing_split(df, vocab_size, max_length):
    """
    切分訓練集、測試集
    """
    text = df['text'].tolist()
    label = df['label'].tolist()

    # word to vector
    print("vocab size is", vocab_size)
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(text,vocab_size,max_subword_length=10)
    
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

    # vec_dimensions = len(word_vec_mapping.get('men'))
    # print("vec_dimensions:", vec_dimensions)
    # print("word_vec_mapping length:", len(list(word_vec_mapping.items())))
    # print(doc2vec(text[2], word_vec_mapping))
    # print(len(text))
    
    text_vecs = np.array([doc2vec(t, word_vec_mapping) for t in text])
    print(text_vecs[1])

    # norm = np.linalg.norm(text_vecs)
    # text_vecs = text_vecs/norm

    # print(text_vecs.shape)
    # print(text_vecs[1])

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

def method_2_LSTM(df):
    """
    做 word to vector 之後使用 LSTM
    """
    vocab_size = 1000
    max_length = 50
    train_seq, train_labels, test_seq, test_labels = preprocessing_split(df, vocab_size, max_length)

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
    x_train, y_train, x_val, y_val = preprocessing_split(df, vocab_size, max_length)
    
    # x_train, y_train, x_val, y_val = preprocessing_glove(df, vocab_size, max_length)# 可用在 SVM or Decision tree
    
    # X, Y = preprocessing_no_split(df, vocab_size, max_length)

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

    print(model.summary())

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-1,
        decay_steps=5,
        decay_rate=0.9
    )
    opt_sgd = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.5)

    opt_adagrad = tf.keras.optimizers.Adagrad(
        learning_rate=0.0001,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        name="Adagrad",
    )

    # opt = keras.optimizers.Adam(learning_rate=0.001)
    opt_adam = tf.keras.optimizers.Adam(
        learning_rate=3e-4, # 3e-4
        beta_1=0.85,
        beta_2=0.99,
        epsilon=1e-07,
        amsgrad=True,
        name="Adam"
    )

    opt_adadelta = tf.keras.optimizers.Adadelta(
        learning_rate=0.1, rho=0.9, epsilon=1e-07, name="Adadelta"
    )
    # loss: categorical_crossentropy, sparse_categorical_crossentropy
    model.compile(optimizer=opt_adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=60, validation_data=(x_val, y_val)#,callbacks=my_callbacks
    )

    # model.save('./model.h5')

    # 自動切割 validation set
    # history = model.fit(
    #     X, Y, batch_size=32, epochs=60, validation_split=0.2
    # )

    plot_graphs(history, "accuracy", 'Attention')
    plot_graphs(history, "loss", 'Attention')

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

def method_6_kag_ANN(df):
    """
    試試看kag的模型，莫名跑出89.5%
    """
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
        Dense(128,activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        Dense(64,activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        Dense(2,activation='sigmoid')
    ])

    model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=0)
    checkpoint =ModelCheckpoint(str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+".hdf5",save_best_only=True)
    callback_list=[earlystopper,checkpoint]
    history=model.fit(X_train_dense, y_train, epochs=100, batch_size=config.BATCH_SIZE,validation_split=config.VALIDATION_SIZE,callbacks=callback_list)

#########################################################################################

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

def main():
    """
    英文 NLP - Sentiment analysis
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
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(processed_df.head(1))

    # X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=config.TEST_SIZE, random_state=123)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # method_1_ann(X_train, X_test, y_train, y_test)

    # method_2_LSTM(df.reset_index(drop=True))

    # method_3_Attention(df.reset_index(drop=True))

    # method_4_decisiontree(df.reset_index(drop=True))
    
    # method_5_SVM(df.reset_index(drop=True))

if __name__ == "__main__":
    main()