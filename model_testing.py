import os
import config
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sentiment_analysis import select_rows, tokenize_split
from multi_head_self_attention import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding
# data analysis
from data_analysis_func import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

def lstm_model_test(df, target_column, max_length, vocab_size, model_path):
    """
    測試每個資料量的 LSTM 最佳模型
    """
    print("Model path: ", model_path)
    df = df.dropna()

    train_x, train_y, val_x, val_y, test_x, test_y = tokenize_split(df, target_column, vocab_size, max_length)

    # 讀取model
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

    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    model.load_weights(model_path)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_x, test_y)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

def attention_model_test(df, target_column, max_length, vocab_size, model_path):
    """
    測試每個資料量的 Attention 最佳模型
    """
    print("Model path: ", model_path)
    df = df.dropna()

    train_x, train_y, val_x, val_y, test_x, test_y = tokenize_split(df, target_column, vocab_size, max_length)
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
    x = layers.Dense(16)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    opt_adam = tf.keras.optimizers.Adam(
        learning_rate=3e-4, # 3e-4
        beta_1=0.85,
        beta_2=0.99,
        epsilon=1e-07,
        amsgrad=True,
        name="Adam"
    )

    model.compile(optimizer=opt_adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.load_weights(model_path)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_x, test_y)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

def ann_model_test(df, target_column, max_length, vocab_size, model_path):
    """
    docstring
    """
    rows_num = str(len(df.index))
    # df = df.dropna()

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

    model.load_weights(model_path)

    # Re-evaluate the model
    loss, acc = model.evaluate(X_test, y_test)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

if __name__ == "__main__":
    processed_df = pd.read_csv("./processed_data.csv")

    df_500, df_1000, df_1500, df_2000, df_2500, df_3000 = select_rows(processed_df)

    max_length = 92
    vocab_size = 4456

    # # LSTM model testing
    # lstm_model_dir = "./lstm_model/best_model/"

    # lstm_model_names = sorted(os.listdir(lstm_model_dir), reverse=True)
    # lstm_model_names.append(lstm_model_names[0])
    # lstm_model_names.pop(0)

    # lstm_model_test(df_500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, lstm_model_dir + lstm_model_names[5])
    # print("==============================================================================================================================")
    # lstm_model_test(df_1000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, lstm_model_dir + lstm_model_names[4])
    # print("==============================================================================================================================")
    # lstm_model_test(df_1500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, lstm_model_dir + lstm_model_names[3])
    # print("==============================================================================================================================")
    # lstm_model_test(df_2000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, lstm_model_dir + lstm_model_names[2])
    # print("==============================================================================================================================")
    # lstm_model_test(df_2500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, lstm_model_dir + lstm_model_names[1])
    # print("==============================================================================================================================")
    # lstm_model_test(df_3000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, lstm_model_dir + lstm_model_names[0])
    # print("==============================================================================================================================")

    # # attention model testing
    # attention_model_dir = "./attention_model/best_model/"

    # attention_model_names = sorted(os.listdir(attention_model_dir), reverse=True)
    # attention_model_names.append(attention_model_names[0])
    # attention_model_names.pop(0)

    # attention_model_test(df_500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, attention_model_dir + attention_model_names[5])
    # print("==============================================================================================================================")
    # attention_model_test(df_1000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, attention_model_dir + attention_model_names[4])
    # print("==============================================================================================================================")
    # attention_model_test(df_1500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, attention_model_dir + attention_model_names[3])
    # print("==============================================================================================================================")
    # attention_model_test(df_2000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, attention_model_dir + attention_model_names[2])
    # print("==============================================================================================================================")
    # attention_model_test(df_2500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, attention_model_dir + attention_model_names[1])
    # print("==============================================================================================================================")
    # attention_model_test(df_3000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, attention_model_dir + attention_model_names[0])
    
    # ann model testing
    ann_model_dir = "./ann_model/best_model/"

    ann_model_names = sorted(os.listdir(ann_model_dir), reverse=True)
    ann_model_names.append(ann_model_names[0])
    ann_model_names.pop(0)
    ann_model_test(df_500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, ann_model_dir + ann_model_names[5])
    print("==============================================================================================================================")
    ann_model_test(df_1000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, ann_model_dir + ann_model_names[4])
    print("==============================================================================================================================")
    ann_model_test(df_1500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, ann_model_dir + ann_model_names[3])
    print("==============================================================================================================================")
    ann_model_test(df_2000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, ann_model_dir + ann_model_names[2])
    print("==============================================================================================================================")
    ann_model_test(df_2500.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, ann_model_dir + ann_model_names[1])
    print("==============================================================================================================================")
    ann_model_test(df_3000.reset_index(drop=True), config.TARGET_COLUMNS[1], max_length, vocab_size, ann_model_dir + ann_model_names[0])