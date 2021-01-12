import config # 路徑設定

import re
import numpy as np
import pandas as pd

import nltk
import string

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from itertools import accumulate, chain
from collections import Counter

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import random

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

eng_stopwords = nltk.corpus.stopwords.words("english")
sns.set_style("darkgrid", {"axes.facecolor": ".7"})
sns.set_context(rc = {'patch.linewidth': 0.1})
cmap = sns.light_palette("#434343") # light:b #69d ch:start=.2,rot=-.3, ch:s=.25,rot=-.25 , dark:salmon_r

def volume_analysis(df):
    """
    資料量分析
    """
    volume_df = pd.DataFrame({
        "label": ["Positive", "Negative"],
        "volume": [df['label'].value_counts().loc[1], df['label'].value_counts().loc[0]]
    })
    
    sns.barplot(x='label', y='volume', data=volume_df, palette=cmap)
    plt.margins(0.02)
 
    plt.title('Data volume analysis')
    plt.ylim([0, 1600])
    plt.yticks(np.arange(0, 1600, 500))

    plt.savefig(config.PLOT_PATH + "analysis/" + "data_volume_analysis.png")
    plt.cla()
    plt.clf()

def length_analysis(df):
    """
    句子長度分析
    """
    df['length'] = df['text'].apply(lambda x: len(x))
    
    len_df = df.groupby('length').count()
    sent_length = len_df.index.tolist()
    sent_freq = len_df['text'].tolist()
    print("sent_length: ", max(sent_length))
    print("top 5 of sent_length: ", sorted(sent_length, reverse=True)[:5])
    print("sent_freq: ", max(sent_freq))
    print("top 5 of sent_freq: ", sorted(sent_freq, reverse=True)[:5])

    # 繪製句子長度及出現頻數統計圖
    tmp_df = pd.DataFrame({
        "sentence_length": sent_length,
        "sentence_frequency": sent_freq
    })
    
    sns.barplot(x='sentence_length', y='sentence_frequency', data=tmp_df, palette=cmap)
    plt.margins(0.02)

    plt.title("Sentnece length and frequency") # , fontproperties=my_font
    plt.xlabel("length") # , fontproperties=my_font
    plt.ylabel("frequency") # , fontproperties=my_font
    # plt.xlim([0, 480])
    plt.xticks(np.arange(0, max(sent_length), step=20), rotation=90, fontsize=8)
    plt.ylim([0, 55])
    plt.savefig(config.PLOT_PATH + "analysis/" + "sentnece_length_frequency.png")
    plt.cla()
    plt.clf()

    return sent_length, sent_freq

def length_cdf_analysis(df, sent_length, sent_freq):
    """
    句子長度 累積分佈函式 ( CDF, Cumulative Distribution Function )
    """
    sent_pentage_list = [(count/sum(sent_freq)) for count in accumulate(sent_freq)]

    # 繪製CDF
    plt.plot(sent_length, sent_pentage_list)

    # 尋找分位點為 quantile 的句子長度
    quantile = 0.95
    #print(list(sent_pentage_list))
    for length, per in zip(sent_length, sent_pentage_list):
        if round(per, 2) == quantile:
            index = length
            break
    print("\n分位點為%s的句子長度:%d." % (quantile, index))
    # 繪製句子長度累積分佈函式圖
    plt.plot(sent_length, sent_pentage_list, 'k')
    plt.hlines(quantile, 0, index, colors="w", linestyles="dashed")
    plt.vlines(index, 0, quantile, colors="w", linestyles="dashed")
    plt.text(0, quantile, str(quantile))
    plt.text(index, 0, str(index))
    plt.title("Sentence length Cumulative Distribution") # , fontproperties=my_font
    plt.xlabel("length") # , fontproperties=my_font
    plt.ylabel("frequency") # , fontproperties=my_font
    plt.savefig(config.PLOT_PATH + "analysis/" + "length_frequency_cumulative_distribution.png")
    plt.cla()
    plt.clf()

def puncs_stopword_removal_length_analysis(df):
    """
    去除標點符號、停用字後的句子長度分析
    """
    df['puncs_stopword_removal_length'] = df['text_remove_puncs_remove_stopwords'].apply(lambda x: len(x))
    
    len_df = df.groupby('puncs_stopword_removal_length').count()
    sent_length = len_df.index.tolist()
    sent_freq = len_df['text_remove_puncs_remove_stopwords'].tolist()
    print("puncs_stopword_removal_sent_length: ", max(sent_length))
    print("top 5 of sent_length: ", sorted(sent_length, reverse=True)[:5])
    print("puncs_stopword_removal_sent_freq: ", max(sent_freq))
    print("top 5 of sent_freq: ", sorted(sent_freq, reverse=True)[:5])

    # 繪製句子長度及出現頻數統計圖
    tmp_df = pd.DataFrame({
        "sentence_length": sent_length,
        "sentence_frequency": sent_freq
    })

    # 繪製句子長度及出現頻數統計圖
    sns.barplot(x='sentence_length', y='sentence_frequency', data=tmp_df, palette=cmap)
    plt.margins(0.02)
    # plt.bar(sent_length, sent_freq)
    plt.title("Sentnece length and frequency (Remove stopwords and puncs)") # , fontproperties=my_font
    plt.xlabel("length") # , fontproperties=my_font
    plt.ylabel("frequency") # , fontproperties=my_font
    # plt.xlim([0, 350])
    plt.xticks(np.arange(0, max(sent_length), step=20), rotation=90, fontsize=8)
    plt.ylim([0, 85])
    plt.savefig(config.PLOT_PATH + "analysis/" + "puncs_stopword_removal_sentnece_length_frequency.png")
    plt.cla()
    plt.clf()

    return sent_length, sent_freq

def puncs_stopword_removal_length_cdf_analysis(df, sent_length, sent_freq):
    """
    去除停用字之後的句子長度 累積分佈函式 ( CDF, Cumulative Distribution Function )
    """
    sent_pentage_list = [(count/sum(sent_freq)) for count in accumulate(sent_freq)]

    # 繪製CDF
    plt.plot(sent_length, sent_pentage_list)

    # 尋找分位點為 quantile 的句子長度
    quantile = 0.95
    #print(list(sent_pentage_list))
    for length, per in zip(sent_length, sent_pentage_list):
        if round(per, 2) == quantile:
            index = length
            break
    print("\n分位點為%s的句子長度:%d." % (quantile, index))
    # 繪製句子長度累積分佈函式圖
    plt.plot(sent_length, sent_pentage_list, 'k')
    plt.hlines(quantile, 0, index, colors="w", linestyles="dashed")
    plt.vlines(index, 0, quantile, colors="w", linestyles="dashed")
    plt.text(0, quantile, str(quantile))
    plt.text(index, 0, str(index))
    plt.title("Sentence length Cumulative Distribution (Remove stopwords and puncs)") # , fontproperties=my_font
    plt.xlabel("length") # , fontproperties=my_font
    plt.ylabel("frequency") # , fontproperties=my_font
    plt.savefig(config.PLOT_PATH + "analysis/" + "puncs_stopword_removal_length_frequency_cumulative_distribution.png")
    plt.cla()
    plt.clf()

def remove_eng_stopwords(text):
    token_text = nltk.word_tokenize(text)
    remove_stop = [word for word in token_text if word not in eng_stopwords]
    join_text = ' '.join(remove_stop)
    return join_text

def punctuation_removal(x):
    text = x
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
    text = re.sub('\n', '', text)
    return text

def find_top_n_common_words(df, top_n):
    """
    找出共同使用的字，將其去除可增加每筆資料之間的差異性，更容易區分不同標籤鎖對應到的文本
    """
    list_words = df['text_remove_puncs_remove_stopwords'].str.split()
    list_words_merge = list(chain(*list_words))
    d = Counter(list_words_merge)
    common_words_df = pd.DataFrame(data=d, index=['count'])
    top_common_words = common_words_df.T.sort_values(by=['count'], ascending=False).reset_index().head(top_n)

    plt.figure(figsize=(15,12))
    sns.barplot(x="index", y='count', data=top_common_words, palette=cmap)
    plt.xticks(rotation=90,fontsize=8)
    plt.margins(0.02)
    plt.savefig(config.PLOT_PATH + "analysis/" + "top_" + str(top_n) + "_common_words.png", bbox_inches='tight')
    plt.cla()
    plt.clf()
    return top_common_words

def remove_new_stopwords(text, new_stop_words):
    token_text = nltk.word_tokenize(text)
    remove_stop = [word for word in token_text if word not in new_stop_words]
    join_text = ' '.join(remove_stop)
    return join_text

def generate_new_stop_words(top_n_common_words):
    """
    把太常出現且與情緒無關的字詞找出來作為新的停用詞
    """
    common_words_value = top_n_common_words['index'].values
    remove_words = ['dont', 'amazing', 'delicious', 'good', 'great', 'like', 'bad', 'best', 'well', 'love', 'nice', 'pretty', 'friendly', 'better', 'disappointed', 'didnt', 'wont']
    return [x for x in common_words_value if x not in remove_words]

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

def generate_word_cloud(df):
    """
    生成正負面的文字雲
    """
    original_positive_text = df[df.label==1]['text'].values
    original_negative_text = df[df.label==0]['text'].values
    processed_positive_text = df[df.label==1]['text_remove_puncs_remove_stopwords'].values
    processed_negative_text = df[df.label==0]['text_remove_puncs_remove_stopwords'].values

    wc = WordCloud(
        background_color="black", 
        max_words=10000,
        stopwords=STOPWORDS, 
        max_font_size=60,
        width=800,
        height=400
    ).generate(" ".join(original_positive_text))

    wc.to_file(config.PLOT_PATH + "analysis/" + 'original_positive_word_cloud.png')

    wc = WordCloud(
        background_color="black", 
        max_words=10000,
        stopwords=STOPWORDS, 
        max_font_size=60,
        width=800,
        height=400
    ).generate(" ".join(original_negative_text))
    wc.recolor(color_func=grey_color_func, random_state=3)
    wc.to_file(config.PLOT_PATH + "analysis/" + 'original_negative_word_cloud.png')

    wc = WordCloud(
        background_color="black", 
        max_words=10000,
        stopwords=STOPWORDS, 
        max_font_size=60,
        width=800,
        height=400
    ).generate(" ".join(processed_positive_text))

    wc.to_file(config.PLOT_PATH + "analysis/" + 'processed_positive_word_cloud.png')

    wc = WordCloud(
        background_color="black", 
        max_words=10000,
        stopwords=STOPWORDS, 
        max_font_size=60,
        width=800,
        height=400
    ).generate(" ".join(processed_negative_text))
    wc.recolor(color_func=grey_color_func, random_state=3)
    wc.to_file(config.PLOT_PATH + "analysis/" + 'processed_negative_word_cloud.png')

def noun_num(row):
    """function to give us fraction of noun over total words """
    text = row['text_remove_new_stopwords']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = nltk.pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    return noun_count

def adj_num(row):
    """function to give us fraction of adjectives over total words in given text"""
    text = row['text_remove_new_stopwords']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    return adj_count

def verbs_num(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text_remove_new_stopwords']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return verbs_count

def part_of_speech_analysis(df):
    """
    詞性分析
    """
    df['noun_num'] = df.apply(lambda row: noun_num(row), axis =1)
    df['adj_num'] = df.apply(lambda row: adj_num(row), axis =1)
    df['verbs_num'] = df.apply(lambda row: verbs_num(row), axis =1)

    positive_text = df[df['label']==0]
    negative_text = df[df['label']==1]

    positive_df = pd.DataFrame({
        "part_of_speech": ["noun", "adj", "verbs"],
        "num": [positive_text['noun_num'].sum(), positive_text['adj_num'].sum(), positive_text['verbs_num'].sum()]
    })
    
    sns.barplot(x='part_of_speech', y='num', data=positive_df, palette=cmap)
    plt.margins(0.02)
 
    plt.title('Part of speech analysis - Positive', fontsize=16)
    plt.ylim([0, 4000])
    plt.xticks(fontsize=16)
    plt.yticks(np.arange(0, 4000, 500), fontsize=12)
    plt.xlabel("part of speech", fontsize=14)
    plt.ylabel("number", fontsize=14)
    plt.savefig(config.PLOT_PATH + "analysis/" + "part_of_speech_analysis_positive.png", bbox_inches='tight')
    plt.cla()
    plt.clf()

    negative_df = pd.DataFrame({
        "part_of_speech": ["noun", "adj", "verbs"],
        "num": [negative_text['noun_num'].sum(), negative_text['adj_num'].sum(), negative_text['verbs_num'].sum()]
    })
    
    sns.barplot(x='part_of_speech', y='num', data=negative_df, palette=cmap)
    plt.margins(0.02)
 
    plt.title('Part of speech analysis - Negative', fontsize=16)
    plt.ylim([0, 4000])
    plt.xticks(fontsize=16)
    plt.yticks(np.arange(0, 4000, 500), fontsize=12)
    plt.xlabel("part of speech", fontsize=14)
    plt.ylabel("number", fontsize=14)
    plt.savefig(config.PLOT_PATH + "analysis/" + "part_of_speech_analysis_negative.png", bbox_inches='tight')
    plt.cla()
    plt.clf()

def w2v_glove(text, word_vec_mapping):
    """
    使用預訓練的 glove model 做 word to vector
    """
    if pd.notnull(text):
        # 使用剛剛定義好的tokenize函式tokenize doc，並指派到terms
        # 找出每一個詞彙的代表向量(word_vec_mapping)
        # 並平均(element-wise)所有出現的詞彙向量(注意axis=0)，作為doc的代表向量
        terms = [w.lower() for w in nltk.wordpunct_tokenize(text)]  ## 把類別tokenize成一個個的詞彙
        termvecs = [word_vec_mapping.get(term) for term in terms if term in word_vec_mapping.keys()]
        text_vec = np.average(np.array(termvecs), axis=0)
    
    if np.sum(np.isnan(text_vec)) > 0:
        ## 若找不到對應的詞向量，則給一條全部為零的向量，長度為原詞彙代表向量的長度(vec_dimensions)
        text_vec=np.zeros(100, )  ## 先初始化一條向量，如果某個類別裡面的字都沒有在字典裡，那麼會回傳這條向量
        # text_vec=np.zeros(25, )  ## 先初始化一條向量，如果某個類別裡面的字都沒有在字典裡，那麼會回傳這條向量

    return text_vec

def generate_glove_word_vec_mapping_dict():
    """
    避免重複生成 word vector dictionary
    """
    word_vec_mapping = {}

    path = "./glove_model/glove.twitter.27B.100d.txt"
    # path = "./glove_model/glove.twitter.27B.25d.txt"

    # 打開上述檔案，並將每一行中的第一個詞作為key，後面的數字做為向量，加入到word_vec_mapping
    with open(path, 'r', encoding='utf8') as f:  # 這個文檔的格式是一行一個字並配上他的向量，以空白鍵分隔
        for line in f:  
            tokens = line.split()
            token = tokens[0]  # 第一個token就是詞彙
            vec = tokens[1:]  # 後面的token向量
            word_vec_mapping[token] = np.array(vec, dtype=np.float32) # 把整個model做成一個字典，以利查找字對應的向量

    return word_vec_mapping

def w2v_dim_reduction(df_vectors):
    """
    使用 PCA 降維成 2 and 3 維
    """
    w2v_glove_arr = df_vectors.values
    target = []
    for v in w2v_glove_arr:
        target.append(v)

    d = 3 # 要降成多少維度

    pca_3d = PCA(n_components=d)
    pca_3d.fit(target)

    pca_2d = PCA(n_components=(d-1))
    pca_2d.fit(target)

    return pca_2d.transform(target), pca_3d.transform(target), TSNE(n_components=(d-1), perplexity=50).fit_transform(target), TSNE(n_components=d, perplexity=50).fit_transform(target)

def visualize_vectors(df, dim, method, vec_type):
    """
    將 PCA 降維後的數據可視化
    """
    fig = plt.figure()

    if dim == "2d":
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')

    for index, row in df.iterrows():
        coordinates = row[method + '_' + dim + '_w2v_' + vec_type]
        color = 'b'
        if row['label'] == 1:
            color = 'r'
        if dim == "2d":
            ax.scatter(coordinates[0], coordinates[1], c=color, s=0.8)
        else:
            ax.scatter(coordinates[0], coordinates[1], coordinates[2], c=color, s=0.8)

    plt.title(vec_type + " vectors " + method + " - " + dim)

    plt.savefig(config.PLOT_PATH + "analysis/" + method + "_" + dim + "_w2v_" + vec_type + ".png")
    plt.cla()
    plt.clf()
