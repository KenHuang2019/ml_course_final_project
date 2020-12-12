"""
Sentiment analysis
Ref
https://github.com/GoatWang/IIIMaterial/blob/master/08_InformationRetreival/main08.ipynb
"""
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

stops = stopwords.words('english')
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()


testStr = "This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts."
# 請使用nltk.word_tokenize及nltk.wordpunct_tokenize進行分詞，並比較其中差異。
#=============your works starts===============#
word_tokenize_tokens = nltk.word_tokenize(testStr)
wordpunct_tokenize_tokens = nltk.wordpunct_tokenize(testStr)
#==============your works ends================#

print("/".join(word_tokenize_tokens))
print("/".join(wordpunct_tokenize_tokens))