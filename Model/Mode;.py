import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

train=pd.read_csv('../nlp-getting-started/train.csv')
test = pd.read_csv('../nlp-getting-started/test.csv')
def includes_website(text):
    website = int('http' in text)
    return website
def hashtags(text):
    hashtags=[]
    for word in text.split():
        if '#' in word:
            if word.count('#') == 1:
                hashtags.append(word[word.find('#')+1:])
            else:
                for subword in word[word.find('#'):].split('#')[1:]:
                    hashtags.append(subword)
    return ' '.join(hashtags)
def mentions(text):
    mentions=[]
    for word in text.split():
        if '@' in word:
            if word.count('@') == 1:
                mentions.append(word[word.find('@')+1:])
            else:
                for subword in word[word.find('@'):].split('@')[1:]:
                    mentions.append(subword)
    return mentions
def remove_extras(text):
    out_text=[]
    for word in text.split():
        if '#' in word:
            out_text.append(word[0:word.find('#')])
        elif '@' in word:
            out_text.append(word[0:word.find('@')])
        elif 'http' in word:
            out_text.append(word[0:word.find('http')])
        else:
            out_text.append(word)
    return ' '.join(out_text).replace('  ',' ')
def alphanumeric(text):
    output = re.sub(r'[^A-Za-z0-9 ]+','',text)
    return output.replace('  ',' ')
stopwords = []
for word in nltk.corpus.stopwords.words('english'):
    stopwords.append(alphanumeric(word))
stopwords = set(stopwords)
def remove_stopwords(text):
    out = []
    for word in text.split():
        if word.lower() in stopwords:
            pass
        else:
            out.append(word.lower())
    return ' '.join(out)
def preprocess(data):
    data['includes_website'] = data.text.apply(lambda x: includes_website(x))
    data['hashtags'] = data.text.apply(lambda x: hashtags(x))
    data['num_hashtags'] = data.hashtags.apply(lambda  x: len(x.split()))
    data['mentions'] = data.text.apply(lambda x: mentions(x))
    data['num_mentions'] = data.mentions.apply(lambda x:len(x))
    data['tweet_length'] = data.text.apply(lambda x: len(x))
    data['clean_text'] = data.text.apply(lambda x: remove_stopwords(alphanumeric(remove_extras(x))))
    return data

train_preprocessed = preprocess(train)
test_preprocessed = preprocess(test)
train_set, validation_set = train_test_split(train_preprocessed, train_size=0.8, random_state=77)
CV = CountVectorizer()
train_vectoriser = CV.fit_transform(train_set.clean_text.values)
validation_vectoriser = CV.transform(validation_set.clean_text.values)
HCV = CountVectorizer()
train_hashtag_vectoriser = HCV.fit_transform(train_set.hashtags.values)
validation_hashtag_vetoriser = HCV.transform(validation_set.hashtags.values)
def column_names(vectoriser, prefix):
    columns = []
    for column in vectoriser.get_feature_names():
        columns.append(prefix+'_'+column)
    return columns
train_text_dense = pd.DataFrame(train_vectoriser.todense(),columns=column_names(CV,'text'))
valid_text_dense = pd.DataFrame(validation_vectoriser.todense(),columns=column_names(CV,'text'))
train_hash_dense = pd.DataFrame(train_hashtag_vectoriser.todense(),columns=column_names(HCV,'hash'))
valid_hash_dense = pd.DataFrame(validation_hashtag_vetoriser.todense(),columns=column_names(HCV,'hash'))
keyword_labeller = LabelBinarizer()
train_keywords = keyword_labeller.fit_transform(train_set.keyword.astype(str))
valid_keywords = keyword_labeller.transform(validation_set.keyword.astype(str))
keyword_cols = []
for keyword in keyword_labeller.classes_:
    keyword_cols.append('keyword_'+keyword)
train_keywords_df = pd.DataFrame(train_keywords, columns=keyword_cols)
valid_keywords_df =pd.DataFrame(valid_keywords, columns=keyword_cols)
train_filtered = train_set[['target', 'includes_website','num_hashtags', 'num_mentions', 'tweet_length']].reset_index()
valid_filtered = validation_set[['target', 'includes_website','num_hashtags', 'num_mentions','tweet_length']].reset_index()
train_full = pd.concat([train_filtered,train_text_dense, train_hash_dense, train_keywords_df], axis=1)
valid_full = pd.concat([valid_filtered,valid_text_dense, valid_hash_dense, valid_keywords_df],axis=1)
print(valid_full)

