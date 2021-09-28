import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

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
test_hashtag_vectoriser = HCV.fit_transform(train_set.hashtags.values)
validation_hashtag_vetoriser = HCV.transform(validation_set.hashtags.values)
print(HCV.get_feature_names())


