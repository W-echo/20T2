import sys
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
# nltk.download('vader_lexicon')


train_file = sys.argv[1]
test_file = sys.argv[2]

url_reg = r'[a-z]*[:.]+\S+'     # urls in text
stop_words = set(stopwords.words('english'))

def predict_and_test(model, X_test_bag_of_words):
    predicted_y = model.predict(X_test_bag_of_words)
    # print(y_test, predicted_y)
    # print(model.predict_proba(X_test_bag_of_words))
    print(classification_report(y_test, predicted_y))


def read_data(file):
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.replace("\n", "")
        data.append(line.split("\t"))
    f.close()

    def clean_en_text(text):
        # keep English, digital and space
        comp = re.compile('[^A-Z^a-z^0-9^#@_$%^ ]')
        return comp.sub('', text)

    # preprocessing by removing junk characters
    for i in range(0, len(data)):
        # remove URLS
        data[i][1] = re.sub(url_reg, '', data[i][1])
        # remove invalid symbols
        data[i][1] = clean_en_text(data[i][1])

    # remove stopwords in English and process porter stemming
    porter = PorterStemmer()
    for i in range(0, len(data)):
        word_tokens = word_tokenize(data[i][1])
        # print(data[i][1])
        filtered_sentence = ''
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence = filtered_sentence + ' ' + porter.stem(w).lower()
        data[i][1] = filtered_sentence

    return data

# Create text
f = open(train_file,'r')
train_data = read_data(f)

f = open(test_file,'r')
test_data = read_data(f)

# split into train and test
train_data = np.array(train_data)
test_data = np.array(test_data)

# p = 0.8
# split = int(len(train_data)*0.8)
X_train = train_data[:, 1]
y_train = train_data[:, -1]
test_id = test_data[:, 0]
X_test = test_data[:, 1]
y_test = test_data[:, -1]


# most frequent 1000 words
# create count vectorizer and fit it with training data
count_f = CountVectorizer(max_features=1000,lowercase=True, token_pattern='[A-Za-z#@_$%]{2,}')
x_train = count_f.fit_transform(X_train)
# create count vectorizer and fit it with test data
test_bow = count_f.transform(X_test)

# model takes the most frequent 1000 words
clf = MultinomialNB()
model = clf.fit(x_train, y_train)
predicted_y = model.predict(test_bow)

for i in range(0,len(test_id)):
    print(str(test_id[i]), end=' ')
    print(predicted_y[i])






