import sys
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

train_file = sys.argv[1]
test_file = sys.argv[2]



def predict_and_test(model, X_test_bag_of_words):
    predicted_y = model.predict(X_test_bag_of_words)
    print(predicted_y)
    # print(model.predict_proba(X_test_bag_of_words))
    print(classification_report(y_test, predicted_y))


url_reg = r'[a-z]*[:.]+\S+'     # urls in text
invalid_symbol = "#@_$%"      # invalid symbol in text
pun = string.punctuation
stop_words = set(stopwords.words('english'))


def read_data(file):
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.replace("\n", "")
        data.append(line.split("\t"))
    f.close()

    # preprocessing by removing “junk” characters
    sentiment = []
    table = str.maketrans('', '', invalid_symbol)
    for i in range(0, len(data)):
        # remove invalid symbols
        data[i][1] = data[i][1].translate(table)
        # remove URLS
        data[i][1] = re.sub(url_reg, '', data[i][1])
        # words = data[i][1].split(" ")
        sentiment.append(data[i][-1])
    # print(data[9])

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
# split = int(len(X)*0.8)
X_train = train_data[:, 1]
y_train = train_data[:, -1]
test_id = test_data[:, 0]
X_test = test_data[:, 1]
y_test = test_data[:, -1]


# create count vectorizer and fit it with training data
# the most frequent 1000 words
# count_f = CountVectorizer(max_features=1000)
# x_train_bow_f = count_f.fit_transform(X_train)
# # count_test_f = CountVectorizer(max_features=1000)
# x_test_bow_f = count_f.transform(X_test)

# # model takes the most frequent 1000 words
# clf = BernoulliNB()
# train_model_f = clf.fit(x_train_bow_f, y_train)
# predict_and_test(train_model_f, x_test_bow_f)


# all words considered
count = CountVectorizer(lowercase=False, token_pattern='[A-Za-z0-9#@_$%]{2,}')
x_train_bow = count.fit_transform(X_train)
# count_test = CountVectorizer()
test_bow = count.transform(X_test)

# model takes all words considered
clf = BernoulliNB()
model = clf.fit(x_train_bow, y_train)

predicted_y = model.predict(test_bow)

f = open("output.txt", 'a')
for i in range(0,len(test_id)):
    f.write(str(test_id[i]))
    f.write(' ')
    f.write(predicted_y[i])
    f.write('\n')
f.close()

