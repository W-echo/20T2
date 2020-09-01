from sklearn.tree import DecisionTreeClassifier
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize

# train_file = sys.argv[1]
# test_file = sys.argv[2]

url_reg = r'[a-z]*[:.]+\S+'  # urls in text
invalid_symbol = "#,@_$%."  # invalid symbol in text


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
        # keeping only <space>, a-z, A-Z, 0-9, and #@$%_
        comp = re.compile('[^A-Z^a-z^0-9^#@$%_^ ]')
        return comp.sub('', text)

    # preprocessing by removing ?junk? characters

    table = str.maketrans('', '', invalid_symbol)
    for i in range(0, len(data)):
        # remove invalid symbols
        data[i][1] = data[i][1].translate(table)
        # remove URLS
        data[i][1] = re.sub(url_reg, '', data[i][1])
        data[i][1] = clean_en_text(data[i][1])
    return data


# Create text
f = open('dataset.tsv', 'r')
train_data = read_data(f)

f = open('dataset.tsv', 'r')
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
feature = int(0.01*len(X_train))
regex = '[A-Za-z0-9#@_$%]{2,}'
count = CountVectorizer(max_features=1000, lowercase=False,token_pattern=regex)
train_bow = count.fit_transform(X_train)
test_bow = count.transform(X_test)
# print()
# train the model
clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=feature, random_state=0)
model = clf.fit(train_bow, y_train)
predicted_y = model.predict(test_bow)

f = open("output.txt", 'a')
for i in range(0, len(test_id)):
    f.write(str(test_id[i]))
    f.write(' ')
    f.write(predicted_y[i])
    f.write('\n')
f.close()




