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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
# nltk.download('vader_lexicon')


# train_file = sys.argv[1]
# test_file = sys.argv[2]

url_reg = r'[a-z]*[:.]+\S+'     # urls in text
invalid_symbol = "#,@_$%."      # invalid symbol in text
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

    # preprocessing by removing “junk” characters
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
        # porter.stem(w).lower()
        data[i][1] = filtered_sentence

    return data

# Create text
f = open('dataset.tsv','r')
train_data = read_data(f)

# f = open(test_file,'r')
# test_data = read_data(f)

# split into train and test
train_data = np.array(train_data)
# test_data = np.array(test_data)

# p = 0.8
split = int(len(train_data)*0.8)
X_train = train_data[:split, 1]
y_train = train_data[:split, -1]
# test_id = test_data[:, 0]
X_test = train_data[split:, 1]
y_test = train_data[split:, -1]


# create count vectorizer and fit it with training data
# the most frequent 1000 words
count = CountVectorizer(max_features=1000,lowercase=False, token_pattern='[A-Za-z0-9#@_$%]{2,}')
train_bow = count.fit_transform(X_train)
test_bow = count.transform(X_test)
# train the model
clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, random_state=0)
model = clf.fit(train_bow, y_train)
predict_and_test(model, test_bow)

# f = open("output.txt", 'a')
# for i in range(0,len(test_id)):
#     f.write(str(test_id[i]))
#     f.write(' ')
#     f.write(predicted_y[i])
#     f.write('\n')
# f.close()

# # analyse with VADER
# y_predict = []
# analyser = SentimentIntensityAnalyzer()
# for text in X:
#     score = analyser.polarity_scores(text)
#     # print(score)
#     if score['compound'] >= 0.05:
#         y_predict.append('positive')
#         # print(text+": "+"VADER positive")
#     elif score['compound'] <= -0.05:
#         y_predict.append('negative')
#         # print(text+": "+"VADER negative")
#     else:
#         y_predict.append('neutral')
#         # print(text+": "+"VADER neutral")
# print(classification_report(Y, y_predict))

# # descriptive statistics showing the frequency distribution for the sentiment classes
# plt.figure(figsize=(6,9)) #调节图形大小
# labels = [u'positive',u'negative',u'neutral'] #定义标签
# dic = {'positive':0,'negative':0,'neutral':0}
# for i in range(0,len(sentiment)):
#     dic[sentiment[i]] = dic[sentiment[i]] +1
# sizes = [dic['positive'], dic['negative'], dic['neutral']] #每块值
# colors = ['lemonchiffon', 'lightblue','plum'] #每块颜色定义
# #patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
# patches,text1,text2 = plt.pie(sizes,
#                       labels=labels,
#                       colors=colors,
#                       labeldistance = 1.2,#图例距圆心半径倍距离
#                       autopct = '%3.2f%%', #数值保留固定小数位
#                       shadow = False, #无阴影设置
#                       startangle =90, #逆时针起始角度设置
#                       pctdistance = 0.6) #数值距圆心半径倍数距离
#
# # show the sentiment class distribution
# plt.axis('equal')   # 'equal' makes sure the figure as circle
# plt.legend()
# plt.show()

