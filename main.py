import os
import numpy as np 
from collections import Counter 
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from visualization import visualize_conf_matr

# get dictionary of most common words
def make_Dictionary(train_dir):
    # read mails
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    # remove not needed words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    # get the most 3000 common words
    dictionary = dictionary.most_common(3000)
    return dictionary

# extract futures
def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    # get word repetition count
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix

# Create a dictionary of words with its frequency
train_dir = 'mail-set'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors
train_y = np.zeros(740)
train_y[457:740] = 1
train_x = extract_features(train_dir)

# training model
model1 = MultinomialNB()
model1.fit(train_x,train_y)

# Test the unseen mails for Spam
test_dir = 'train-set'
test_x = extract_features(test_dir)
train_y = np.zeros(646)
train_y[483:646] = 1
result1 = model1.predict(test_x)

# console graph
print("------------------------------------------------------")
print(classification_report(train_y, result1))
print()
print("------------------------------------------------------")
print("Confusion Matrix: \n", confusion_matrix(train_y, result1))
print("------------------------------------------------------")
print("Accuracy: \n", accuracy_score(train_y, result1))
print("------------------------------------------------------")

# visualize confusion matrix
visualize_conf_matr(confusion_matrix(train_y, result1))