import os
import numpy as np 
from collections import Counter 
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary

def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
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

# Prepare feature vectors per training mail and its labels
train_labels = np.zeros(740)
train_labels[640:740] = 1
train_matrix = extract_features(train_dir)

# training models
model1 = MultinomialNB()
model2 = BernoulliNB()
model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)

# Test the unseen mails for Spam
test_dir = 'train-set'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(580)
test_labels[480:580] = 1
result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)

print(classification_report(test_labels, result1))
print()
print("Confusion Matrix: \n", confusion_matrix(test_labels, result1))
print("Accuracy: \n", accuracy_score(test_labels, result1))
print("------------------------------------------------------")
print(classification_report(test_labels, result2))
print()
print("Confusion Matrix: \n", confusion_matrix(test_labels, result2))
print("Accuracy: \n", accuracy_score(test_labels, result2))