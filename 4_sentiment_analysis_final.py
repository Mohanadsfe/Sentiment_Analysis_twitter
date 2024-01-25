# -*- coding: utf-8 -*-
"""

# Emoji Sentiment Analysis with Tweets_Eng
        
## step4-Sentiment analysis

#### 4.1 Constructing train and test dataset
- data cleaning
- preparing new columns for 4.3 and 4.4
- split the dataset

#### 4.2 Classification without emojis
- LSTM
- DNN

#### 4.3 Classification with replacing the emojis with their descriptive names
- LSTM
- DNN

#### 4.4 Classification with replacing the emojis with the most similar 5 text tokens in Word2Vec model
- LSTM
- DNN

#### 4.5 Classification with emojis word embedding vectors in Word2Vec model
- LSTM
- DNN


### 4.1 read the dataset with sentiment labels
"""



import pandas as pd
import os
os.getcwd()
df = pd.read_csv('emoji_tweets_tagged_data.csv')
df

df.iloc[:-20,1:3]


# Commented out IPython magic to ensure Python compatibility.
## data cleaning
#i. lemmatization

from textblob import TextBlob

def lemm(text):
    textTB = TextBlob(text)
    words = textTB.words
    words_lemmatized = words.lemmatize()
    return ' '.join(words_lemmatized)


# %time df['tweets']=df['tweets'].apply(lambda x : lemm(x))

#ii. keep only english characters and emojis

import emoji
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())

def keepengemoji(text):
    ls = []
    for w in text.split(' '):
        if w in words:
            ls.append(w)
        elif w in emoji.EMOJI_DATA.keys():
            w = ' '+w+' '
            ls.append(w)
        else:
            continue
    return ' '.join(ls)


df['tweets']=df['tweets'].apply(lambda x : keepengemoji(x))

import emoji

def remove_emojis(text):
  return ''.join(c for c in text if c not in emoji.EMOJI_DATA.keys())



puretext = [remove_emojis(t).strip() for t in df['tweets']]

df['text']=puretext

# new column, replace emoji with its name, for 4.3
import demoji
wtemo = [demoji.replace_with_desc(df.tweets[i]).replace(':','') for i in range(len(df))]
df['wtemo'] = wtemo



# new column, replace emoji with its most similar text tokens, for 4.4
emosimi_df = pd.read_csv("en_most_similar_names.csv")  # joined str of 5 most similar text tokens of emojis



import emoji
ls = []
for i in df['tweets']:
    wls = []
    words = i.split()
    for word in words:
        if word not in emoji.EMOJI_DATA.keys():
            wls.append(word)
        elif word in emosimi_df['Unnamed: 0'].to_list():
            wls.append(emosimi_df.loc[emosimi_df['Unnamed: 0'] == word, '0'].iloc[0])
        else:
            wls.append(demoji.replace_with_desc(word).replace(':',''))
    simi = ' '.join(wls)
    ls.append(simi)



df['simiemo'] = ls







# split dataset
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)

y_train = train['labels']
y_test = test['labels']

print(len(y_train),len(y_test))



"""### 4.2 Classification without emojis
using text as x
- LSTM: 0.57
- DNN: 0.56
"""

# classification with pure text
x_train42 = train['text'].to_list()
x_test42 = test['text'].to_list()

print(len(x_train42),len(x_test42))

########################################################################

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the dataset
df = pd.read_csv('emoji_tweets_tagged_data.csv')

# Data cleaning and preparation
# (Your existing data cleaning and preparation code goes here)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['tweets'], df['labels'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SVM model training
svm_model = SVC(kernel='linear')  # Linear kernel for simplicity, you can try other kernels as well
svm_model.fit(X_train_tfidf, y_train)

# SVM model evaluation
svm_predictions = svm_model.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Test Accuracy:", svm_accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, svm_predictions)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


########################################################################




# encode the words
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print('Loading data...')
def get_sequences(tokenizer, tweets):
  sequences = tokenizer.texts_to_sequences(tweets)
  padded = pad_sequences(sequences, truncating ='post', maxlen = 40)
  return padded


# tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(x_train42)
print(tokenizer.texts_to_sequences([x_train42[0]]))
x_train42_seq = get_sequences(tokenizer, x_train42)

tokenizer.fit_on_texts(x_test42)
print(tokenizer.texts_to_sequences([x_test42[0]]))
x_test42_seq = get_sequences(tokenizer, x_test42)

print('x_train shape:', x_train42_seq.shape)
print('x_train shape:', x_test42_seq.shape)

get_sequences(tokenizer, x_train42[:2])

# set parameters
max_features = 60000 # cut texts after this number of words (among top max_features most common words)
embedding_dims = 300
maxlen = 40 #based on 2, most tweets are shorter than 40 words

# LSTM model
print('Build LSTM model...')

model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000, 300, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(6, activation= 'softmax')
])

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = ['accuracy']
)


# run lstm model
# the model, with training set, validation set
h = model.fit(
    x_train42_seq, y_train,
    validation_data=( x_test42_seq, y_test,),
    epochs=10,
    callbacks=[
               tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
)

# testing model
score, accuracy = model.evaluate(x_test42_seq, y_test)
print('Test accuracy: {}, Test loss: {}'.format(accuracy, score))



# DNN model

# let's try a more complicated DNN since the dimension is high


print('Build DNN model...')
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# config model
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training model
model.fit(x_train42_seq, y_train, batch_size=64, epochs=20, validation_data=(x_test42_seq, y_test))

# testing DNN model
score, accuracy = model.evaluate(x_test42_seq, y_test)
print('Test accuracy: {}, Test loss: {}'.format(accuracy, score))






########################################################################



"""### 4.3 Classification with replacing the emojis with their descriptive names

using the wtemo column as x
- LSTM: 0.77
- DNN: 0.78
"""

# classification with emojis' descriptive names
x_train43 = train['wtemo'].to_list()
x_test43 = test['wtemo'].to_list()

print(len(x_train43),len(x_test43))

# encode the words
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(x_train43)
print(tokenizer.texts_to_sequences([x_train43[0]]))
x_train43_seq = get_sequences(tokenizer, x_train43)

tokenizer.fit_on_texts(x_test43)
print(tokenizer.texts_to_sequences([x_test43[0]]))
x_test43_seq = get_sequences(tokenizer, x_test43)

print('Loading data...')
print('x_train shape:', x_train43_seq.shape)
print('x_train shape:', x_test43_seq.shape)
get_sequences(tokenizer, x_train43[:2])

# LSTM model
print('Build LSTM model...')

model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000, 300, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(6, activation= 'softmax')
])

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = ['accuracy']
)


# run lstm model
# the model, with training set, validation set
h = model.fit(
    x_train43_seq, y_train,
    validation_data=( x_test43_seq, y_test,),
    epochs=10,
    callbacks=[
               tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
)

# testing model
score, accuracy = model.evaluate(x_test43_seq, y_test)
print('Test accuracy: {}, Test loss: {}'.format(accuracy, score))

# let's try a more complicated DNN since the dimension is high


print('Build DNN model...')
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# config model
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training model
model.fit(x_train43_seq, y_train, batch_size=64, epochs=20, validation_data=(x_test43_seq, y_test))

# testing DNN model
score, accuracy = model.evaluate(x_test43_seq, y_test)
print('Test accuracy: {}, Test loss: {}'.format(accuracy, score))


########################################################################

"""### 4.4 Classification with replacing the emojis with the most similar 5 text tokens

using the simiemo column as x

if the emoji is not shown in our trained Word2Vec model, than replacing it with its descriptive name as 4.3

- LSTM: 0.76
- DNN: 0.75
"""

# classification with emojis' descriptive names
x_train44 = train['simiemo'].to_list()
x_test44 = test['simiemo'].to_list()

print(len(x_train44),len(x_test44))

# encode the words
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(x_train44)
print(tokenizer.texts_to_sequences([x_train44[0]]))
x_train44_seq = get_sequences(tokenizer, x_train44)

tokenizer.fit_on_texts(x_test44)
print(tokenizer.texts_to_sequences([x_test44[0]]))
x_test44_seq = get_sequences(tokenizer, x_test44)

print('Loading data...')
print('x_train shape:', x_train44_seq.shape)
print('x_train shape:', x_test44_seq.shape)
get_sequences(tokenizer, x_train44[:2])

# LSTM model
print('Build LSTM model...')

model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000, 300, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(6, activation= 'softmax')
])

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = ['accuracy']
)


# run lstm model
# the model, with training set, validation set
h = model.fit(
    x_train44_seq, y_train,
    validation_data=( x_test44_seq, y_test,),
    epochs=10,
    callbacks=[
               tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
)

# testing model
score, accuracy = model.evaluate(x_test44_seq, y_test)
print('Test accuracy: {}, Test loss: {}'.format(accuracy, score))

# DNN model

print('Build DNN model...')
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# config model
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training model
model.fit(x_train44_seq, y_train, batch_size=64, epochs=20, validation_data=(x_test44_seq, y_test))

# testing DNN model
score, accuracy = model.evaluate(x_test44_seq, y_test)
print('Test accuracy: {}, Test loss: {}'.format(accuracy, score))




########################################################################


"""### 4.5 Classification with emojis word embedding vectors
using original tweets & pre-trained model

* the result is not very satisfying because many tokens are droped due to key missing in pre-trained model

- LSTM: 0.74
- DNN: 0.74
"""
from gensim.models import KeyedVectors 
#import the trianed Word2Vec model
# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("en_word2vec.wordvectors", mmap='r')


#example
word_index = ['😂']

# drop the words if not in word_index
wls = []
for i in train['tweets']:
    ls = []
    words = i.split(' ')
    for w in words:
        try:
            word_index[w]
            ls.append(w)
        except:
            continue
    wls.append(' '.join(ls))

wls2 = []
for i in test['tweets']:
    ls = []
    words = i.split(' ')
    for w in words:
        try:
            word_index[w]
            ls.append(w)
        except:
            continue
    wls2.append(' '.join(ls))

x_train45 = wls
x_test45 = wls2

# encode the words
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print('Loading data...')
def get_sequences(tokenizer, tweets):
  sequences = tokenizer.texts_to_sequences(tweets)
  padded = pad_sequences(sequences, truncating ='post', maxlen = maxlen)
  return padded

# tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(x_train45)
word_index = tokenizer.word_index #get word_index
print(tokenizer.texts_to_sequences([x_train45[0]]))
x_train45_seq = get_sequences(tokenizer, x_train45)

tokenizer.fit_on_texts(x_test45)
print(tokenizer.texts_to_sequences([x_test45[0]]))
x_test45_seq = get_sequences(tokenizer, x_test45)

print('x_train shape:', x_train45_seq.shape)
print('x_train shape:', x_test45_seq.shape)

padding_type='post'
truncation_type='post'

# set parameters
max_features = 60000 # cut texts after this number of words (among top max_features most common words)
embedding_dims = 300
maxlen = 40 #based on 2, most tweets are shorter than 40 words

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dims))
for word, i in word_index.items():
    try:
        embedding_vector = wv[word]
    except:
        embedding_vector = None
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_matrix.shape

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

embedding_layer = Embedding(input_dim=len(word_index)+1,
                            output_dim= embedding_dims,
                            weights=[embedding_matrix],
                            input_length=40,
                            trainable=False)



# LSTM model
print('Build LSTM model...')


from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import Sequential
model = Sequential([
    embedding_layer,
    Bidirectional(LSTM(20, return_sequences=True)),
    Bidirectional(LSTM(20)),
    Dense(128, activation='relu'),
   Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# testing model
log_folder = 'logs'
callbacks = [
            EarlyStopping(patience = 10),
            TensorBoard(log_dir=log_folder)
            ]
num_epochs = 10
model.fit(x_train45_seq, y_train, epochs=num_epochs, validation_data=(x_test45_seq, y_test),
          callbacks=[
               tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ])


# testing model
score, accuracy = model.evaluate(x_test45_seq, y_test)
print('Test accuracy: {}, Test loss: {}'.format(accuracy, score))

# DNN model

print('Build DNN model...')
model = tf.keras.Sequential()
model.add(embedding_layer)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# config model
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training model
model.fit(x_train45_seq, y_train, batch_size=64, epochs=20, validation_data=(x_test45_seq, y_test),
         callbacks=[
               tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ])

# testing DNN model
score, accuracy = model.evaluate(x_test45_seq, y_test)
print('Test accuracy: {}, Test loss: {}'.format(accuracy, score))