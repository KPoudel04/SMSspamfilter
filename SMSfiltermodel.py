# All the imports
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import pickle
import streamlit as sl
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout

# Download the UCI datasets
url = 'https://raw.githubusercontent.com/ShresthaSudip/SMS_Spam_Detection_DNN_LSTM_BiLSTM/master/SMSSpamCollection'
messages = pd.read_csv(url, sep ='\t',names=["label", "message"])
print(messages[:3])

# Visualize the data
print(messages.describe())

duplicatedRow = messages[messages.duplicated()]
print(duplicatedRow[:5])

# Separate all the ham and spam messages
ham_msg = messages[messages.label == 'ham']
spam_msg = messages[messages.label == 'spam']
# Create numpy list
ham_msg_text = " ".join(ham_msg.message.to_numpy().tolist())
spam_msg_text = " ".join(spam_msg.message.to_numpy().tolist())

# wordcloud of ham messages
ham_msg_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS,max_font_size=50, background_color ="black", colormap='Blues').generate(ham_msg_text)
plt.figure(figsize=(16,10))
plt.imshow(ham_msg_cloud, interpolation='bilinear')
plt.axis('off')
#plt.show()
# wordcloud of spam messages
spam_msg_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS,max_font_size=50, background_color ="black", colormap='Blues').generate(spam_msg_text)
plt.figure(figsize=(16,10))
plt.imshow(spam_msg_cloud, interpolation='bilinear')
plt.axis('off') # turn off axis
#plt.show()

# Only 85% of the dataset is not spam, we will use downsampling to fix this
ham_msg_df = ham_msg.sample(n = len(spam_msg), random_state = 44)
spam_msg_df = spam_msg
print(ham_msg_df.shape, spam_msg_df.shape)
# Now, both shapes are equal
msg_df = ham_msg_df.append(spam_msg_df).reset_index(drop=True)
msg_df['text_length'] = msg_df['message'].apply(len)
labels = msg_df.groupby('label').mean()
# print(labels)

# Now, it's time to prep the data

# Map ham label as 0 and spam as 1
msg_df['msg_type']=msg_df['label'].map({'ham':0,'spam':1})
msg_label = msg_df['msg_type'].values

# Split data into training and testing
(train_msg, test_msg, train_labels, test_labels)= train_test_split(msg_df['message'], msg_label, test_size=0.2, random_state=434)

# Now, we first need to tokenize the data

# Define pre-processing hyperparameters
max_len=50
trunc_type="post"
padding_type="post"
oov_tok = "<OOV>"
vocab_size = 500

tokenizer = Tokenizer(num_words=vocab_size,char_level=False,oov_token=oov_tok)
tokenizer.fit_on_texts(train_msg)

# num_words indicates how many unique words to load in training and testing data
# oov_token or out of vocab token will be added to word index to replace out of vocav words
# char_level: If it is "True" then every character will be treated as a token

# Now, every word has it's own token

training_sequences = tokenizer.texts_to_sequences(train_msg)
training_padded = pad_sequences(training_sequences,maxlen=max_len,padding = padding_type, truncating = trunc_type)
testing_sequences = tokenizer.texts_to_sequences(test_msg)
testing_padded = pad_sequences(testing_sequences, maxlen = max_len,
padding = padding_type, truncating = trunc_type)

# padding = pre or post, by using pre we pad before each sequence
# maxlen = maximum length of all sequences, here max_len = 50
# truncating = 'pre' or 'post'If a sequence length is larger than the provided maxlen value then, these
# values will be truncated to maxlen. ‘pre’ option will truncate at the beginning where as ‘post’
# will truncate at the end of the sequences.

# With our data loaded and preprocessed, we can now train

# Dense Spam Detection Model

# Define hyper-parameters

vocab_size = 500
embeding_dim = 24
drop_value = 0.2
n_dense = 24

# Dense model architecture
model = Sequential() # calls for Keras sequential model in which layers are added in sequence (L to R)
model.add(Embedding(vocab_size, embeding_dim, input_length=max_len)) # maps each word to a N-dimensional vector of
# real numbers, the embeding_dim is the size of this vector which is 24
model.add(GlobalAveragePooling1D()) # Pooling layer helps to reduce the numbers of parameters in the model
# Helps to avoid over fitting
# We then use a dense layer with activation function 'relu' followed by a dropout layer to avoid overfitting and a final
# output layer with sigmoid activation function, since there are only 2 classes to classify, we use only a single output
# neuron
model.add(Dense(24,activation='relu'))
model.add(Dropout(drop_value))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
#Now, we fit the model
numEpochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(training_padded, train_labels, epochs=numEpochs, validation_data=(
    testing_padded, test_labels),callbacks =[early_stop], verbose=2)
def predict_spam(predict_msg):
    new_seq = tokenizer.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    return (model.predict(padded))

#model.save('SMSspamfilter.h5')
sl.header("Developed by Kiran Poudel")
text = sl.text_input("Enter Something", "")
Result_text = "Please type something in to test the spam filter"
Result = sl.markdown(f'<h1 style="color:#07080a;font-size:24px;">{(Result_text)}</h1>',
                     unsafe_allow_html=True)
explain = "First of all, the dataset used to train was from UCI datasets which already was pre-labeled as spam or ham. The model used was Dense Text Classifier and has an accuracy rate of about 98%. First, there was an imbalance of data with not spam(ham) messages being 85% of the messages and spam only being 15%. I used downsampling to fix this problem and get the distribution of not spam(ham) and spam messages to be equal. I then had to tokenize the data at the word level and then convert all words into an integer index adjusting the hyperparameters when needed I then had to pad the data so all the sequences could have equal length. Then, with my data preprocessed, I used a Dense architecture(adjusting the hyper parameters for better accuracy) and fit the model. The model results in a training loss of 0.08, accuracy of 98%, validation loss of 0.13 and validation accuracy of 94%. This image from 'Spam filtering in SMS using recurrent neural networks by R. Taheri and R. Javidan' explains it fairly well."
if sl.button('Compile'):
    prediction = predict_spam({text})
    if(prediction >= 0.51):
        Result_text = "This is SPAM"
        Result = sl.markdown(f'<h1 style="color:#a30000;font-size:24px;">{"This is SPAM"}</h1>',
                     unsafe_allow_html=True)
    if (prediction < 0.51):
        Result = sl.markdown(f'<h1 style="color:#0ea303;font-size:24px;">{"This is NOT SPAM"}</h1>',
                             unsafe_allow_html=True)

sl.markdown(f'<text style="font-family:IBM Plex Sans,color:#07080a;font-size:16px;">{"So, how does it work?"}</text>',
                unsafe_allow_html=True)
with sl.container():
    sl.markdown(f'<text style="font-family:IBM Plex Sans,color:#07080a;font-size:16px;">{(explain)}</text>',
                unsafe_allow_html=True)
    response = requests.get('https://d3i71xaburhd42.cloudfront.net/3cc554098a87c113e447ecc4fb14ead650d503bb/3-Figure2-1.png')
    img = Image.open(BytesIO(response.content))
    sl.image(img)




