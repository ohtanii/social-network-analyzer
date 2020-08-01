import pandas as pd
import numpy as np
import re
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import models
import pymorphy2
import os.path
import pickle


morph = pymorphy2.MorphAnalyzer()

path = ''
# Объем словаря
max_features = 100000
# Максимальная длина поста
maxlen = 100
batch_size = 32
t = Tokenizer(num_words=max_features, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

# Создание модели нейронной сети
def createModel():
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    return model

# Предобработка текста
def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()

# Считываем данные для обучения
def readTrainData():
    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    data_positive = pd.read_csv('data/positive.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])
    data_negative = pd.read_csv('data/negative.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])

    # Формируем сбалансированный датасет
    sample_size = min(data_positive.shape[0], data_negative.shape[0])
    raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                            data_negative['text'].values[:sample_size]), axis=0)
    labels = [1] * sample_size + [0] * sample_size

    # Предобработка: 
    # приведение к нижнему регистру;
    # замена «ё» на «е»;
    # замена ссылок на токен «URL»;
    # замена упоминания пользователя на токен «USER»;
    # удаление знаков пунктуации.
    data = [preprocess_text(t) for t in raw_data]

    df_train=pd.DataFrame(columns=['Text', 'Label'])
    df_test=pd.DataFrame(columns=['Text', 'Label'])

    df_train['Text'], df_test['Text'], df_train['Label'], df_test['Label'] = train_test_split(data, labels, test_size=0.2, random_state=1)

    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=1)

    # Разделение на данные и значения тональности
    x_df_train = df_train.iloc[:,0]
    y_df_train = df_train.iloc[:,1]
    x_df_val = df_val.iloc[:,0]
    y_df_val = df_val.iloc[:,1]

    # Преобразование в numpy array
    x_train = x_df_train.to_numpy()
    x_val = x_df_val.to_numpy()
    y_train = y_df_train.to_numpy()
    y_val = y_df_val.to_numpy()

    new_data = []
    for line in x_train:
        line = [morph.parse(word)[0].normal_form for word in line.split()]
        line = str(line)
        line = preprocess_text(line)
        new_data.append(line)
    x_train = np.array(new_data)
    new_data = []
    for line in x_val:
        line = [morph.parse(word)[0].normal_form for word in line.split()]
        line = str(line)
        line = preprocess_text(line)
        new_data.append(line)
    x_val = np.array(new_data)

    return x_train, y_train, x_val, y_val

# Обучение модели
def trainModel():
    model = createModel()
    epoch_num = 10
    x_train, y_train, x_val, y_val = readTrainData()
    t.fit_on_texts(x_train)
    x_train = t.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    x_val = t.texts_to_sequences(x_val)
    x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
    model.save_weights('./checkpoint/cp4')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1)
    checkpoint = ModelCheckpoint('./checkpoint/cp4', monitor='val_loss', verbose=1, save_best_only=True)
    callbacks = [early_stop, checkpoint]
    model.fit(x_train, y_train, epochs=epoch_num, validation_data=(x_val, y_val), callbacks=callbacks)

    model.save_weights('text_weights/weight4')
    model.save('text_models/model4.h5')

# Считываем данные для предсказывания
def readPredictData(foldername):
    n = ['description', 'likes']
    data = pd.read_csv(foldername+'/posts.csv', sep='",', error_bad_lines=False, names=n, usecols=['description'])
    data = data.iloc[1:,0]
    data = data.to_numpy()
    new_data = []
    for line in data:
        line = preprocess_text(line)
        line = [morph.parse(word)[0].normal_form for word in line.split()]
        line = str(line)
        line = preprocess_text(line)
        new_data.append(line)
    data = np.array(new_data)
    return data

def sentimentAn(foldername):
    if (os.path.exists(foldername+'/posts.csv')):
        negative = 0
        positive = 0
        data = readPredictData(foldername)
        print(data)

        with open('tokenizer.pickle', 'rb') as handle:
            t = pickle.load(handle)
        
        data = t.texts_to_sequences(data)
        data = sequence.pad_sequences(data, maxlen=maxlen)
        new_model = models.load_model('text_models/model4.h5')
        predictions = new_model.predict(data)
        
        for prediction in predictions:
            if (prediction[0]>=0.5):
                positive+=1
            else:
                negative+=1
        
        num = positive+negative
        positive = (positive/num) *100
        negative = (negative/num) *100

        print ("Анализ тональности постов:")
        print ("Позитивная тональность:     ", round(positive, 2),"%")
        print ("Негативная тональность:     ", round(negative,2),"%")
