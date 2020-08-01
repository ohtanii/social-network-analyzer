import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout


labels = ['Злость', 'Отвращение', 'Страх', 'Радость', 'Грусть', 'Удивление','Нейтральное']
# Чтение данных из файла
def read_data(filename):
    df = pd.read_csv(filename)
    df.head()
    df_train = df[df['Usage']=='Training']
    df_valid = df[df['Usage']=='PublicTest']
    df_test = df[df['Usage']=='PrivateTest']

    x_df_train = df_train.iloc[:,1]
    y_df_train = df_train.iloc[:,0]
    x_df_valid = df_valid.iloc[:,1]
    y_df_valid = df_valid.iloc[:,0]
    x_df_test = df_test.iloc[:,1]
    y_df_test = df_test.iloc[:,0]

    x_train = x_df_train.to_list()
    y_train = y_df_train.to_numpy()
    x_valid = x_df_valid.to_list()
    y_valid = y_df_valid.to_numpy()
    x_test = x_df_test.to_list()
    y_test = y_df_test.to_numpy()

    for i in range(len(x_train)):
        x_train[i] = list(x_train[i].split(' '))
    for i in range(len(x_train)):
        x_train[i] = [int(j) for  j in x_train[i]]
    
    for i in range(len(x_valid)):
        x_valid[i] = list(x_valid[i].split(' '))
    for i in range(len(x_valid)):
        x_valid[i] = [int(j) for  j in x_valid[i]]

    for i in range(len(x_test)):
        x_test[i] = list(x_test[i].split(' '))
    for i in range(len(x_test)):
        x_test[i] = [int(j) for  j in x_test[i]]

    x_train = np.array(x_train)
    x_valid = np.array(x_valid)
    x_test = np.array(x_test)

    x_train = x_train/255
    x_train = np.reshape(x_train, (len(x_train), 48, 48, 1))
    y_train = utils.to_categorical(y_train, len(labels))
    
    x_valid=x_valid/255
    x_valid = np.reshape(x_valid, (len(x_valid), 48, 48, 1))
    y_valid = utils.to_categorical(y_valid, len(labels))
    
    x_test=x_test/255
    x_test = np.reshape(x_test, (len(x_test), 48, 48, 1))
    y_test = utils.to_categorical(y_test, len(labels))

    return x_train, y_train, x_valid, y_valid, x_test, y_test

# Отображение лица и определенной эмоции
def show(img, emotion):
    plt.imshow(img[:,:,0], cmap='gray',interpolation='none')
    plt.title('Классифицирована эмоция: '+emotion)
    plt.show()

# Создание архитектуры нейронной сети
def createModel():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=3,input_shape=(48,48,1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(32,kernel_size=3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64,kernel_size=3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0,5))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

# Классифицирование эмоций
def predict_emotion(imgs):
    emotions = [0] * len(labels)
    new_model = models.load_model('img_models/model14.h5')
    predictions = new_model.predict(imgs)
    for i in range(len(predictions)):
        emotion = np.argmax(predictions[i])
        #show(imgs[i], labels[emotion])
        emotions[emotion] += 1
    return emotions
