import math
import os
import random

import librosa
import numpy as np

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras import layers, models


class Network:
    def __init__(self, input_shape):
        self.model = models.Sequential()
        self.model.add(layers.Dense(input_shape=input_shape, units=128, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(2, activation='exponential'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['loss', 'accuracy'])
        self.history = None

    def train(self, x_train, y_train, epochs=100, batch_size=100):
        self.history = self.model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x_test, y_test):
        loss, acc = self.model.evaluate(x_test, y_test)
        print(f'Loss: {loss}, Accuracy: {acc}')
        return loss, acc

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = models.load_model(filepath)

    def predict(self, audio_data):
        return self.model.predict(audio_data, verbose=0)


class Helper:

    @staticmethod
    def trim_data(data, target_length):
        for i in range(len(data)):
            if len(data[i][0]) > target_length:
                data[i] = (data[i][0][:target_length], data[i][1])
        return data

    @staticmethod
    def convert_to_mel_spectrograms(data_list):
        result = []
        for element in data_list:
            result.append((librosa.feature.melspectrogram(y=element[0]), element[1]))
        return result

    @staticmethod
    def prepare_dataset(size=1000, pos_to_neg_ratio=0.5, train_to_test_ratio=0.7, path='data/', mel_spectrogram=True,
                        verbose=True):
        temp = []
        pos_size = int(size * pos_to_neg_ratio)
        neg_size = size - pos_size

        min_len = math.inf
        needs_adjusting = False

        path = os.path.join(os.path.pardir, path)

        for i in range(pos_size):
            filepath = os.path.join(path, f'positive/pos_{random.randint(1, 1000)}.mp3')
            try:
                x, sr = librosa.load(filepath)
                temp.append((x, [0, 1]))
                if len(x) < min_len:
                    min_len = len(x)
                    needs_adjusting = True
                if verbose:
                    print(f'Finished positive example {i + 1}/{pos_size}')
            except RuntimeError:
                print(f'File: {filepath} is corrupted or contains data in an unknown format!')
                i = i - 1
                continue

        for i in range(neg_size):
            x, sr = librosa.load(os.path.join(path, f'negative/neg_{random.randint(1, 1000)}.mp3'))
            temp.append((x, [1, 0]))
            if len(x) < min_len:
                min_len = len(x)
                needs_adjusting = True
            if verbose:
                print(f'Finished negative example {i + 1}/{neg_size}')

        if needs_adjusting:
            temp = Helper.trim_data(temp, min_len)

        random.shuffle(temp)
        print(f'Length of the specimen should be {len(temp[0][0])}')
        if mel_spectrogram:
            temp = Helper.convert_to_mel_spectrograms(temp)
        train_size = int(len(temp) * train_to_test_ratio)
        train = temp[:train_size]
        test = temp[train_size:]

        return train, test

    @staticmethod
    def get_x_y(data):
        x, y = [], []
        for element in data:
            x.append(element[0])
            y.append(element[1])
        x = np.array(x)
        y = np.array(y)
        return x, y
