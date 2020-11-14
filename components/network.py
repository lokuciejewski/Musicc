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
        self.model.add(layers.Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    def convert_to_feature(data_list, feature):
        result = []
        for element in data_list:
            result.append((feature(y=element[0]), element[1]))
        return result

    @staticmethod
    def prepare_dataset(positives, negatives, train_to_test_ratio=0.7):
        temp = list(positives)
        temp.extend(negatives)
        np.random.shuffle(temp)
        train_size = int(len(temp) * train_to_test_ratio)
        train = temp[:train_size]
        test = temp[train_size:]

        return train, test

    @staticmethod
    def get_positives(number_of_samples=1000, path='data/positive/', feature=None, verbose=True):
        positives = []

        path = os.path.join(os.path.pardir, path)
        needs_adjusting = False
        min_len = math.inf

        for i in range(1, number_of_samples + 1):
            filepath = os.path.join(path, f'pos ({i}).mp3')
            try:
                x, sr = librosa.load(filepath)
                positives.append((x, 1))
                if len(x) < min_len:
                    min_len = len(x)
                    needs_adjusting = True
                if verbose:
                    print(f'Finished positive example {i}/{number_of_samples}', end='\r')
            except RuntimeError:
                print(f'File: {filepath} is corrupted or contains data in an unknown format!')
                i = i - 1
                continue

        if needs_adjusting:
            positives = Helper.trim_data(positives, min_len)
        random.shuffle(positives)
        print(f'Length of the specimen should be {len(positives[0][0])}')
        if feature is not None:
            positives = Helper.convert_to_feature(positives, feature)
        return positives

    @staticmethod
    def get_negatives(number_of_samples=1000, sample_length=None, path='data/negative/', feature=None,
                      verbose=True):
        negatives = []

        path = os.path.join(os.path.pardir, path)
        needs_adjusting = False
        min_len = math.inf

        file_count = len(next(os.walk(os.path.join(os.path.pardir, 'data/negative/')))[2])
        if verbose:
            print(f'Number of all negative samples: {file_count}')
        for i in range(1, number_of_samples + 1):
            filepath = os.path.join(path, f'neg_{random.randint(1, file_count)}.wav')
            try:
                x, sr = librosa.load(filepath)
                negatives.append((x, 0))
                if len(x) < min_len:
                    min_len = len(x)
                    needs_adjusting = True
                if verbose:
                    print(f'Finished negative example {i}/{number_of_samples}', end='\r')
            except RuntimeError:
                print(f'File: {filepath} is corrupted or contains data in an unknown format!')
                i = i - 1
                continue

        if needs_adjusting:
            negatives = Helper.trim_data(negatives, min_len)
        if sample_length is not None:
            negatives = Helper.trim_data(negatives, sample_length)
        random.shuffle(negatives)
        print(f'Length of the specimen should be {len(negatives[0][0])}')
        if feature is not None:
            negatives = Helper.convert_to_feature(negatives, feature)
        return negatives

    @staticmethod
    def get_x_y(data):
        x, y = [], []
        for element in data:
            x.append(element[0])
            y.append(element[1])
        x = np.array(x)
        y = np.array(y)
        return x, y
