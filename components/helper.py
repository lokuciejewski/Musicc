import math
import os
import random
import multiprocessing as mp
from multiprocessing import Pool as ThreadPool

import librosa
import numpy as np


class Helper:

    @staticmethod
    def get_data_list_of_all_features(features_list, data_list):
        result = {}
        for i in range(len(features_list)):
            result[features_list[i]] = Helper.get_data_list_of_feature(features_list[i], data_list)
        return result

    @staticmethod
    def get_data_list_of_feature(feature, data_list):
        print(f'Started feature: {feature}')
        x, y = list(zip(*data_list))
        pool = ThreadPool(mp.cpu_count())
        result = pool.map(feature, x)
        print(f'Finished feature: {feature}')
        return list(zip(result, y))

    @staticmethod
    def trim_data(data, target_length):
        for i in range(len(data)):
            if len(data[i][0]) > target_length:
                data[i] = (data[i][0][:target_length], data[i][1])
        return data

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
            positives = Helper.get_data_list_of_feature(feature, positives)
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
            negatives = Helper.get_data_list_of_feature(feature, negatives)
        return negatives
