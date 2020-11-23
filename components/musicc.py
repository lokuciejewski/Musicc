import os

import numpy as np

from components.helper import Helper
from components.network import Network
from components.specimen import Evolution


class Musicc:

    def __init__(self, features, number_of_samples, sample_length, force_new_positives, number_of_specimen,
                 initial_mutation_chance, crossover_chance,
                 filepath=os.path.join(os.path.pardir, 'data/train_test_file_small.npz'), load_networks=False):
        self.positive_examples = None
        self.negative_examples = None
        self.load_positive_examples(number_of_samples, force_new_positives, filepath)
        self.load_negative_examples(number_of_samples, sample_length)
        self.train_set = None
        self.test_set = None
        self.load_train_test()
        self.features = features
        self.networks = []
        if load_networks:
            self.load_networks()
        else:
            self.create_networks()
        self.evolutionary_algorithm = Evolution(neural_networks=self.networks, number_of_specimen=number_of_specimen,
                                                specimen_length=sample_length, mutation_chance=initial_mutation_chance,
                                                crossover_chance=crossover_chance,
                                                features_list=list(zip(*self.features))[0])

    def load_positive_examples(self, number_of_samples, force_new_samples,
                               filepath=os.path.join(os.path.pardir, 'data/train_test_file_small.npz')):
        if force_new_samples:
            self.positive_examples = Helper.get_positives(number_of_samples)
            np.savez(filepath, positives=self.positive_examples)
        else:
            file = np.load(filepath, allow_pickle=True)
            self.positive_examples = file['positives']

    def load_negative_examples(self, number_of_samples, sample_length):
        self.negative_examples = Helper.get_negatives(number_of_samples, sample_length)

    def load_train_test(self):
        self.train_set, self.test_set = Helper.prepare_dataset(self.positive_examples, self.negative_examples,
                                                               train_to_test_ratio=0.8)

    def load_networks(self):
        self.networks.clear()
        for feature in self.features:
            network = Network(input_shape=feature[1])
            feature_name = str(feature).split(' ')[2]
            network.load_model(filepath=os.path.join(os.path.pardir, f'models/model_{feature_name}'))
            self.networks.append(network)

    def save_networks(self):
        for i, network in enumerate(self.networks):
            feature_name = str(self.features[i]).split(' ')[2]
            network.save_model(os.path.join(os.path.pardir, f'models/model_{feature_name}'))

    def create_networks(self):
        self.networks.clear()
        for feature in self.features:
            network = Network(input_shape=feature[1])
            self.networks.append(network)

    def train_networks(self, epochs, batch_size):
        for network in self.networks:
            network.train(x_train=self.train_set[0], y_train=self.train_set[1], epochs=epochs, batch_size=batch_size)
            network.evaluate(x_test=self.test_set[0], y_test=self.test_set[1])

    def run_evolution(self, number_of_epochs, decreasing_mutation_factor=0.1, save=False, sr=22050,
                      save_as_negative=False, epsilon=0.1):
        self.evolutionary_algorithm.run_epochs(number_of_epochs, decreasing_mutation_factor, save, sr,
                                               save_as_negative, epsilon)
