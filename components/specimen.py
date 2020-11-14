import os
import random

import librosa
import numpy as np
import soundfile as sf


class Specimen:
    length = 0
    features = None
    fitness = 0.0

    def __init__(self, length=None, features=None):
        if features is None and length is not None:
            self.length = length
            self.features = self.generate_features()
        else:
            self.features = features
            self.length = len(features)
        self.fitness = 100000

    def generate_features(self):
        return np.random.rand(self.length) * np.random.choice([-1, 1], self.length)

    def mutate(self, mutation_chance=0.001):
        mu = np.mean(self.features)
        sigma = np.std(self.features)
        for i, row in enumerate(self.features):
            if mutation_chance > random.random():
                self.features[i] += random.gauss(mu=mu, sigma=sigma)

    def crossover(self, partner, cuts=100):
        cuts_ind = []
        for i in range(cuts):
            x = random.randint(1, self.features.shape[0] - 1)
            while x in cuts_ind:
                x = random.randint(1, self.features.shape[0] - 1)
            cuts_ind.append(x)
        cuts_ind.sort()
        features_1 = np.copy(self.features)
        features_2 = np.copy(partner.features)
        last_cut = 0
        for cut in cuts_ind:
            temp = features_1[last_cut:cut]
            features_1[last_cut:cut] = features_2[last_cut:cut]
            features_2[last_cut:cut] = temp
            last_cut = cut
        return Specimen(features=features_1), Specimen(features=features_2)

    def set_fitness(self, fitness):
        self.fitness = fitness


class Evolution:
    specimens = []
    mutation_chance = 0.0
    crossover_chance = 0.0
    target = None

    def __init__(self, neural_networks, number_of_specimen, specimen_length, target, mutation_chance, crossover_chance,
                 features_list):
        self.neural_networks = neural_networks
        self.mutation_chance = mutation_chance
        self.crossover_chance = crossover_chance
        self.target = target
        self.fitnesses = [0 for i in range(number_of_specimen)]
        self.features_list = features_list
        for i in range(number_of_specimen):
            specimen = Specimen(specimen_length)
            self.specimens.append(specimen)
            print(f'Generated {i} specimen of {number_of_specimen}', end='\r')
        self.calculate_all_fitnesses()
        for i in range(number_of_specimen):
            self.specimens[i].set_fitness(self.fitnesses[i])
        self.specimens.sort(key=lambda x: -x.fitness)

    def calculate_all_fitnesses(self):
        features = [[] for feature in self.features_list]
        for specimen in self.specimens:
            for i, feature in enumerate(self.features_list):
                features[i].append(feature(specimen.features))
        print('Started fitness calculation')
        temp = []
        for i, network in enumerate(self.neural_networks):
            temp.append(network.predict(np.array(features[i])))
        for specimen in range(len(self.specimens)):
            self.fitnesses[specimen] = np.avg([feature[specimen] for feature in temp])
        print('Finished fitness calculation')

    def select_n_best(self, n):
        return self.specimens[:n]

    def select_random(self):
        return self.specimens[random.randint(0, len(self.specimens) - 1)]

    def run_epochs(self, number_of_epochs, decreasing_mutation_factor=0.1, save=False, sr=22050, save_as_negative=False,
                   epsilon=0.1):
        best = -np.math.inf
        specimen_count = len(self.specimens)
        for i in range(1, number_of_epochs + 1):
            print(f'Started epoch {i}')
            n = 2 * int(len(self.specimens) / 5)
            new_specimens = self.select_n_best(n)
            random.shuffle(new_specimens)
            while len(new_specimens) != specimen_count:
                print(f'Started specimen {len(new_specimens)}/{specimen_count}', end='\r')
                father = self.select_random()
                mother = self.select_random()
                if random.random() < self.crossover_chance:
                    kid_1, kid_2 = father.crossover(mother)
                    kid_1.mutate(mutation_chance=self.mutation_chance*(i**-decreasing_mutation_factor))
                    kid_2.mutate(mutation_chance=self.mutation_chance*(i**-decreasing_mutation_factor))
                    new_specimens.append(kid_1)
                    new_specimens.append(kid_2)
                else:
                    father.mutate(mutation_chance=self.mutation_chance*(i**-decreasing_mutation_factor))
                    mother.mutate(mutation_chance=self.mutation_chance*(i**-decreasing_mutation_factor))
                    new_specimens.append(father)
                    new_specimens.append(mother)
            self.specimens = new_specimens
            self.calculate_all_fitnesses()
            for specimen in range(len(self.specimens)):
                self.specimens[specimen].set_fitness(self.fitnesses[specimen])
            self.specimens.sort(key=lambda x: -x.fitness)
            print(f'Best fitness: {self.specimens[0].fitness}')
            print(f'Worst fitness: {self.specimens[-1].fitness}')
            if save_as_negative:
                file_count = len(next(os.walk(os.path.join(os.path.pardir, 'data/negative/')))[2])
                for specimen in self.specimens:
                    if abs(specimen.fitness - 1) <= epsilon:
                        file_count += 1
                        sf.write(os.path.join(os.path.pardir, f'data/negative/neg_{file_count}.wav'),
                                 specimen.features, sr)
                        print(f'Saved file: neg_{file_count}.wav')
            if save and best <= self.specimens[0].fitness:
                sf.write(os.path.join(os.path.pardir,
                                      f'data/generated/epoch-{i}-fitness-{self.specimens[0].fitness}.wav'),
                         self.specimens[0].features, sr)
                best = self.specimens[0].fitness
