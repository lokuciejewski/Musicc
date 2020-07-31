import random

import librosa
import numpy as np
from scipy.spatial.distance import cdist, sqeuclidean
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
        return np.random.rand(self.length)

    def mutate(self, mutation_chance=0.001):
        for i, row in enumerate(self.features):
            if mutation_chance > random.random():
                self.features[i] = -self.features[i]

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

    def __init__(self, number_of_specimen, specimen_length, target, mutation_chance, crossover_chance, feature=None):
        self.mutation_chance = mutation_chance
        self.crossover_chance = crossover_chance
        self.target = target
        for i in range(number_of_specimen):
            specimen = Specimen(specimen_length)
            specimen.set_fitness(self.calculate_fitness(specimen, feature))
            self.specimens.append(specimen)
            print(f'Generated {i} specimen of {number_of_specimen}')
        self.specimens.sort(key=lambda x: x.fitness)

    def calculate_fitness(self, specimen, feature=librosa.stft):
        if feature is None:
            return abs(np.average(1 - cdist(librosa.stft(specimen.features), self.target)))
        else:
            return abs(np.average(1 - cdist(feature(specimen.features), self.target)))

    def select_n_best(self, n):
        self.specimens.sort(key=lambda x: x.fitness)
        return self.specimens[:n]

    def select_random(self):
        return self.specimens[random.randint(0, len(self.specimens) - 1)]

    def run_epochs(self, number_of_epochs, save=False, sr=22050):
        best = 100000
        for i in range(number_of_epochs):
            print(f'Started epoch {i}')
            new_specimens = self.select_n_best(int(len(self.specimens)/10))
            for j in range(int((len(self.specimens) * 0.9)/2)):
                print(f'Started specimens {2 * j}/{len(self.specimens) * 0.9}')
                father = self.select_random()
                mother = self.select_random()
                if random.random() < self.crossover_chance:
                    kid_1, kid_2 = father.crossover(mother)
                    kid_1.mutate(mutation_chance=self.mutation_chance)
                    kid_2.mutate(mutation_chance=self.mutation_chance)
                    new_specimens.append(kid_1)
                    new_specimens.append(kid_2)
                else:
                    father.mutate(mutation_chance=self.mutation_chance)
                    mother.mutate(mutation_chance=self.mutation_chance)
                    new_specimens.append(father)
                    new_specimens.append(mother)
            self.specimens = new_specimens
            for specimen in self.specimens:
                specimen.set_fitness(self.calculate_fitness(specimen))
            self.specimens.sort(key=lambda x: x.fitness)
            print(f'Best fitness: {self.specimens[0].fitness}')
            print(f'Worst fitness: {self.specimens[-1].fitness}')
            if save and best > self.specimens[0].fitness:
                sf.write(f'epoch-{i}-fitness-{self.specimens[0].fitness}.wav', self.specimens[0].features, sr)
                best = self.specimens[0].fitness
