import math
import multiprocessing as mp
import os
import random
import time

import numpy as np
import soundfile as sf
from pathos.multiprocessing import ProcessPool as Pool


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
        self.fitness = math.inf

    def generate_features(self):
        return np.random.uniform(-0.1, 0.1, self.length)

    def mutate(self, mutation_chance=0.001):
        # instead of going through ALL genes I could just randomly choose X,
        # where X is the len(genotype)/mutation_chance and then mutate them, SHOULD be faster?
        mu = float(np.mean(self.features))
        sigma = float(np.std(self.features))

        number_of_genes = int(len(self.features) * mutation_chance)
        for i in range(number_of_genes):
            gene = random.randint(0, len(self.features) - 1)
            self.features[gene] = np.average([self.features[gene], self.features[gene - 1]])
            self.features[gene] += random.gauss(mu=mu, sigma=sigma)
        # Not sure if this is ok, maybe there are better ways to mutate
        """
        for i, row in enumerate(self.features):
            if mutation_chance > random.random():
                self.features[i] = np.average([self.features[i], self.features[i - 1]])
                self.features[i] += random.gauss(mu=mu, sigma=sigma)"""

    def crossover(self, partner, cuts=100):  # lepiej uniform crossover?
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

    def __init__(self, neural_networks, number_of_specimen, specimen_length, mutation_chance, crossover_chance,
                 features_list):
        self.neural_networks = neural_networks
        self.mutation_chance = mutation_chance
        self.crossover_chance = crossover_chance
        self.fitnesses = [0 for i in range(number_of_specimen)]
        self.features_list = features_list
        self.specimens = []
        for i in range(number_of_specimen):
            specimen = Specimen(specimen_length)
            self.specimens.append(specimen)
            print(f'Generated {i + 1} specimen of {number_of_specimen}', end='\r')
        self.calculate_all_fitnesses()
        for i in range(number_of_specimen):
            self.specimens[i].set_fitness(self.fitnesses[i])
        self.specimens.sort(key=lambda x: -x.fitness)

    def calculate_all_fitnesses(self):
        print('\nStarted fitness calculation')
        features = [[] for feature in self.features_list]
        print('Started specimen conversion')
        # TODO: make this parallel
        for specimen in self.specimens:
            for i, feature in enumerate(self.features_list):
                features[i].append(feature(specimen.features))
        print('Finished specimen conversion')
        temp = []
        for i, network in enumerate(self.neural_networks):
            start = time.time()
            temp.append(network.predict(np.array(features[i])))
            stop = time.time()
            print(f'Finished network {i + 1}/{len(self.neural_networks)} in {stop - start}s')
        for specimen in range(len(self.specimens)):
            # min_fitness = min([temp[prediction][specimen][1] for prediction in range(len(self.features_list))])
            average_fitness = np.average([temp[prediction][specimen][1]
                                          for prediction in range(len(self.features_list))])
            self.fitnesses[specimen] = average_fitness
            # 1 because list [0, 1] represents max similarity, [1, 0] represents min
            # similarity
        print('Finished fitness calculation')

    def select_n_best(self, n):
        return self.specimens[:n]

    def inject_specimen(self, specimen, index):
        self.specimens[index] = specimen

    def create_offsprings(self, epoch_number, decreasing_mutation_factor):
        father = random.choice(self.specimens)
        mother = random.choice(self.specimens)
        if random.random() < self.crossover_chance:
            first_offspring, second_offspring = father.crossover(mother)
            first_offspring.mutate(mutation_chance=self.mutation_chance * (epoch_number ** -decreasing_mutation_factor))
            second_offspring.mutate(
                mutation_chance=self.mutation_chance * (epoch_number ** -decreasing_mutation_factor))
            return first_offspring, second_offspring
        else:
            father.mutate(mutation_chance=self.mutation_chance * (epoch_number ** -decreasing_mutation_factor))
            mother.mutate(mutation_chance=self.mutation_chance * (epoch_number ** -decreasing_mutation_factor))
            return father, mother

    @staticmethod
    def create_n_offsprings(n, specimens, crossover_chance, mutation_chance, current_epoch, decreasing_mutation_factor):
        result = []
        while len(result) != n:
            if random.random() < crossover_chance:
                father = random.choice(specimens)
                mother = random.choice(specimens)
                specimen, _ = father.crossover(mother)
            else:
                specimen = random.choice(specimens)
            specimen.mutate(mutation_chance * (current_epoch ** -decreasing_mutation_factor))
            result.append(specimen)
        return result

    def run_epochs(self, number_of_epochs, decreasing_mutation_factor=0.1, save=False, sr=22050, save_as_negative=False,
                   epsilon=0.1):
        best = -np.math.inf
        specimen_count = len(self.specimens)
        for i in range(1, number_of_epochs + 1):
            print(f'Started epoch {i}')
            n = 2 * int(specimen_count / 20)
            new_specimens = self.select_n_best(n)
            missing_specimen = specimen_count - n
            print('Creating new specimens.', end='')
            pool = Pool(mp.cpu_count())
            result = pool.amap(Evolution.create_n_offsprings, [missing_specimen], [self.specimens],
                               [self.crossover_chance],
                               [self.mutation_chance], [i], [decreasing_mutation_factor])
            while not result.ready():
                print('.', end='')
                time.sleep(1)
            result = result.get()
            new_specimens.extend(result[0])
            '''
            while len(new_specimens) != specimen_count:
                print(f'Started specimen {len(new_specimens)}/{specimen_count}', end='\r')
                offspring_1, offspring_2 = self.create_offsprings(i, decreasing_mutation_factor)
                new_specimens.append(offspring_1)
                new_specimens.append(offspring_2)
            '''
            print('\r', end='')
            print('\nFinished all specimens', end='')
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
