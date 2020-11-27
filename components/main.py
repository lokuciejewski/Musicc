import librosa

from components.musicc import Musicc

if __name__ == '__main__':
    features = [(librosa.feature.melspectrogram, (128, 1291)),
                (librosa.feature.tempogram, (384, 1291)),
                (librosa.feature.rms, (1, 1291)),
                (librosa.feature.mfcc, (20, 1291)),
                (librosa.feature.chroma_stft, (12, 1291))]
    musicc = Musicc(features, number_of_samples=500, sample_length=660719, force_new_positives=False,
                    number_of_specimen=100, initial_mutation_chance=0.01, crossover_chance=0.1)
    musicc.train_networks(50, 100)
    musicc.run_evolution(number_of_epochs=10, decreasing_mutation_factor=0.1, save=True,
                         save_as_negative=True, epsilon=0.1)