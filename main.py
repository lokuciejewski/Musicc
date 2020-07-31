import librosa.display
import specimen
import soundfile as sf

filepath = '/home/anemiq/Desktop/Music ML Data/fma_medium/000/'


if __name__ == '__main__':

    y, sr = librosa.load(f'{filepath}000208.mp3')
    s = librosa.stft(y)
    beats = librosa.beat.beat_track(y, sr=sr)
    # test = librosa.istft(x.features)

    evo = specimen.Evolution(1000, len(y), s, 0.0001, 0.1, librosa.stft)
    evo.run_epochs(100, save=True)


