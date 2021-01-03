import http
import os

import librosa
from flask_restful import Resource, reqparse

from components.musicc import Musicc


class MusiccRes(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('title', type=str)
    parser.add_argument('networks_weights', type=dict)

    features = [(librosa.feature.melspectrogram, 128),
                (librosa.feature.tempogram, 384),
                (librosa.feature.rms, 1),
                (librosa.feature.mfcc, 20),
                (librosa.feature.chroma_stft, 12)]

    # musicc = Musicc(features, number_of_samples=1000, sample_length=660719, force_new_positives=False,
    #               number_of_specimen=1000, initial_mutation_chance=0.1, crossover_chance=0.1, load_networks=True)

    def get(self):
        """
        Method to get all saved music files urls
        :return: dict: {"songs": <list of dict: {<title>: <url>}}
        """
        res = []
        for file in os.listdir('../data/saved'):
            filename = file.split('.')[0]
            res.append({'title': filename, 'url': os.path.join('../data/saved', file)})
        return {'songs': res}, http.HTTPStatus.OK

    def post(self):
        """
        Method to start music generation with correct params
        :return: dict: {"message": <status>}
        """
        data = self.parser.parse_args()
        self.musicc.run_evolution(100)
        pass


class DataAccess(Resource):

    def get(self):
        """
        Method to return the data file from the location
        :return:
        """
        pass
