import glob
import os

from flask_restful import Resource


class Musicc(Resource):

    def get(self):
        """
        Method to get all saved music files
        :return: dict: {"filename": "url"}
        """
        res = {}
        for file in os.listdir('../data/saved'):
            res[file] = os.path.join('../data/saved', file)
        return res
