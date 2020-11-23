from flask import Flask
from flask_restful import Api

from backend import resources

if __name__ == '__main__':
    app = Flask(__name__)
    api = Api(app)

    api.add_resource(resources.Musicc, '/')

    app.run(debug=True)
