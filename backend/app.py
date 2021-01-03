from flask import Flask
from flask_cors import CORS
from flask_restful import Api

from backend import resources

if __name__ == '__main__':
    app = Flask(__name__)
    api = Api(app)

    cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

    api.add_resource(resources.MusiccRes, '/api')

    app.run(debug=True)
