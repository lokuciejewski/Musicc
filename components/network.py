import os

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras import layers, models


class Network:
    def __init__(self, input_shape):
        self.model = models.Sequential()
        self.model.add(layers.Dense(input_shape=input_shape, units=128, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.history = None

    def train(self, x_train, y_train, epochs=100, batch_size=100):
        self.history = self.model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x_test, y_test):
        loss, acc = self.model.evaluate(x_test, y_test)
        print(f'Loss: {loss}, Accuracy: {acc}')
        return loss, acc

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = models.load_model(filepath)

    def predict(self, audio_data):
        return self.model.predict(audio_data, verbose=0)
