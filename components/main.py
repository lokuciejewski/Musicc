import warnings

from components.network import Network, Helper

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    train, test = Helper.prepare_dataset(size=10)

    input_shape = train[0][0].shape

    network = Network(input_shape=input_shape)

    x_train, y_train = Helper.get_x_y(train)

    network.train(x_train=x_train, y_train=y_train, epochs=100)

    x_test, y_test = Helper.get_x_y(test)

    loss, acc = network.evaluate(x_test=x_test, y_test=y_test)

    network.save_model(f'models/model_a{acc}_l{loss}')
