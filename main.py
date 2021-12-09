from src.mnist_loader import load_data_wrapper
from src.network import Network


def main():
    training_data, _, test_data = load_data_wrapper()

    network = Network([784, 30, 10])

    epochs = 5
    mini_batch_size = 10
    eta = 3.0

    network.SGD(
        training_data,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        eta=eta,
        test_data=test_data,
    )


if __name__ == "__main__":
    main()
