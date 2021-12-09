from src.mnist_loader import load_data_wrapper
from src.network import Network
import sys


def main():
    training_data, _, test_data = load_data_wrapper()

    network = Network([784, 30, 10])

    epochs = 10
    mini_batch_size = 10
    eta = 3.0

    network.SGD(
        training_data,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        eta=eta,
    )

    # Validate model accuracy
    accuracy_baseline = 93.0
    accuracy_tolerance = 1.2
    total = len(test_data)
    correct = network.evaluate(test_data)
    accuracy = round(correct/total, 2)
    print("Accuracy: {} / {} = {}".format(correct, total, accuracy))
    print("Expected: {} +- {}".format(accuracy_baseline, accuracy_tolerance))

    if accuracy < accuracy_baseline - accuracy_tolerance:
        sys.exit(1)


if __name__ == "__main__":
    main()
