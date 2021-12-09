from src.mnist_loader import load_data_wrapper
from src.network import Network
import sys


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
    )

    # Validate model accuracy
    accuracy_baseline = 0.93
    accuracy_tolerance = 0.012
    total = len(test_data)
    correct = network.evaluate(test_data)
    accuracy = round(correct/total, 2)
    print("Accuracy: {} / {} = {}".format(correct, total, accuracy))
    print("Expected: {} +- {}".format(accuracy_baseline, accuracy_tolerance))

    if accuracy < accuracy_baseline - accuracy_tolerance:
        print("Accuracy too low!")
        sys.exit(1)


if __name__ == "__main__":
    main()
