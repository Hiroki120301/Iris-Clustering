import math
from matplotlib import pyplot as plt
from excercise_2 import line_plot, simple_classifier, single_layer, scatter_plot, iris_object
from excercise import open_file


class Iris:
    def __init__(self, name, petal_lengths, petal_widths) :
        self.name = name
        self.petal_lengths = petal_lengths
        self.petal_widths = petal_widths


def error_computation(iris, weight, names, output):
    sum = 0
    num_points = 0
    for name in names:
        for petal_length, petal_width in zip(iris[name].petal_lengths, iris[name].petal_widths):
            num_points += 1
            diff = single_layer([1, petal_length, petal_width], weight) - output[name]
            sum += diff ** 2
    return sum / num_points


def gradient_computation(iris, weights, names, output):
    num_points = 0
    gradient = [0, 0, 0]
    for name in names:
        for (x, y) in zip(iris[name].petal_lengths, iris[name].petal_widths):
            num_points += 1
            sigmoid = single_layer([1, x, y], weights)
            derivative = (sigmoid - output[name]) * sigmoid * (1 - sigmoid)
            gradient[0] += derivative * 1
            gradient[1] += derivative * float(x)
            gradient[2] += derivative * float(y)
    epsilon = .25
    for i in range(0, len(gradient)):
        gradient[i] = (gradient[i] * 2) / num_points
        change = gradient[i] * epsilon
        weights[i] -= change
    return weights


def main():
    # data initialization
    data_set = open_file('irisdata.csv')
    colors_maps = {"virginica" : "blue", "versicolor" : "green", "setosa" : "red"}
    iris = iris_object(data_set, colors_maps)
    expected_output = {'green': 0, 'blue': 1}
    cols = ['green', 'blue']
    # weight initialization of small and large error
    small_error_weights = [-11, 1.25, 2.9]
    large_error_weights = [-3.0, 0.23, 1.9]
    target_input = {'0': 'green', '1': 'blue'}

    # Question 3b
    small_error_classes = simple_classifier(iris, small_error_weights, cols, target_input)
    print(error_computation(iris, small_error_weights, cols, expected_output))
    print(error_computation(iris, large_error_weights, cols, expected_output))
    fig = plt.figure(1, (10, 9))
    fig.subplots_adjust(hspace=0.3)
    fig_sub = fig.add_subplot(221)
    scatter_plot(fig_sub, small_error_classes, cols, "Small Error Classification")
    line_plot(fig_sub, small_error_weights)
    large_error_classes = simple_classifier(iris, large_error_weights, cols, target_input)
    fig_sub2 = fig.add_subplot(222)
    scatter_plot(fig_sub2, large_error_classes, cols, "Large Error Classification")
    line_plot(fig_sub2, large_error_weights)

    # Question 3d
    fig_sub3 = fig.add_subplot(223)
    scatter_plot(fig_sub3, large_error_classes, cols, "Large Error Classification")
    line_plot(fig_sub3, large_error_weights)

    # Question 3e
    for i in range(0, 5):
        large_error_weights = gradient_computation(iris, large_error_weights, cols, expected_output)
    print(large_error_weights)
    print(error_computation(iris, large_error_weights, cols, expected_output))
    large_error_classes = simple_classifier(iris, large_error_weights, cols, target_input)
    fig_sub4 = fig.add_subplot(224)
    scatter_plot(fig_sub4, large_error_classes, cols, "Large Error Classification After 5 steps")
    line_plot(fig_sub4, large_error_weights)
    plt.show()


if __name__ == "__main__":
    main()
