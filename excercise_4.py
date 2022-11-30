import random

import numpy as np
from matplotlib import pyplot as plt
from excercise import open_file
from excercise_2 import scatter_plot, line_plot, iris_object
from excercise_3 import error_computation, gradient_computation


class Iris :
    def __init__(self, name, petal_lengths, petal_widths) :
        self.name = name
        self.petal_lengths = petal_lengths
        self.petal_widths = petal_widths


def gradient_descent(iris, weights, colors, outputs) :
    errors = [0.00001 + error_computation(iris, weights, colors, outputs), error_computation(iris, weights, colors, outputs)]
    list_of_weights = [weights, weights]
    steps = 1

    while errors[steps - 1] - errors[steps] > 0.000001:
        previous = list_of_weights[steps].copy()
        updated_weights = gradient_computation(iris, previous, colors, outputs)
        list_of_weights.append(updated_weights)
        errors.append(error_computation(iris, updated_weights, colors, outputs))
        steps += 1

    return list_of_weights, steps, errors


def random_weight_generator() :
    return [-1 - random.random() * 12, random.random() * 3, random.random() * 6]


def main() :
    # data initialization
    data_set = open_file('irisdata.csv')
    colors_set = {"virginica" : "red", "versicolor" : "blue", "setosa" : "green"}
    iris = iris_object(data_set, colors_set)
    colors = ['blue', 'red']
    expected_output = {'blue': 0, 'red': 1}
    weight_set = [-11, 1.25, 2.9]

    # Question b
    output_b, steps_b, errors_b = gradient_descent(iris, weight_set, colors, expected_output)

    fig_b = plt.figure(3, figsize=(10, 9))
    fig_b.subplots_adjust(hspace=0.35)

    fig_initial_b = fig_b.add_subplot(221)
    scatter_plot(fig_initial_b, iris, colors, "Initial Classification")
    line_plot(fig_initial_b, output_b[1])

    fig_middle_b = fig_b.add_subplot(222)
    scatter_plot(fig_middle_b, iris, colors, f"After {int(len(output_b) / 2)}")
    line_plot(fig_middle_b, output_b[int(len(output_b) / 2)])

    fig_final_b = fig_b.add_subplot(223)
    scatter_plot(fig_final_b, iris, colors, f"After {int(len(output_b))}")
    line_plot(fig_final_b, output_b[int(len(output_b) - 1)])

    fig_obj = fig_b.add_subplot(224)
    fig_obj.plot(range(0, steps_b + 1), errors_b)
    fig_obj.set_title("Objective Function")
    fig_obj.set_ylabel("Mean Squared Error")
    fig_obj.set_xlabel("Number of Iterations")
    fig_obj.grid(True)
    plt.show()

    # Question c
    output_c, steps_c, errors_c = gradient_descent(iris, random_weight_generator(), colors, expected_output)
    fig_c = plt.figure(3, figsize=(10, 9))
    fig_c.subplots_adjust(hspace=0.35)
    fig_initial_c = fig_c.add_subplot(221)
    scatter_plot(fig_initial_c, iris, colors, "Initial Classification")
    line_plot(fig_initial_c, output_c[1])

    fig_middle_c = fig_c.add_subplot(222)
    scatter_plot(fig_middle_c, iris, colors, f"After {int(len(output_c) / 2)}")
    line_plot(fig_middle_c, output_c[int(len(output_c) / 2)])

    fig_final_c = fig_c.add_subplot(223)
    scatter_plot(fig_final_c, iris, colors, f"After {int(len(output_c))}")
    line_plot(fig_final_c, output_c[int(len(output_c) - 1)])
    plt.show()


if __name__ == "__main__" :
    main()
