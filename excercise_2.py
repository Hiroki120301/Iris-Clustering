import numpy as np
from matplotlib import cm
import matplotlib.patches as patches
import math
import matplotlib.pyplot as plt
from excercise import open_file


class Iris :
    def __init__(self, name, petal_lengths, petal_widths) :
        self.name = name
        self.petal_lengths = petal_lengths
        self.petal_widths = petal_widths


def iris_object(data, color_map) :
    flowers = set()
    for row in data :
        flowers.add(row[4])

    iris_objects = {}
    for flower in flowers :
        iris_objects[color_map[flower]] = (Iris(flower, [], []))

    for row in data :
        iris_objects[color_map[row[4]]].petal_widths.append(float(row[3]))
        iris_objects[color_map[row[4]]].petal_lengths.append(float(row[2]))

    return iris_objects


def scatter_plot(plot, iris_objects, names, title):
    blue_patch = patches.Patch(color='blue', label='Virginica')
    green_patch = patches.Patch(color='green', label='Versicolor')

    for name in names :
        plot.scatter(iris_objects[name].petal_lengths, iris_objects[name].petal_widths, c=name, alpha=0.5)

    plot.legend(handles=[green_patch, blue_patch], loc='lower right')
    plot.set_ylim([0.9, 2.6])
    plot.set_xlim([2.5, 7.5])
    plot.set_title(title)
    plot.set_ylabel('Petal Width (cm)')
    plot.set_xlabel('Petal Length (cm)')
    plot.grid(True)


def line_calculation(inputs, weights) :
    return (inputs[0] * weights[0] + inputs[1] * weights[1]) / (-weights[2])


def line_plot(plot, weights) :
    line_lengths = np.linspace([1, 7.5], 100)
    line_widths = []
    for i in range(len(line_lengths)) :
        line_widths.append(line_calculation([1, line_lengths[i]], weights))
    plot.plot(line_lengths, line_widths, 'r')


def weighting_input(inputs, weights) :
    sum = 0
    for input, weight in zip(inputs, weights) :
        sum += input * weight
    return sum


def single_layer(inputs, weights) :
    res = math.pow(math.e, -weighting_input(inputs, weights)) + 1
    return 1 / res


def surface_plot(plot, weights, title) :
    width = np.linspace(0, 4, 100)
    length = np.linspace(0, 7.5, 100)
    x_axis, y_axis = np.meshgrid(length, width)
    outputs = []

    for x in range(0, 100) :
        outputs.append([])
        for y in range(0, 100) :
            outputs[x].append(single_layer([1, x_axis[x][y], y_axis[x][y]], weights))

    graph_outputs = np.array(outputs)

    plot.set_zlim([0, 1])
    plot.plot_surface(x_axis, y_axis, graph_outputs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plot.set_ylabel('Petal Width (cm)')
    plot.set_xlabel('Petal Length (cm)')
    plot.set_title(title)


def simple_classifier(iris_objects, weights, names, desired_class) :
    decided_class = {names[0] : Iris(names[0], [], []), names[1] : Iris(names[1], [], [])}
    for name in names :
        for (x, y) in zip(iris_objects[name].petal_lengths, iris_objects[name].petal_widths) :

            if single_layer([1, x, y], weights) > 0.5 :
                decided_class[desired_class['1']].petal_lengths.append(x)
                decided_class[desired_class['1']].petal_widths.append(y)
            else :
                decided_class[desired_class['0']].petal_lengths.append(x)
                decided_class[desired_class['0']].petal_widths.append(y)

    return decided_class


def main() :
    # Initialization of data
    data_set = open_file('irisdata.csv')
    colors_maps = {"virginica" : "blue", "versicolor" : "green", "setosa" : "red"}
    iris = iris_object(data_set, colors_maps)
    cols = ['green', 'blue']
    weights = [-11, 1.25, 2.9]
    target_input = {'0': 'green', '1': 'blue'}

    # Question 2a
    fig = plt.figure(1, (10, 9))
    fig.subplots_adjust(0.1)
    scatter_plot(fig.add_subplot(221), iris, cols, "2nd and 3rd Iris Classes")

    # Question 2c
    fig_question_c = fig.add_subplot(222)
    scatter_plot(fig_question_c, iris, cols, "Decision Boundary")
    line_plot(fig_question_c, weights)

    # Question 2d
    surface_plot(fig.add_subplot(223, projection='3d'), weights, 'Surface Plot')

    # Question 2e
    decision_class = simple_classifier(iris, weights, cols, target_input)
    fig_question_e = fig.add_subplot(224)
    scatter_plot(fig_question_e, decision_class, cols, "Simple Classifier")
    line_plot(fig_question_e, weights)

    # To display all the plots above
    plt.show()


if __name__ == "__main__" :
    main()
