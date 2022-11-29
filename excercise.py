import csv
import matplotlib.pyplot as plt
import random
import numpy as np


class Iris :
    def __init__(self, sepal_length, sepal_width, petal_lengths, petal_widths, name) :
        self.name = 0 if name == "setosa" else (1 if name == "versicolor" else 2)
        self.sepal_length = float(sepal_length)
        self.sepal_width = float(sepal_width)
        self.petal_lengths = float(petal_lengths)
        self.petal_widths = float(petal_widths)


def converter(data_set) :
    iris_objects = []

    for data in data_set :
        iris = Iris(data[0], data[1], data[2], data[3], data[4])
        iris_objects.append(iris)
    return iris_objects


def open_file(file_name) :
    data_set = []
    label = False
    with open(file_name) as file :
        # each data is separated by a ','
        csvreader = csv.reader(file, delimiter=',')
        for row in csvreader :
            if not label :
                label = True
            else :
                data_set.append(row)
    return data_set


def random_centroids_point(low, high) :
    return low + (high - low) * random.random()


def initialize_centroids(data_set, k) :
    sl_min = sw_min = pl_min = pw_min = float('inf')
    sl_max = sw_max = pw_max = pl_max = float('-inf')
    for data in data_set :
        sl_min = min(data.sepal_length, sl_min)
        sl_max = max(data.sepal_length, sl_max)
        sw_min = min(data.sepal_width, sw_min)
        sw_max = max(data.sepal_width, sw_max)
        pl_min = min(data.petal_lengths, pl_min)
        pl_max = max(data.petal_lengths, pl_max)
        pw_min = min(data.petal_widths, pw_min)
        pw_max = max(data.petal_widths, pw_max)

    centroids = []
    for i in range(k) :
        centroids.append([random_centroids_point(sl_min, sl_max), random_centroids_point(sw_min, sw_max),
                          random_centroids_point(pl_min, pl_max), random_centroids_point(pw_min, pw_max)])

    return centroids


def euclidean(point_one, point_two) :
    return ((point_one.sepal_length - point_two[0]) ** 2 + (point_one.sepal_width - point_two[1]) ** 2
            + (point_one.petal_lengths - point_two[2]) ** 2 + (point_one.petal_widths - point_two[3]) ** 2) ** 0.5


def get_labels(data_set, centroids) :
    labels = []
    for data in data_set :
        min_distance = float('inf')
        label = None
        for i, centroid in enumerate(centroids) :
            dist = euclidean(data, centroid)
            if dist < min_distance :
                min_distance = dist
                label = i
        labels.append(label)

    return labels


def update_centroids(data_set, labels, k) :
    updated_centroids = [[0, 0, 0, 0] for i in range(k)]
    counts = [0] * k

    for data, label in zip(data_set, labels) :
        updated_centroids[label][0] += data.sepal_length
        updated_centroids[label][1] += data.sepal_width
        updated_centroids[label][2] += data.petal_lengths
        updated_centroids[label][3] += data.petal_widths
        counts[label] += 1

    for i, (a, b, x, y) in enumerate(updated_centroids) :
        updated_centroids[i] = (a / counts[i], b / counts[i], x / counts[i], y / counts[i])

    return updated_centroids


# def objective_function(data_set, centroids, labels):
#     return np.sum(np.square(data_set - centroids[0]))


def errors(previous, next):
    movement = 0
    for prev, nxt in zip(previous, next) :
        movement += ((prev[0] - nxt[0]) ** 2 + (prev[1] - nxt[1]) ** 2 +
                     (prev[2] - nxt[2]) ** 2 + (prev[3] - nxt[3]) ** 2) ** 0.5
    return movement < 0.0000001, movement


def print_progress(plot, data_set, centroids, title, num_steps) :
    for i in range(3):
        for data in data_set:
            if data.name == i:
                plot.scatter(data.petal_lengths, data.petal_widths, c="g" if i == 0 else ("r" if i == 1 else "b"))

    for i in range(len(centroids)):
        plt.scatter(centroids[i][2], centroids[i][3], c="k")
    plot.set_title(title + f" After {num_steps} steps")
    plot.set_ylabel('Petal Width')
    plot.set_xlabel('Petal Length')
    plot.set_ylim([0.0, 2.75])
    plot.set_xlim([0.0, 7.5])
    plot.grid(True)


def equidistance(pairs):
    coordinates = []
    t = np.linspace(-2, 2, 100)
    for pair in pairs:
        x = (pair[1][2] + pair[0][2]) / 2 + (pair[0][2] - pair[1][2]) * t
        y = (pair[1][3] + pair[0][3]) / 2 - (pair[0][3] - pair[1][3]) * t
        coordinates.append((x, y))
    return coordinates


def pair_generator(list_of_centroids):
    visited = set()

    pairs = []

    for centroid in list_of_centroids:
        if centroid not in visited:
            visited.add(centroid)
            dist = float('inf')
            pair = (0, 0)
            for point in list_of_centroids:
                if centroid != point:
                    new_dist = ((centroid[2] - point[2]) ** 2 + (centroid[3] - point[3]) ** 2) ** 0.5
                    if new_dist < dist:
                        dist = new_dist
                        pair = (centroid, point)
            visited.add(pair[1])
            pairs.append(pair)
    return pairs


def kmeans(data_set, k) :
    centroids = initialize_centroids(data_set, k)
    stopping_criteria = False

    list_of_centroids = [centroids]
    num_steps = 0
    question_one_b_output = []
    while not stopping_criteria:
        previous_centroids = centroids
        labels = get_labels(data_set, centroids)
        centroids = update_centroids(data_set, labels, k)
        list_of_centroids.append(centroids)
        num_steps += 1
        stop, error = errors(previous_centroids, centroids)
        # question_one_b_output.append(objective_function(data_set, centroids, labels))
        if stop:
            stopping_criteria = True

    fig = plt.figure(1, (10, 9))

    print_progress(fig.add_subplot(221), data_set, list_of_centroids[0], "Initial Cluster", 0)

    print_progress(fig.add_subplot(222), data_set, list_of_centroids[int(num_steps / 2)], "Intermediate Cluster", int(num_steps / 2))

    fig_final = fig.add_subplot(223)
    print_progress(fig_final, data_set, list_of_centroids[int(num_steps)], "Converged Cluster", int(num_steps))

    final_centroid = list_of_centroids[int(num_steps)]

    pairs = pair_generator(final_centroid)
    coordinates = equidistance(pairs)
    for coord in coordinates:
        fig_final.plot(coord[0], coord[1])
    plt.show()
    # plt.scatter(range(0, num_steps), question_one_b_output)
    # plt.grid(True)
    # plt.show()


def main() :
    data_set = open_file('irisdata.csv')
    iris_objects = converter(data_set)
    kmeans(iris_objects, 2)
    kmeans(iris_objects, 3)


if __name__ == "__main__" :
    main()
