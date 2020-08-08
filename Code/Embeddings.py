"""
Code to obtain the word embeddings.
Based on the implementation from
https://github.com/hans/glove.py/blob/582549ddeeeb445cc676615f64e318aba1f46295/glove.py,
personally verified.
"""

import numpy as np
import sys
import os
import errno
import logging
from random import shuffle


class Embeddings:

    co_occurrence_matrix = 0
    num_of_targets = 0
    num_of_contexts = 0

    target_bias = 0
    context_bias = 0

    target_vectors = 0
    context_vectors = 0

    gradient_target_vectors = 0
    gradient_context_vectors = 0

    gradient_target_bias = 0
    gradient_context_bias = 0

    total_progress = 0
    cur_progress = 0

    def __init__(self, path_to_co_occurrence_matrix, embedding_size,max_num_of_iterations, learning_rate, x_max, alpha):

        self.co_occurrence_matrix = np.load(path_to_co_occurrence_matrix)
        self.num_of_targets = self.co_occurrence_matrix.shape[0]
        self.num_of_contexts = self.co_occurrence_matrix.shape[1]

        # Indices of the co-occurrence matrix
        data = []

        for i in range(self.num_of_targets):
            for j in range(self.num_of_contexts):
                data.append((i, j))

        self.target_vectors = (np.random.rand(self.num_of_targets, embedding_size) - 0.5) / \
                              float(embedding_size + 1)
        self.context_vectors = (np.random.rand(self.num_of_contexts, embedding_size) - 0.5) / \
                               float(embedding_size + 1)

        self.target_bias = (np.random.rand(self.num_of_targets) - 0.5) / float(embedding_size + 1)
        self.context_bias = (np.random.rand(self.num_of_contexts) - 0.5) / float(embedding_size + 1)

        self.gradient_target_vectors = np.ones((self.num_of_targets, embedding_size), dtype=np.float64)
        self.gradient_context_vectors = np.ones((self.num_of_contexts, embedding_size), dtype=np.float64)

        self.gradient_target_bias = np.ones(self.num_of_targets, dtype=np.float64)
        self.gradient_context_bias = np.ones(self.num_of_contexts, dtype=np.float64)

        iteration = 1
        self.cur_progress = 1
        self.total_progress = max_num_of_iterations * self.num_of_targets * self.num_of_contexts

        # Logs the Error per Iteration
        logging.basicConfig(filename=path_to_co_occurrence_matrix+"log.info", level=logging.INFO)

        while iteration <= max_num_of_iterations:

            cur_cost = self.run_iter(data, learning_rate, x_max, alpha)

            logging.info("("+str(iteration)+ ", " + str(cur_cost) + ")")

            iteration += 1

        self.save_matrix(self.target_vectors, path_to_co_occurrence_matrix, embedding_size)

    def run_iter(self, data, learning_rate, x_max, alpha):
        """
        Runs a single Iteration of GloVe training using the given co-occurrence matrix and the previously computed
        Weight Vectors and Biases, plus the Gradient Histories

        Returns the Cost associated with the given Weight Assignments and
        updates the Weights by online AdaGrad in Place.
        """

        global_cost = 0

        # Iterate over the Matrix randomly
        shuffle(data)

        for (target, context) in data:

                weight = (self.co_occurrence_matrix[target, context] / x_max) ** alpha if \
                    self.co_occurrence_matrix[target, context] < x_max else 1

                # Compute inner component of cost function, which is used in
                # both overall cost calculation and in gradient calculation
                cost_inner = (self.target_vectors[target].dot(self.context_vectors[context])
                              + self.target_bias[target: target + 1][0]
                              + self.context_bias[context: context + 1][0]
                              - np.log2(self.co_occurrence_matrix[target, context] + 1))

                # Compute cost
                #
                #   $$ J = f(X_{ij}) (J')^2 $$
                cost = weight * (cost_inner ** 2)

                # Add weighted cost to the global cost tracker
                global_cost += 0.5 * cost

                # Compute gradients for word vector terms.
                grad_target = cost_inner * self.context_vectors[context]
                grad_context = cost_inner * self.target_vectors[target]

                # Compute gradients for bias terms
                grad_bias_target = cost_inner
                grad_bias_context = cost_inner

                # Now perform adaptive updates to minimize the Cost
                self.target_vectors[target] -= (learning_rate * grad_target /
                                                np.sqrt(self.gradient_target_vectors[target]))
                self.context_vectors[context] -= (learning_rate * grad_context /
                                                  np.sqrt(self.gradient_context_vectors[context]))

                self.target_bias[target: target + 1] -= (learning_rate * grad_bias_target /
                                                         np.sqrt(self.gradient_target_bias[target: target + 1]))
                self.context_bias[context: context + 1] -= (learning_rate * grad_bias_context /
                                                            np.sqrt(self.gradient_context_bias[context: context + 1]))

                # Update squared gradient sums
                self.gradient_target_vectors[target] += np.square(grad_target)
                self.gradient_context_vectors[context] += np.square(grad_context)
                self.gradient_target_bias[target: target + 1] += grad_bias_target ** 2
                self.gradient_context_bias[context: context + 1] += grad_bias_context ** 2

                self.print_progress(self.cur_progress, self.total_progress)

                self.cur_progress += 1

        return global_cost

    def save_matrix(self, matrix, path, embedding_size):
        """
        Saves an obtained Embedding Matrix in a specified Directory
        """
        np.save(path + "embedding_" + str(embedding_size) + "_matrix.npy", matrix)

    def print_progress(self, iteration, total):
        """
        Visualization of progress
        Based on https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
        :param iteration: current iteration
        :param total: total number of iterations
        :return:
        """

        decimals = 1
        bar_length = 50

        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        prefix = ""
        suffix = ""

        sys.stdout.write('\r%s|%s| %s%s %s' % (prefix, bar, percents, ' %', suffix)),

        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

# Takes the Path to the Co-Occurrence Matrix, the Number of Embeddings, the maximum Number of Iterations,
# the Learning Rate, and x_max/ alpha for the fractional Weighting Function
# The maximum Number of Iterations, learning Rate, and x_max/ alpha are used as in the original GloVe Publication
Embeddings("/Path/to/Co-Occurrence_Matrix/", 5, 50, 0.05, 100, 0.75)
