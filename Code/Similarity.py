"""
Code for translating two Embedding Matrices in unsupervised Fashion.
@author: Tobias Elßner
"""

import numpy as np


class TransRank:

    translation_directory = ""

    def __init__(self, path_to_matrix_a, path_to_matrix_b, t_directory):

        self.translation_directory = t_directory

        print("Calculate similarities...")

        matrix_a = np.load(path_to_matrix_a)
        matrix_b = np.load(path_to_matrix_b)

        self.calculate_bipartite_similarity(matrix_a, matrix_b)

    def calculate_bipartite_similarity(self, matrix_a, matrix_b):
        """
        Computes the similarity between the nodes of two bipartite graphs
        :return: two similarity matrices for each of the two parties in the graphs
        """
        num_of_rows_matrix_a = np.size(matrix_a, 0)
        num_of_cols_matrix_a = np.size(matrix_a, 1)

        num_of_rows_matrix_b = np.size(matrix_b, 0)
        num_of_cols_matrix_b = np.size(matrix_b, 1)

        # Calculate the normalization terms for the word similarities
        normalization_word_similarity = np.zeros((num_of_rows_matrix_a, num_of_rows_matrix_b))

        for k in range(num_of_rows_matrix_a):

            for l in range(num_of_rows_matrix_b):

                normalization_sum = 0

                for i_bar in range(num_of_cols_matrix_a):

                    for j_bar in range(num_of_cols_matrix_b):
                        normalization_sum += self.entry_similarity(matrix_a[k, i_bar], matrix_b[l, j_bar])

                normalization_word_similarity[k, l] = normalization_sum

        print("\t...computed normalization term for word similarities...")

        # Do the same for the embedding similarities
        normalization_embedding_similarity = np.zeros((num_of_cols_matrix_a, num_of_cols_matrix_b))

        for k in range(num_of_cols_matrix_a):

            for l in range(num_of_cols_matrix_b):

                normalization_sum = 0

                for i_bar in range(num_of_rows_matrix_a):

                    for j_bar in range(num_of_rows_matrix_b):
                        normalization_sum += self.entry_similarity(matrix_a[i_bar, k], matrix_b[j_bar, l])

                normalization_embedding_similarity[k, l] = normalization_sum

        print("\t...calculated normalization term for embedding similarities...")

        # Initialize for each similarity matrix 2 instances:
        # One on which the next iteration is calculated, and one containing the old values to determine the convergence
        # Each matrix entry is equally likely, meaning that in case of the embedding similarity matrix, each entry has
        # The value 1 / num_of_cols_matrix_a * num_of_cols_matrix_b
        # And in case of the word_similarity matrix 1 / 1 / num_of_rows_matrix_a * num_of_rows_matrix_b
        embedding_similarity_prev = np.full((num_of_cols_matrix_a, num_of_cols_matrix_b),
                                 1 / (num_of_cols_matrix_a * num_of_cols_matrix_b))
        embedding_similarity_cur = np.full((num_of_cols_matrix_a, num_of_cols_matrix_b),
                                 1 / (num_of_cols_matrix_a * num_of_cols_matrix_b))

        word_similarity_prev = np.full((num_of_rows_matrix_a, num_of_rows_matrix_b),
                             1 / (num_of_rows_matrix_a * num_of_rows_matrix_b))
        word_similarity_cur = np.full((num_of_rows_matrix_a, num_of_rows_matrix_b),
                             1 / (num_of_rows_matrix_a * num_of_rows_matrix_b))

        # Set the tolerance to the smallest possible number x which satisfies 1.0 + x != 1.0
        tolerance = np.finfo(float).eps
        condition = True

        # Count the number of iterations until convergence
        convergence_count = 0

        print("\t...computing similarities:")


        # Iterate as long as the condition holds
        while condition:

            # At each step, increase the count
            convergence_count += 1

            # By default, every iteration is also the last
            condition = False

            # Calculate the similarity between the words first
            for i in range(num_of_rows_matrix_a):
                for j in range(num_of_rows_matrix_b):

                    # Update each entry according to the formula
                    word_similarity_update = 0

                    for k in range(num_of_cols_matrix_a):
                        for l in range(num_of_cols_matrix_b):

                            word_similarity_update += embedding_similarity_prev[k, l] \
                                      * (self.entry_similarity(matrix_a[i, k], matrix_b[j, l]) /
                                         normalization_embedding_similarity[k, l])

                    word_similarity_update = (0.85 * word_similarity_update) + (0.15 / (num_of_rows_matrix_a * num_of_rows_matrix_b))

                    # If the difference between the former and the new value is larger than the tolerance
                    # The loop continues
                    if np.linalg.norm(word_similarity_prev[i, j] - word_similarity_update) > tolerance:
                        condition = True

                    # The entry is set to the update in any case
                    word_similarity_cur[i, j] = word_similarity_update

            # Secondly, the similarities between the embeddings are calculated
            for i in range(num_of_cols_matrix_a):
                for j in range(num_of_cols_matrix_b):

                    # Update each entry according to the formula
                    embedding_similarity_update = 0

                    for k in range(num_of_rows_matrix_a):
                        for l in range(num_of_rows_matrix_b):

                            embedding_similarity_update += word_similarity_prev[k, l] \
                                      * (self.entry_similarity(matrix_a[k, i], matrix_b[l, j]) /
                                         normalization_word_similarity[k, l])

                    embedding_similarity_update = (0.85 * embedding_similarity_update) + (0.15 / (num_of_cols_matrix_a * num_of_cols_matrix_b))

                    if np.linalg.norm(embedding_similarity_prev[i, j] - embedding_similarity_update) > tolerance:
                        condition = True

                    embedding_similarity_cur[i, j] = embedding_similarity_update

            # Overwrite the 'old' entries
            word_similarity_prev = word_similarity_cur.copy()
            embedding_similarity_prev = embedding_similarity_cur.copy()
            print("\t\tnumber of iterations: " + str(convergence_count))

        # When converged, save the matrices
        np.save(self.translation_directory + "emb.npy", embedding_similarity_cur)

        np.save(self.translation_directory + "word_.npy", word_similarity_cur)

        # And a text-file with the number of iterations until convergence
        with open(self.translation_directory + "conv.txt", "w") as f:
            f.write(str(convergence_count))

        print("\t...matrices saved.")


    def entry_similarity(self, entry_1, entry_2):
        """
        Bounded similarity measure between two matrix entries
        :param entry_1: first entry
        :param entry_2: second entry
        :return: similarity measure between two matrix entries; outcome between ]0, 1]
        """
        return 1 / (1 + np.linalg.norm(entry_1 - entry_2))


TransRank("/Path/to/Matrix_a",
          "/Path/to/Matrix_b",
          "/Path/to/Translation_Directory")
