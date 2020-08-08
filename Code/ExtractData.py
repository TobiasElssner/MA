"""
Code to extract {500, 1000, 2000} most common Words from a .txt-Corpus to build a Co-Occurrence Matrix based on Words
and Finite States for Context Windows of Size {2, 4, 8, 50}.
@author: Tobias Elßner
"""
import collections
import numpy as np
import pickle
import os
import errno
import sys


class ExtractData:

    # The automaton is built incrementally
    # Therefore it only needs to be initialized once
    # If the vocabulary is enlarged, new words are simply added
    automaton = {}
    final_states = []
    register = []
    state_count = 0  # Start State
    list_of_relevant_states = []

    def __init__(self, path_to_corpus, main_directory):

        self.read_corpus(path_to_corpus, main_directory)

    def read_corpus(self, corpus, main_directory):

        letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "m", "o", "p", "q", "r", "s",
                  "t", "u", "v", "w", "x", "y", "z", "ä", "ö", "ü", "ß"]
        punctuation = [".", "!", "?", ",", ";", ":", "(", ")", "[", "]", "`", "´", "\"", "'", "„", "“", "»", "«"]

        plain_context = {}

        word_list = []

        sent_count = 1
        num_of_sents = sum(1 for line in open(corpus, 'r'))
        sents = []
        print("Reading the corpus...")
        # Read the corpus
        with open(corpus, 'r') as f:

            for line in f:

                self.print_progress(sent_count, num_of_sents)

                sent_count += 1

                line = line.strip("\n")
                split = line.split("\t")
                if len(split) == 2:
                    sent = split[1].lower()
                    for p in punctuation:
                        sent = sent.replace(p, " " + p + " ")

                    sent = sent.split(" ")

                    # Replace words that do not contain letters (like hyphens, or foreign words)
                    # Or that consist only of a single letter
                    for i in range(len(sent)):

                        word = sent[i]

                        for char in word:
                            if char not in letters:
                                word = "UNK"
                                break

                        if word in letters:
                            word = "UNK"

                        if word != "UNK":
                            word_list.append(word)

                        sent[i] = word
                    sents.append(sent)
        print("...corpus read.\n")

        for top_n_words in [500, 1000, 2000]:

            print("Building co-occurrences for most common " + str(top_n_words) + "...")

            sub_directory = main_directory + "top_" + str(top_n_words) + "_words/"

            self.automaton = {}
            self.final_states = []
            self.register = []
            self.state_count = 0
            self.list_of_relevant_states = []

            try:
                os.makedirs(sub_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            # Count the words and retrieve the n most common ones
            count = collections.Counter(word_list).most_common(top_n_words)

            # Dictionary to map the context words to numbers
            word_to_num_dic = {}

            for word, count in count:
                # The corresponding number of the context word is in a list
                # This way it, can be treated like the word_to_states
                word_to_num_dic[word] = [len(word_to_num_dic)]

            num_to_word_dic = {value[0]: key for key, value in word_to_num_dic.items()}
            print("...extracting context...")
            sent_count = 1

            for sent in sents:

                self.print_progress(sent_count, num_of_sents)
                sent_count += 1

                for i in range(len(sent)):

                    if sent[i] not in word_to_num_dic:
                        break

                    context = {}

                    if sent[i] in plain_context:
                        context = plain_context[sent[i]]

                    for j in range(len(sent)):

                        if i != j and sent[j] in word_to_num_dic:
                            distance = np.abs(i - j)

                            if distance in context:

                                if sent[j] in context[distance]:
                                    context[distance][sent[j]] += 1.0
                                else:
                                    context[distance][sent[j]] = 1.0
                            else:
                                context[distance] = {}
                                context[distance][sent[j]] = 1.0

                    # Store for each word the contextual words and the distance to them
                    plain_context[sent[i]] = context

            print("...context  extracted...\n")
            print("...constructing automaton...")
            # Construct an automaton for the words
            self.construct_automaton(word_to_num_dic.keys())

            print("...automaton constructed.\n")

            # Extract the relevant states
            self.extract_rel_states()

            # Dictionary to map the relevant states to numbers
            rel_state_to_num_dic = {}

            for rel_state in self.list_of_relevant_states:
                rel_state_to_num_dic[rel_state] = len(rel_state_to_num_dic)

            num_to_rel_state_dic = {value[0]: key for key, value in word_to_num_dic.items()}

            # Dictionary to map the words to the set of relevant states they pass
            word_to_rel_states_dic = self.read_words(word_to_num_dic.keys())

            # Map the Words to relevant States to Numbers [0...K-1]
            for word in word_to_rel_states_dic:

                rel_state_nums = []

                for rel_state in word_to_rel_states_dic[word]:
                    rel_state_num = rel_state_to_num_dic[rel_state]
                    rel_state_nums.append(rel_state_num)

                word_to_rel_states_dic[word] = rel_state_nums

            self.save_pickle_object(word_list, sub_directory + "word_list.p")

            self.save_pickle_object(word_to_num_dic, sub_directory + "word_to_num_dic.p")
            self.save_pickle_object(num_to_word_dic, sub_directory + "num_to_word_dic.p")

            self.save_pickle_object(word_to_rel_states_dic, sub_directory + "word_to_states_dic.p")
            self.save_pickle_object(rel_state_to_num_dic, sub_directory + "rel_state_to_num_dic.p")
            self.save_pickle_object(num_to_rel_state_dic, sub_directory + "num_to_rel_state_dic.p")

            self.save_pickle_object(self.automaton, sub_directory + "automaton.p")
            self.save_pickle_object(self.list_of_relevant_states, sub_directory + "rel_states.p")

            for window in [1, 2, 3, 4, 5, 50]:
                word_co_occurrences = self.create_co_occurrences(plain_context,
                                                                 len(num_to_word_dic),
                                                                 word_to_num_dic,
                                                                 window)

                state_co_occurrences = self.create_co_occurrences(plain_context,
                                                                  len(self.list_of_relevant_states),
                                                                  word_to_rel_states_dic,
                                                                  window)

                sub_sub_directory = sub_directory + "window_of_" + str(window) + "/"

                try:
                    os.makedirs(sub_sub_directory)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                sub_sub_state_directory = sub_sub_directory + "state_level/"

                try:
                    os.makedirs(sub_sub_state_directory)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                sub_sub_word_directory = sub_sub_directory + "word_level/"

                try:
                    os.makedirs(sub_sub_word_directory)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                self.save_matrix(state_co_occurrences, sub_sub_state_directory + "state_co_occurrences.npy")
                self.save_matrix(word_co_occurrences, sub_sub_word_directory + "word_co_occurrences.npy")

            print("... co-occurrences for the most common " + str(top_n_words) + " are built.\n")

    def construct_automaton(self, word_list):
        # Following Daciuk et al (2000)
        # See Thesis for Details.

        alphabetically_sorted_words = sorted(word_list)

        num_of_words = len(alphabetically_sorted_words)
        word_count = 1

        for next_word in alphabetically_sorted_words:

            self.print_progress(word_count, num_of_words)
            word_count += 1

            common_prefix, last_state, current_suffix = self.common_prefix(next_word)

            if last_state in self.automaton:
                self.replace_or_register(last_state)

            self.add_suffix(last_state, current_suffix)

        self.replace_or_register(0)

    def common_prefix(self, word):

        prefix = ""
        last_state = 0
        next_state = 0

        while word and next_state != -1:

            last_state = next_state

            # If the current 'last state' is not final
            if last_state in self.automaton:

                next_state = self.automaton.get(last_state).get(word[0], -1)

                if next_state != -1:
                    prefix += word[0]
                    word = word[1::]

            # If last state is final
            else:
                next_state = -1

        return prefix, last_state, word

    def replace_or_register(self, state):

        child = self.get_last_child(state)

        if self.has_children(child):
            self.replace_or_register(child)

        merged = False

        for q in self.register:
            if self.equivalent(q, child):
                merged = self.set_last_child(state, q)

                # Deletes child from the automaton in case child is not a final state
                # And if child has been merged with a registered state
                # (Only if there do not exist two transitions outgoing from one state leading then to the same state)
                if child in self.automaton and merged:
                    del self.automaton[child]

                    if child in self.final_states:
                        self.final_states.remove(child)

                break

        if not merged:
            self.register.append(child)

    def equivalent(self, state_1, state_2):

        # Modified - two final states are equivalent only if they share the same incoming and outgoing transitions
        if state_1 not in self.automaton and state_2 not in self.automaton:

            state_1_incoming_transitions = []
            state_2_incoming_transitions = []

            for state in self.automaton:
                for char in self.automaton[state]:
                    if self.automaton[state][char] == state_1:
                        state_1_incoming_transitions.append(char)
                    if self.automaton[state][char] == state_2:
                        state_2_incoming_transitions.append(char)

            if len(state_1_incoming_transitions) == len(state_2_incoming_transitions):
                for transition in state_1_incoming_transitions:
                    if transition not in state_2_incoming_transitions:
                        return False
                return True
            else:
                return False

        # If both states have subsequent states
        # Make sure that those are the same
        if self.has_children(state_1) and self.has_children(state_2):
            if len(self.automaton[state_1]) == len(self.automaton[state_2]):

                for transition in self.automaton[state_1]:
                    if transition in self.automaton[state_2]:

                        # If there exists one transition with the same label that does not lead to the same state
                        if self.automaton[state_1][transition] != self.automaton[state_2][transition]:
                            return False

                    # If there exists one outgoing transition from state_1 that does not exist for state_2
                    else:
                        return False

                return True
            # If they do not share the same number of outgoing transitions
            else:
                return False
        # If one of both states does not have subsequent states
        else:
            return False

    def has_children(self, state):
        return state in self.automaton

    def get_last_child(self, state):

        # Return the subsequent state with the 'highest' lexicographic transition,
        # i.e. the one that has been added most recently
        return self.automaton[state][sorted(self.automaton[state].keys())[-1]]

    def set_last_child(self, state, registered_state):

        # Set the subsequent state with the 'highest' lexicographic transition,
        # i.e. the one that has been added most recently, to registered_state
        # Modified: There should not exist two outgoing transitions from one state leading to the same state

        merged = True

        for char in self.automaton[state]:
            if self.automaton[state][char] == registered_state:
                merged = False
        if merged:
            self.automaton[state][sorted(self.automaton[state].keys())[-1]] = registered_state

        return merged
        # self.automaton[state][sorted(self.automaton[state].keys())[-1]] = registered_state

    def add_suffix(self, last_state, current_suffix):

        # Add the remaining characters in the suffix to the automaton
        for char in current_suffix:

            if last_state in self.automaton:
                self.state_count += 1
                added_state = self.state_count
                self.automaton[last_state][char] = added_state
            else:
                self.state_count += 1
                added_state = self.state_count
                self.automaton[last_state] = {char: added_state}

            last_state = self.state_count

        # The last state added must be a final state
        self.final_states.append(last_state)

    def extract_rel_states(self):

        # Relevant states are the minimal number of states that need to be weighted in order to define a path
        # Therefore, only those states are selected, which are subsequent to a 'crossroad',
        # i.e. children to a state with multiple branches
        for state in self.automaton:
            if len(self.automaton[state]) > 1:

                for subsequent_state in list(self.automaton[state].values()):
                    if subsequent_state not in self.list_of_relevant_states:
                        self.list_of_relevant_states.append(subsequent_state)

        for state in self.final_states:
            if state in self.automaton:
                if state not in self.list_of_relevant_states:
                    self.list_of_relevant_states.append(state)

                for subsequent_state in (self.automaton[state].values()):
                    if subsequent_state not in self.list_of_relevant_states:
                        self.list_of_relevant_states.append(subsequent_state)

    def read_words(self, word_list):

        dictionary = {}

        for word in word_list:
            cur_state = 0
            rel_state_list = []

            # Collect all the states in the automaton that are worth weighting
            for char in word:
                next_state = self.automaton[cur_state][char]

                if next_state in self.list_of_relevant_states:
                    rel_state_list.append(next_state)

                cur_state = next_state

            dictionary[word] = rel_state_list

        return dictionary

    def create_co_occurrences(self, data, num_of_entries, word_dic, window):
        """
        Creates a numpy co-occurrence matrix based on the data, a word-to-entry dictionary, and a given window.
        :param data: nested dictionary of how often words co-occur with other words in a certain distance
        :param num_of_entries: number of
        :param word_dic:
        :param window:
        :return:
        """

        co_occurrences = np.zeros((num_of_entries, num_of_entries))

        # For every possibly target word in the corpus
        for target_word in data:

            # If the target word is among the predefined top k most common words
            if target_word in word_dic:

                # As long as the distance is in the range of the given window
                for distance in data[target_word]:
                    if distance > window:
                        break

                    # Calculate the weight (i.e., the inverse distance)
                    weight = 1.0 / distance

                    # Get all context words in that window
                    for context_word in data[target_word][distance]:

                        # If the context word is among the predefined top k most common words
                        if context_word in word_dic:

                            # Go over all entries of the target word
                            # Single entry on word level
                            # Multiple entries on finite state automaton level
                            for target_word_entry in word_dic[target_word]:

                                # Do the same for the context word
                                for context_word_entry in word_dic[context_word]:

                                    # The entry in the co-occurrence matrix is updated by the number of times
                                    # The target and context word appear together in that distance
                                    # Multiplied by the inverse of that distance
                                    co_occurrences[target_word_entry, context_word_entry] += weight * data[target_word][distance][context_word]

        return co_occurrences

    def save_matrix(self, matrix, dest):
        np.save(dest, matrix)

    def save_pickle_object(self, pickle_object, dest):
        pickle.dump(pickle_object, open(dest, "wb"))

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


ExtractData("/Path/to/Text_Corpus.txt",
            "/Path/to/")
