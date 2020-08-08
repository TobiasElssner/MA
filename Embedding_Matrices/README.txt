The {deu, eng}-Directories are organized as follows:
	- Sub-Directories for the top {500, 1000, 2000} Words, containing:
		- A Dictionary assigning each Word to its (Row)-Number in the Embedding Matrix, word_to_num_dic.p,
		- A Dictionary assigning each Row)-Number in the Embedding Matrix to a word, num_to_word_dic.p,
		- A Dictionary mapping each Word to its finite States, word_to_states_dic.p,  
		- A List of relevant States, rel_States.p, 
		- A Dictionary mapping each relevant State to its (Row)-Number in the Embedding Matrix, rel_state_to_num_dic.p,		
		- A Dictionary mapping each Word to its relevant States,  word_to_rel_states_dic.p
		- An FSA accepting the top_{500, 1000, 2000}_words (depending on the Sub-Directory), stored as Dictionary, mapping State Numbers to
		  Dictionaries with Characters as Key and subsequent State Numbers as Values, and
		- Sub-Directories for {1, 2, 4, 50} Context-Sizes, containing 
			- Sub-Directories for {state, word} Embeddings, containing			
				- The Co-Occurrence Matrix, {word, state}_co_occurrences.npy
				- Sub-Directories for {5, 10, 15, 20} Embedding Size, containing
					- Actual Embedding Matrices, embedding_{5, 10, 15, 20}_matrix.npy

All "*.p" Files are pickled lists or dictionaries, using python3.6.
All "*.npy" Files use the current numpy-library for python3.6
Context-Sizes denote the Window Size in one Direction, thus need to be multiplied by two.