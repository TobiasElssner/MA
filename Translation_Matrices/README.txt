The Sub-Directories contain the Alignments for the top {500, 1000, 2000} Words, containing
	- Sub-Directories for the {1, 2, 4, 50} Context-Size, containing
		- Sub-Directories for {state, word} Embeddings, containing			
			- Sub-Directories for {5, 10, 15, 20} Embedding Size, containing
				-  A .txt-file about the Iterations until convergence
				- The Similarity Matrix for embeddings, emb.npy
				- The Similarity Matrix for Words, word_.npy
				
Only Embeddings for the same number of most-common Words, of the same Type, 
the same Context-Size, and the same Embedding Size are translated into each other.

