Embeddings.py
Calculates Embeddings for given Co-Occurrence Matrix

ExtractData.py
Extracts {500, 1000, 2000} most common Words from a .txt Corpus,
constructs a minimal, acyclic FSA which accepts those as Language, 
restricts the Number of States to the relevant ones, and builds a 
Co-Occurrence Matrix for both the States and the Words for
a Context-Window of Size {1,2,4,50}.

Similarity.py
Calculates a Similarity Matrix between two Embedding Matrices.