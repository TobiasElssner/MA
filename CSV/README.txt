conv.csv:                      						Contains Iteration until Convergence for given Parameters Vocabulary Size, Context Size, Embedding Mode, Embedding Size

{deu, eng}_vocab.csv:						Contains Word Forms from the Test Sets, plus POS, Question Type, English Lemma, and the Vocabulary they are in.

{deu, eng}_q.csv:                         				Contains the Questions from the Test Set, plus Question Type,  and the Vocabulary they are in.

{deu, eng}_results.csv:             				Contains the Results of the Monolingual Embedding Vectors, plus Vocabulary Size , Context Size , Embedding Mode , Embedding Size , Question 		
											Type , Answer Vocabulary  Size , Question , (Average Question Type) Relative Rank , and Answer.

{deu, eng}_to_{deu, eng}_results.csv: 	Contains the Results of the Translations, plus Vocabulary Size, Context Size, Embedding Mode, Embedding Size, POS Tag, English Lemma, Word 
											Form, Translation Vocabulary Size, Closest Correct Translation, Closest Correct Translation Vocabulary Size, (Average Lemma) Relative Rank, Actual 											Closest Translation, and Average POS Relative Rank 


All Context Sizes are denoted for one Direction, thus must be multiplied by two in Order to get the whole Window Size.