README

I dump a list of objectc to binary file using 'pickle' library

To load a file:

import _pickle #python 3

with open('training_binary', 'rb') as infile:
	training = _pickle.load(infile)
	
Structure:
Each element of list contains 4 sublists:
[0] - a list of acronym[0] and long form[1]
[1] - sentence that conitains acronym and long form
[2] - list of arrays - candidates for a long form
[3] - list of features vectors for earch candidates

features in order are:
[0] #The ratio of words starting from acronym letters
[1] #The ratio of letters from acronym appearing in order in the full long form:
[2] #The ratio of letters from acronym appearing in order in the first words of long form:
[3] #The ratio of # of the letter in acronym to # of words in the long form
[4] #Number of stop words
[5] #The ratio of # of digits in long form to # of digits in the acronym
[6] LABER - 1 is a LF, 0 is not a LF