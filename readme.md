# Character level Acronym Detection using LSTM

Requirements:

- pandas

- tensorflow 1.4.1

- keras 2.1.5

- numpy 

- sklearn

- matplotlib

To Run the RNN and CNN codes:
python PYTHONFILE FileName (example python CNN.py transform_all_2char_L_uniqe3)

Data Description:
L --> Left (Begining)
LR --> Left and Right (Begining and End)
uniqe1 --> Unique Words
uniqe2 --> Unique Tokens
uniqe3 --> Unique Structure

2char_L_uniqe3 - Constructed with 2 characters to the left (begining) for Unique Transformation Data

2char_LR_uniqe1 - Constructed with 2 characters to the left(begining) and right(end) for Unique Word Data

-------------------------------------------------------------------
acronyms.py
-------------------------------------------------------------------

Required packages:
nltk
operator
bioc
numpy
Bio 
string
jellyfish
sklearn
pickle

Rest:

Datasets used for testing include:
"medstract_bioc_gold.xml", "Ab3P_bioc_gold.xml",
"bioadi_bioc_gold.xml", "SH_bioc_gold.xml" datasets in BioC format.
Download data from
https://sourceforge.net/p/bioc/blog/2014/06/i-am-looking-for-the-abbreviation-definition-corpus-where-can-i-find-it/
and place them in /data folder


Acronym detection using Decision Trees,
Call function:
decition_tree_accronym()
to run 5-fold CV testing decision trees methods.


Long form detection


1. Long form detection

Candidates and features generation from a given set:

LF_DL( datasets )

For details - look on comments in the code



2. Supporting methods with low-performance results

Acronym detection using Decision Trees:

decition_tree_accronym(datasets)



Long Form identification: 

Method #1 - simple rules based method, call function: long_form_detection( datasets )

Method #2 - naive NLP, call function long_form_detection_tagging( datasets )

Where datasets argument is a name(string) of datafile sets in BioC format, 
eg. 
datasets = ["medstract_bioc_gold.xml", "Ab3P_bioc_gold.xml", "bioadi_bioc_gold.xml", "SH_bioc_gold.xml"]
