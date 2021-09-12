# OntoNotes5.0 Dataset

The original dataset is presented in the following paper and can be downloaded [here](https://catalog.ldc.upenn.edu/LDC2013T19):

Ralph Weischedel, Martha Palmer, Mitchell Marcus, Eduard Hovy, Sameer Pradhan, Lance Ramshaw, Nianwen Xue et al. "Ontonotes release 5.0 ldc2013t19." Linguistic Data Consortium, Philadelphia, PA (2013).

We provide eight files with the same preprocessing as in [this paper](https://arxiv.org/abs/1511.08308): 
* `{train/valid/test}_text.txt` are the text files where each line is one text sequence, and words are separated by whitespace characters. 
* `{train/valid/test}_label_true.txt` are the ground truth label files where each line is one sequence of labels for the corresponding sequence in the text file; they are not used for model training and are provided for completeness and evaluation purpose.
* `train_label_dist.txt` is the distant training label file as in the [BOND paper](https://arxiv.org/abs/2006.15509) and the original file can be found [here](https://github.com/cliang1453/BOND/tree/master/dataset).
* `types.txt` contains the entity types in the dataset; it does not include the `O` class or prefixes (e.g., `I-`/`B-`).
