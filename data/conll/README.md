# CoNLL03 Dataset

The original dataset is presented in the following paper:

Erik F. Tjong Kim Sang and Fien De Meulder. "Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition." In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003.

We provide eight files: 
* `{train/valid/test}_text.txt` are the text files where each line is one text sequence, and words are separated by whitespace characters. 
* `{train/valid/test}_label_true.txt` are the ground truth label files where each line is one sequence of labels for the corresponding sequence in the text file; they are not used for model training and are provided for completeness and evaluation purpose.
* `train_label_dist.txt` is the distant training label file as in the [BOND paper](https://arxiv.org/abs/2006.15509) and the original file can be found [here](https://github.com/cliang1453/BOND/tree/master/dataset).
* `types.txt` contains the entity types in the dataset; it does not include the `O` class or prefixes (e.g., `I-`/`B-`).
