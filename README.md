# RoSTER

The source code used for [**Distantly-Supervised Named Entity Recognition with Noise-Robust Learning and Language Model Augmented Self-Training**](https://arxiv.org/abs/2109.05003), published in EMNLP 2021.

## Requirements

At least one GPU is required to run the code.

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

Python 3.6 or above is strongly recommended; using older python versions might lead to package incompatibility issues.

## Reproducing the Results

The three datasets used in the paper can be found under the [`data`](data) directory. We provide three bash scripts [`run_conll.sh`](run_conll.sh), [`run_onto.sh`](run_onto.sh) and [`run_wikigold.sh`](run_wikigold.sh) for running the model on the three datasets.

**Note: Our model does not use any ground truth training/valid/test set labels but only distant labels; we provide the ground truth label files only for completeness and evaluation.**

The training bash scripts assume you use one GPU for training (a GPU with around 20GB memory would be sufficient). If your GPUs have smaller memory sizes, try increasing `gradient_accumulation_steps` or using more GPUs (by setting the `CUDA_VISIBLE_DEVICES` environment variable). However, the `train_batch_size` should be always kept as `32`.

## Command Line Arguments

The meanings of the command line arguments will be displayed upon typing
```
python src/train.py -h
```
The following arguments are important and need to be set carefully:

* `train_batch_size`: The **effective** training batch size **after** gradient accumulation. Usually `32` is good for different datasets.
* `gradient_accumulation_steps`: Increase this value if your GPU cannot hold the training batch size (while keeping `train_batch_size` unchanged).
* `eval_batch_size`: This argument only affects the speed of the algorithm; use as large evaluation batch size as your GPUs can hold.
* `max_seq_length`: This argument controls the maximum length of sequence fed into the model (longer sequences will be truncated). Ideally, `max_seq_length` should be set to the length of the longest document (`max_seq_length` cannot be larger than `512` under RoBERTa architecture), but using larger `max_seq_length` also consumes more GPU memory, resulting in smaller batch size and longer training time. Therefore, you can trade model accuracy for faster training by reducing `max_seq_length`.
* `noise_train_epochs`, `ensemble_train_epochs`, `self_train_epochs`: They control how many epochs to train the model for noise-robust training, ensemble model training and self-training, respectively. Their default values will be a good starting point for most datasets, but you may increase them if your dataset is small (e.g., `Wikigold` dataset) and decrease them if your dataset is large (e.g., `OntoNotes` dataset).
* `q`, `tau`: Hyperparameters used for noise-robust training. Their default values will be a good starting point for most datasets, but you may use higher values if your dataset is more noisy and use lower values if your dataset is cleaner.
* `noise_train_update_interval`, `self_train_update_interval`: They control how often to update training label weights in noise-robust training and compute soft labels in soft-training, respectively. Their default values will be a good starting point for most datasets, but you may use smaller values (more frequent updates) if your dataset is small (e.g., `Wikigold` dataset).

Other arguments can be kept as their default values.

## Running on New Datasets

To execute the code on a new dataset, you need to 

1. Create a directory named `your_dataset` under `data`.
2. Prepare a training corpus `train_text.txt` (one sequence per line; words separated by whitespace) and the corresponding distant label `train_label_dist.txt` (one sequence per line; labels separated by whitespace) under `your_dataset` for training the NER model.
3. Prepare an entity type file `types.txt` under `your_dataset` (each line contains one entity type; no need to include `O` class; no need to prepend `I-`/`B-` to type names). The entity type names need to be consistant with those in `train_label_dist.txt`.
4. (Optional) You can choose to provide a test corpus `test_text.txt` (one sequence per line) with ground truth labels `test_label_true.txt` (one sequence per line; labels separated by whitespace). If the test corpus is provided and the command line argument `do_eval` is turned on, the code will display evaluation results on the test set during training, which is useful for tuning hyperparameters and monitoring the training progress.
5. Run the code with appropriate command line arguments (I recommend creating a new bash script by referring to the three example scripts).
6. The final trained classification model will be saved as `final_model.pt` under the output directory specified by the command line argument `output_dir`.

You can always refer to the example datasets when preparing your own datasets.

## Citations

Please cite the following paper if you find the code helpful for your research.
```
@inproceedings{meng2021distantly,
  title={Distantly-Supervised Named Entity Recognition with Noise-Robust Learning and Language Model Augmented Self-Training},
  author={Meng, Yu and Zhang, Yunyi and Huang, Jiaxin and Wang, Xuan and Zhang, Yu and Ji, Heng and Han, Jiawei},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year={2021},
}
```
