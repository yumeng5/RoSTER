import argparse
import shutil
from trainer import RoSTERTrainer

def main():
    parser = argparse.ArgumentParser()

    # data preparation parameters
    parser.add_argument("--data_dir",
                        default="conll",
                        type=str,
                        help="The dataset directory.")
    parser.add_argument("--pretrained_model", 
                        default='roberta-base', 
                        type=str,
                        help="Pre-trained language model, default to roberta base.")
    parser.add_argument('--temp_dir',
                        type=str,
                        default="tmp",
                        help="temporary directory for saved models")
    parser.add_argument("--output_dir",
                        default='out',
                        type=str,
                        help="The output directory where the final model checkpoint will be saved.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum input sequence length.")
    parser.add_argument("--tag_scheme",
                        default='iob',
                        type=str,
                        choices=['iob', 'io'],
                        help="The tagging scheme used.")

    # training settting parameters
    parser.add_argument("--do_train",
                        action='store_true',
                        help="whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="whether to run eval on eval set or not.")
    parser.add_argument("--eval_on",
                        default="test",
                        choices=['valid', 'test'],
                        help="run eval on valid/test set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="effective batch size for training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="batch size for eval.")
    parser.add_argument("--noise_train_update_interval",
                        default=200,
                        type=int,
                        help="number of batches to periodically perform noisy label removal for noise robust training.")
    parser.add_argument("--self_train_update_interval",
                        default=100,
                        type=int,
                        help="number of batches to periodically compute new soft labels for self-training.")
    parser.add_argument("--noise_train_lr",
                        default=3e-5,
                        type=float,
                        help="the peak learning rate for noise robust training.")
    parser.add_argument("--ensemble_train_lr",
                        default=1e-5,
                        type=float,
                        help="the peak learning rate for ensemble model training.")
    parser.add_argument("--self_train_lr",
                        default=5e-7,
                        type=float,
                        help="the peak learning rate for self-training.")
    parser.add_argument("--noise_train_epochs",
                        default=3,
                        type=int,
                        help="total number of training epochs for noise robust training.")
    parser.add_argument("--ensemble_train_epochs",
                        default=2,
                        type=int,
                        help="total number of training epochs for ensemble model training.")
    parser.add_argument("--self_train_epochs",
                        default=5,
                        type=int,
                        help="total number of training epochs for self-training.")
    parser.add_argument("--q",
                        default=0.7,
                        type=float,
                        help="the hyperparameter of GCE loss. Larger value means higher tolerance to noise (for noisy data). Smaller value means better convergence (for clean data).")
    parser.add_argument("--tau",
                        default=0.7,
                        type=float,
                        help="the threshold for noisy label removal.")
    parser.add_argument("--num_models",
                        default=5,
                        type=int,
                        help="total number of models to ensemble.")
    parser.add_argument("--warmup_proportion", 
                        default=0.1,
                        type=float, 
                        help="proportion of learning rate warmup.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="weight decay for model training.")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="dropout ratio")
    parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help="number of gpus to use")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for training")
    
    args = parser.parse_args()

    print(args)

    if args.do_train:

        # train K models for ensemble
        for i in range(args.num_models):
            trainer = RoSTERTrainer(args)
            trainer.noise_robust_train(i)
            args.seed = args.seed + 1
        
        # ensemble K model predictions and train an ensembled model
        trainer = RoSTERTrainer(args)
        trainer.ensemble_pred(trainer.temp_dir)
        trainer.ensemble_train()
        
        # self-training
        trainer.self_train()

        shutil.rmtree(trainer.temp_dir, ignore_errors=True)

    if args.do_eval:
        trainer.load_model("final_model.pt", args.output_dir)
        y_pred, _ = trainer.eval(trainer.model, trainer.eval_dataloader)
        trainer.performance_report(trainer.y_true, y_pred)


if __name__ == "__main__":
    main()
