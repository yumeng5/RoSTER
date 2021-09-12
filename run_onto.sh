export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

CORPUS=onto
SEED=30
TEMP_DIR=tmp_${CORPUS}_$SEED
OUT_DIR=out_$CORPUS
mkdir -p $TEMP_DIR
mkdir -p $OUT_DIR

python -u src/train.py --data_dir data/$CORPUS --output_dir $OUT_DIR --temp_dir $TEMP_DIR \
    --pretrained_model roberta-base --tag_scheme 'io' --max_seq_length 180 \
    --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 64 \
    --noise_train_lr 3e-5 --ensemble_train_lr 1e-5 --self_train_lr 5e-7 \
    --noise_train_epochs 3 --ensemble_train_epochs 2 --self_train_epochs 1 \
    --noise_train_update_interval 200 --self_train_update_interval 100 \
    --dropout 0.1 --warmup_proportion 0.1 --seed $SEED \
    --q 0.7 --tau 0.7 --num_models 5 \
    --do_train --do_eval --eval_on "test" | tee $OUT_DIR/train_log.txt
