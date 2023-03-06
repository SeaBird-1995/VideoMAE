# Set the path to save checkpoints
OUTPUT_DIR='experiments/small_one_224'
DATA_PATH='datasets/train_list2.txt'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 12320 --nnodes=1 --node_rank=0 \
        vipc_run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.125 \
        --model pretrain_vipc_self_distillation \
        --decoder_depth 4 \
        --batch_size 16 \
        --num_frames 8 \
        --sampling_rate 4 \
        --num_workers 32 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 5 \
        --save_ckpt_freq 5 \
        --epochs 101 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}