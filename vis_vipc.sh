
# Set the path to save video
OUTPUT_DIR='experiments/self_distillation_vipc_test_epoch_89'
# path to video for visualization
VIDEO_PATH='TODO/TODO.mp4'
# path to pretrain model
# MODEL_PATH='experiments/videomae_pretrain_base_patch16_224_mask_one_view/checkpoint-9.pth'
MODEL_PATH="experiments/normalize_vipc_mae_pretrain_base_patch16_224_mask_one_view_total_epoch/checkpoint-84.pth"

MODEL_PATH="experiments/small_one_224/checkpoint-89.pth"

python run_vipc_mae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_vipc_self_distillation \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}