
# Set the path to save video
OUTPUT_DIR='experiments/vipc_test'
# path to video for visualization
VIDEO_PATH='TODO/TODO.mp4'
# path to pretrain model
MODEL_PATH='experiments/videomae_pretrain_base_patch16_224_mask_one_view/checkpoint-9.pth'

python run_vipc_mae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_vipc_mae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}