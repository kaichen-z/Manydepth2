CUDA_VISIBLE_DEVICES=1  python -u -m manydepth2.train \
    --data_path /root/autodl-tmp/kai/data/1_mgdepth/KITTI \
    --log_dir logs  \
    --png \
    --freeze_teacher_epoch 10 \
    --model_name models_many2 \
    --pytorch_random_seed 1 \
    --batch_size 12 \
    --mode many2 