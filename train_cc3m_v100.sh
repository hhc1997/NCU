python -m src.main \
  --name NCU_CC3M_F \
  --batch_size 2048 \
  --train_data /mnt/NCU/data/cc3m/train.csv \
  --eval_data_type ImageNet1K \
  --eval_test_data_dir /mnt/NCU/data/ImageNet1K/validation \
  --image_key image_path \
  --caption_key text_content \
  --device_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --model_name CLIP_VITB16 \
  --checkpoint /mnt/NCU/pretrained_model/LaCLIP_Provided/vitb16_cc3m_clip.pt \
  --lr_UL 5e-5 \
  --lr_pre 1e-4 \
  --epochs 10 \
  --lr_HN 3e-4 \
  --NC_ratio 0.1 \
  --weight_decay 0.2 \
  --weight_decay_prompt 0.2 \
  --distributed



