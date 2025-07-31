python -m src.main \
  --name eval_NCU_EuroSAT \
  --eval_data_type EuroSAT \
  --eval_test_data_dir /mnt/data/EuroSAT/test \
  --batch_size 512 \
  --device_ids 1 \
  --num_workers 4 \
  --checkpoint /mnt/NCU/logs/NCU_CC3M/checkpoints/epoch_5.pt \
  --distributed
