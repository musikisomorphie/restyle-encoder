CUDA_VISIBLE_DEVICES=2 taskset -c 64-95 python -m scripts.train_restyle_psp \
--rna=tabular \
--rna_num=61 \
--dataset_type=Visium \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Data/Visium/GAN/e0_64 \
--max_steps=800000 \
--workers=8 \
--batch_size=16 \
--test_batch_size=8 \
--test_workers=8 \
--board_interval=500 \
--val_interval=50000 \
--save_interval=100000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0.5 \
--l2_lambda=20 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=3 \
--input_ch=-1 \
--n_iters_per_batch=1 \
--output_size=128 \
--stylegan_weights=Data/Visium/GAN/d0/checkpoint/400000.pt