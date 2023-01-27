CUDA_VISIBLE_DEVICES=0 python -m scripts.train_restyle_psp \
--dataset_type=NSTG \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Data/NSTG/GAN/encoder0 \
--max_steps=800000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=50000 \
--save_interval=100000 \
--image_interval=50000 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=40 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=6 \
--input_ch=-1 \
--n_iters_per_batch=1 \
--output_size=128 \
--stylegan_weights=Data/NSTG/GAN/decoder0/checkpoint/400000.pt

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_restyle_psp \
--dataset_type=NSTG \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Data/NSTG/GAN/encoder1 \
--max_steps=800000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=50000 \
--save_interval=100000 \
--image_interval=50000 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=40 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=6 \
--input_ch=-1 \
--n_iters_per_batch=1 \
--output_size=128 \
--stylegan_weights=Data/NSTG/GAN/decoder1/checkpoint/400000.pt