# # 90000.pt
# python scripts/train_restyle_e4e.py \
# --dataset_type=ham10k \
# --encoder_type=ResNetProgressiveBackboneEncoder \
# --exp_dir=Experiment/non_IID/ham10k_e4e_9 \
# --max_steps=100000 \
# --workers=8 \
# --batch_size=8 \
# --test_batch_size=8 \
# --test_workers=8 \
# --val_interval=5000 \
# --save_interval=10000 \
# --image_interval=10000 \
# --start_from_latent_avg \
# --lpips_lambda=0 \
# --l2_lambda=10 \
# --moco_lambda=0.5 \
# --delta_norm_lambda=0.0002 \
# --use_w_pool \
# --w_discriminator_lambda=0.1 \
# --progressive_start=20000 \
# --progressive_step_every=2000 \
# --input_nc=6 \
# --n_iters_per_batch=1 \
# --output_size=128 \
# --train_decoder=False \
# --stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/090000.pt

# python scripts/train_restyle_psp.py \
# --dataset_type=ham10k \
# --encoder_type=ResNetBackboneEncoder \
# --exp_dir=Experiment/non_IID/ham10k_psp_9 \
# --max_steps=100000 \
# --workers=8 \
# --batch_size=8 \
# --test_batch_size=8 \
# --test_workers=8 \
# --val_interval=5000 \
# --save_interval=10000 \
# --image_interval=10000 \
# --start_from_latent_avg \
# --lpips_lambda=0 \
# --l2_lambda=10 \
# --moco_lambda=0.5 \
# --w_norm_lambda=0 \
# --input_nc=6 \
# --n_iters_per_batch=1 \
# --output_size=128 \
# --train_decoder=False \
# --stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/090000.pt

# 70000.pt
python scripts/train_restyle_e4e.py \
--dataset_type=ham10k \
--encoder_type=ResNetProgressiveBackboneEncoder \
--exp_dir=Experiment/non_IID/ham10k_e4e_7 \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=10 \
--moco_lambda=0.5 \
--delta_norm_lambda=0.0002 \
--use_w_pool \
--w_discriminator_lambda=0.1 \
--progressive_start=20000 \
--progressive_step_every=2000 \
--input_nc=6 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/070000.pt

python scripts/train_restyle_psp.py \
--dataset_type=ham10k \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Experiment/non_IID/ham10k_psp_7 \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=10 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=6 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/070000.pt

# 50000.pt
python scripts/train_restyle_e4e.py \
--dataset_type=ham10k \
--encoder_type=ResNetProgressiveBackboneEncoder \
--exp_dir=Experiment/non_IID/ham10k_e4e_5 \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=10 \
--moco_lambda=0.5 \
--delta_norm_lambda=0.0002 \
--use_w_pool \
--w_discriminator_lambda=0.1 \
--progressive_start=20000 \
--progressive_step_every=2000 \
--input_nc=6 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/050000.pt

python scripts/train_restyle_psp.py \
--dataset_type=ham10k \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Experiment/non_IID/ham10k_psp_5 \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=10 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=6 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/050000.pt

# 20000.pt
python scripts/train_restyle_e4e.py \
--dataset_type=ham10k \
--encoder_type=ResNetProgressiveBackboneEncoder \
--exp_dir=Experiment/non_IID/ham10k_e4e_2 \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=10 \
--moco_lambda=0.5 \
--delta_norm_lambda=0.0002 \
--use_w_pool \
--w_discriminator_lambda=0.1 \
--progressive_start=20000 \
--progressive_step_every=2000 \
--input_nc=6 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/020000.pt

python scripts/train_restyle_psp.py \
--dataset_type=ham10k \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Experiment/non_IID/ham10k_psp_2 \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=10 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=6 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/020000.pt

# 00000.pt
python scripts/train_restyle_e4e.py \
--dataset_type=ham10k \
--encoder_type=ResNetProgressiveBackboneEncoder \
--exp_dir=Experiment/non_IID/ham10k_e4e_0 \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=10 \
--moco_lambda=0.5 \
--delta_norm_lambda=0.0002 \
--use_w_pool \
--w_discriminator_lambda=0.1 \
--progressive_start=20000 \
--progressive_step_every=2000 \
--input_nc=6 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/000000.pt

python scripts/train_restyle_psp.py \
--dataset_type=ham10k \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Experiment/non_IID/ham10k_psp_0 \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=10 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=6 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/ham10k_tiny/checkpoint/000000.pt