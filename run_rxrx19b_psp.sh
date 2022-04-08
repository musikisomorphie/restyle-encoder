python scripts/train_restyle_psp.py \
--dataset_type=rxrx19b \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=/home/jwu/Experiment/non_IID/rxrx19b_psp \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=50 \
--w_norm_lambda=0 \
--moco_lambda=0.5 \
--input_nc=6 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=/home/jwu/Data/non_IID_old/encoder/rxrx19b_cell/checkpoint/010000.pt