python scripts/train_restyle_psp.py \
--dataset_type=rxrx19a_HRCE \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Experiment/non_IID/rxrx19a_HRCE \
--max_steps=800000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=50 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=10 \
--input_ch=-1 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/rxrx19a_HRCE/checkpoint/790000.pt

python scripts/train_restyle_psp.py \
--dataset_type=rxrx19a_HRCE \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Experiment/non_IID/rxrx19a_HRCE_chn2 \
--max_steps=800000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=50 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=2 \
--input_ch=2 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/rxrx19a_HRCE_chn2/checkpoint/790000.pt

python scripts/train_restyle_psp.py \
--dataset_type=rxrx19a_VERO \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Experiment/non_IID/rxrx19a_VERO_chn2 \
--max_steps=800000 \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0 \
--l2_lambda=50 \
--moco_lambda=0.5 \
--w_norm_lambda=0 \
--input_nc=2 \
--input_ch=2 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/rxrx19a_VERO_chn2/checkpoint/390000.pt