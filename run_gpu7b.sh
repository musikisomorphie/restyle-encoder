# python scripts/train_restyle_psp.py \
# --dataset_type=rxrx19b_HUVEC \
# --encoder_type=ResNetBackboneEncoder \
# --exp_dir=Experiment/non_IID/rxrx19b_HUVEC \
# --max_steps=800000 \
# --workers=8 \
# --batch_size=8 \
# --test_batch_size=8 \
# --test_workers=8 \
# --val_interval=5000 \
# --save_interval=10000 \
# --image_interval=10000 \
# --start_from_latent_avg \
# --lpips_lambda=0 \
# --l2_lambda=50 \
# --moco_lambda=0.5 \
# --w_norm_lambda=0 \
# --input_nc=12 \
# --input_ch=-1 \
# --n_iters_per_batch=1 \
# --output_size=128 \
# --train_decoder=False \
# --stylegan_weights=Data/non_IID/encoder/rxrx19b_HUVEC/checkpoint/790000.pt

python scripts/train_restyle_psp.py \
--dataset_type=rxrx19b_HUVEC \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=Experiment/non_IID/rxrx19b_HUVEC_chn5 \
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
--input_ch=5 \
--n_iters_per_batch=1 \
--output_size=128 \
--train_decoder=False \
--stylegan_weights=Data/non_IID/encoder/rxrx19b_HUVEC_chn5/checkpoint/790000.pt