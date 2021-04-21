import argparse

parser = argparse.ArgumentParser()

def parse_bool(x):
    assert isinstance(x, str)
    if x == "True":
        return True
    elif x == "False":
        return False
    else: 
        assert False, "Arg must be 'True' or 'False'"

# model arguments
parser.add_argument('--img_size', type=int, default=256,
                    help='Image resolution')
parser.add_argument('--num_domains', type=int, default=2,
                    help='Number of domains')
parser.add_argument('--latent_dim', type=int, default=16,
                    help='Latent vector dimension')
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='Hidden dimension of mapping network')
parser.add_argument('--style_dim', type=int, default=64,
                    help='Style code dimension')

# weight for objective functions
parser.add_argument('--lambda_reg', type=float, default=1,
                    help='Weight for R1 regularization')
parser.add_argument('--lambda_cyc', type=float, default=1,
                    help='Weight for cyclic consistency loss')
parser.add_argument('--lambda_sty', type=float, default=1,
                    help='Weight for style reconstruction loss')
parser.add_argument('--lambda_ds', type=float, default=1,
                    help='Weight for diversity sensitive loss')
parser.add_argument('--ds_iter', type=int, default=51000, #100000
                    help='Number of iterations to optimize diversity sensitive loss')
parser.add_argument('--w_hpf', type=float, default=1,
                    help='weight for high-pass filtering')

# training arguments
parser.add_argument('--randcrop_prob', type=float, default=0.5,
                    help='Probabilty of using random-resized cropping')
parser.add_argument('--total_iters', type=int, default=51000, #100000
                    help='Number of total iterations')
parser.add_argument('--resume_iter', type=int, default=0,
                    help='Iterations to resume training/testing')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for training')
parser.add_argument('--val_batch_size', type=int, default=32,
                    help='Batch size for validation')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate for D, E and G')
parser.add_argument('--f_lr', type=float, default=1e-6,
                    help='Learning rate for F')
parser.add_argument('--beta1', type=float, default=0.0,
                    help='Decay rate for 1st moment of Adam')
parser.add_argument('--beta2', type=float, default=0.99,
                    help='Decay rate for 2nd moment of Adam')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay for optimizer')
parser.add_argument('--num_outs_per_domain', type=int, default=10,
                    help='Number of generated images per domain during sampling')

# misc
parser.add_argument('--mode', type=str, required=True,
                    choices=['train', 'sample', 'eval', 'align'],
                    help='This argument is used in solver')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers used in DataLoader')
parser.add_argument('--seed', type=int, default=777,
                    help='Seed for random number generator')

parser.add_argument('--init_seed', type=int, default=777,
                    help='Seed for random number generator')
parser.add_argument('--train_seed', type=int, default=777,
                    help='Seed for random number generator')
parser.add_argument('--eval_seed', type=int, default=777,
                    help='Seed for random number generator')

# directory for training
parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                    help='Directory containing training images')
parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                    help='Directory containing validation images')
parser.add_argument('--sample_dir', type=str, default='expr/samples',
                    help='Directory for saving generated images')
parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                    help='Directory for saving network checkpoints')

# directory for calculating metrics
parser.add_argument('--eval_dir', type=str, default='expr/eval',
                    help='Directory for saving metrics, i.e., FID and LPIPS')

# directory for testing
parser.add_argument('--result_dir', type=str, default='expr/results',
                    help='Directory for saving generated images and videos')
parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                    help='Directory containing input source images')
parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                    help='Directory containing input reference images')
parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                    help='input directory when aligning faces')
parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                    help='output directory when aligning faces')

# face alignment
parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

# step size
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--sample_every', type=int, default=5000)
parser.add_argument('--save_every', type=int, default=50000)
parser.add_argument('--eval_every', type=int, default=25000)

parser.add_argument('--print_learned', type=int, default=1000)
parser.add_argument('--print_std', type=int, default=1000)
parser.add_argument('--print_sqrt_error', type=int, default=1000)
parser.add_argument('--print_color', type=int, default=1000)
parser.add_argument('--alpha_white', type=float, default=1.0)
parser.add_argument('--alpha_color', type=float, default=1.0)
parser.add_argument('--notes_path', type=str, default='expr/')

# Vanilla
parser.add_argument('--method', type=str, default='whitening', choices=['whitening', 'std', 'baseline', 'ortho'])
parser.add_argument('--use_mlp', type=parse_bool, default=False)
parser.add_argument('--learn_alpha_white', type=parse_bool, default=False)
parser.add_argument('--learn_alpha_color', type=parse_bool, default=False)
parser.add_argument('--normalizer_type', type=str, default='BlockwiseWhitening')
parser.add_argument('--num_whitening_iters', type=int, default=10)
parser.add_argument('--use_mean_shift', type=parse_bool, default=False)
parser.add_argument('--make_color_symmetric', type=parse_bool, default=True)
parser.add_argument('--make_positive_definite', type=parse_bool, default=True)
parser.add_argument('--center_color_at_identity', type=parse_bool, default=False)
parser.add_argument('--block_size', type=int, default=64)
parser.add_argument('--num_blocks', type=int, default=-1)

import sys
import json
import os

ARGS = parser.parse_args()

#save args as json file
os.makedirs(ARGS.eval_dir, exist_ok=True)
file_name = os.path.join(ARGS.eval_dir, "args.json")
    with open(file_name, "wt") as f:
            json.dump(vars(ARGS), f, indent=4)
            
print(sys.argv)
print(ARGS)