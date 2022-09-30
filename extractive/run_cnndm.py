import os
import argparse
from train import str2bool

DATASET = "CNNDM"


def run(inp_cmd):
    print(inp_cmd)
    os.system(inp_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "test", "eval", "preprocess"])
    parser.add_argument('--warmup', default=False, type=str2bool)
    parser.add_argument('--gpus', default="0")
    parser.add_argument('--batch_size', default=64, help=" change this according to your GPU memory size")
    parser.add_argument('--accum_count', default=1, help=" used to simulate large batch size ")
    parser.add_argument('--validate_every', default=800, type=int)
    parser.add_argument('--ext_num', default=5, type=int)
    # no need to set in training mode
    parser.add_argument('--save_path', default="")  # checkpoints/CNNDM
    args = parser.parse_args()

    # ext_num for colo is the candidate sentences size
    if args.mode != "train":
        test_cmd = f"python train.py --gpus {args.gpus} --dataset {DATASET} " \
                   f" --batch_size 1  --mode {args.mode} " \
                   f" --save_path {args.save_path} --ext_num {args.ext_num} "
        run(test_cmd)
    else:
        if args.warmup:
            wp_cmd = f"python train.py --mode train  --dataset {DATASET} --batch_size {args.batch_size} --warmup True --gpus {args.gpus}  " \
                     f" --n_epochs 2 --ext_num 3  --validate_every {args.validate_every} "
            run(wp_cmd)

        train_cmd = f"python train.py --mode train  --dataset {DATASET} --batch_size {args.batch_size} --warmup False  --gpus {args.gpus}  " \
                    f" --n_epochs 5 --ext_num {args.ext_num}  --save_path checkpoints/{DATASET} --validate_every {args.validate_every // 2} "
        print("========== make sure there is a warmed-up `checkpoint warmup/CNNDM/pretrain.ext.pt` =========")
        run(train_cmd)
