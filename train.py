import os

import pytorch_lightning as pl
import torch

from lib.model import PatchCore


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    parser.add_argument('--dataset_path', default=r'C:\Users\Frailty\Desktop\py\recode\github_code\data\Surface-Defect-Detection\mvtec')
    parser.add_argument('--category', default='magnetic_tile')
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.001)
    parser.add_argument('--project_root_path', default=r'./test')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=False)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category),
                                            max_epochs=args.num_epochs, gpus=1)
    model = PatchCore(hparams=args)
    model = model.to(device)
    # model.load_from_checkpoint()
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)
