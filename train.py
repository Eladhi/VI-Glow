import sys
import signal
import argparse

from torchvision import transforms, datasets

from misc import util
from network import Builder, Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of "Glow: Generative Flow with Invertible 1x1 Convolutions"')
    parser.add_argument('profile', type=str,
                        default='profile/patch16.json',
                        help='path to profile file')
    return parser.parse_args()


if __name__ == '__main__':
    # this enables a Ctrl-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    # parse arguments
    args = parse_args()

    # initialize logging
    util.init_output_logging()

    # load hyper-parameters
    hps = util.load_profile(args.profile)
    util.manual_seed(hps.ablation.seed)

    # build graph
    builder = Builder(hps)
    state = builder.build()

    # load dataset of patches (load images and crop online)
    transform = transforms.Compose(
        [
            transforms.Resize(hps.dataset.resize),
            transforms.RandomCrop(hps.model.image_shape[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.ImageFolder(hps.dataset.root, transform=transform)

    # start training
    trainer = Trainer(hps=hps, dataset=dataset, **state)
    trainer.train()
