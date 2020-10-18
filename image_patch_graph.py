import sys
import signal
import argparse
import os

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm

from misc import util
from network import Builder

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def parse_args():
    parser = argparse.ArgumentParser(
        description='Likelihood Graph Generation after Training Patch Model"')
    parser.add_argument('profile', type=str,
                        default='profile/patch16_graph.json',
                        help='path to profile file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to output file')
    parser.add_argument('--var_name', type=str, required=True,
                        help='variable name (i.e. Saturation)')
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
    internal_batch = hps.optim.num_batch_train

    # build graph
    builder = Builder(hps)
    devices = hps.device.graph
    state = builder.build(training=False)
    graph = state['graph']

    # load dataset
    transform = transforms.Compose(
        [
            transforms.Resize(hps.model.image_shape[0]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    dataset = datasets.ImageFolder(hps.dataset.root, transform=transform)

    data_loader = DataLoader(dataset, batch_size=hps.optim.num_image_batch,
                             num_workers=hps.dataset.num_workers,
                             shuffle=False,
                             drop_last=False)

    # start inference
    patch_size = hps.model.image_shape[0]
    progress1 = tqdm(data_loader)
    nll_vec = torch.empty(0).to(devices[0])
    for idx, input in enumerate(progress1):
        patches = input[0].to(devices[0])
        with torch.no_grad():
            patch = F.interpolate(patches, size=patch_size)
            z, nll, y_logits = graph(x=patch, reverse=False)
            nll_vec = torch.cat((nll_vec, nll))

    ysmoothed = gaussian_filter1d(torch.exp(-nll_vec).cpu().numpy(), sigma=3)

    os.makedirs(args.output_path, exist_ok=True)

    # plot likelihood
    plt.plot(ysmoothed, linewidth=3)
    plt.xlabel('Center ' + args.var_name, fontsize=20)
    plt.ylabel(r'$\propto$ Patch Likelihood', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, 'likelihood.png'))
    plt.clf()

    # plot percentile rank
    cum_vec = np.cumsum(ysmoothed)
    plt.plot(cum_vec/np.max(cum_vec), linewidth=3)
    plt.xlabel('Center ' + args.var_name, fontsize=20)
    plt.ylabel('Percentile Rank', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, 'percentile.png'))







