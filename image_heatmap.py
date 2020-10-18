import sys
import signal
import argparse
import os

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from tqdm import tqdm

from misc import util
from network import Builder

import matplotlib.pyplot as plt
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generating Image Heatmap according to Likelihood"')
    parser.add_argument('profile', type=str,
                        default='profile/patch16_heatmap.json',
                        help='path to profile file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to output file')
    return parser.parse_args()


# Extracts patches in Row Stack
def extract_patches(x, patch_size):
    num_patches = (x.size(1)-patch_size) * (x.size(2)-patch_size)
    patches = torch.empty(num_patches, 3, patch_size, patch_size)
    for i in range(x.size(1)-patch_size):
        for j in range(x.size(2)-patch_size):
            patches[i*(x.size(2)-patch_size) + j, :, :, :] = x[:, i:i+patch_size, j:j+patch_size]
    return x[:, int(patch_size/2):x.size(1)-int(patch_size/2), int(patch_size/2):x.size(2)-int(patch_size/2)], patches


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
    patch_size = hps.model.image_shape[0]
    image_size = hps.dataset.resize
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: extract_patches(x, patch_size))
        ]
    )

    dataset = datasets.ImageFolder(hps.dataset.root, transform=transform)

    data_loader = DataLoader(dataset, batch_size=hps.optim.num_image_batch,
                             num_workers=hps.dataset.num_workers,
                             shuffle=False,
                             drop_last=False)

    # start inference
    progress1 = tqdm(data_loader)
    for idx, input in enumerate(progress1):
        image = input[0][0].to(devices[0])
        patches = input[0][1].to(devices[0])
        nll_vec = np.zeros(patches.shape[1])

        with torch.no_grad():
            with tqdm(range(math.ceil(patches.size(1) / internal_batch))) as process2:
                for i in process2:
                    # pad batch if needed
                    pad = 0
                    if (i + 1) * internal_batch <= patches.size(1):
                        patch = patches[0, i * internal_batch:(i + 1) * internal_batch, :, :]
                    else:
                        patch = patches[0, i * internal_batch:, :, :]
                        pad = (i + 1) * internal_batch - patches.size(1)
                        zero_fill = torch.zeros(pad, patch.size(1), patch.size(2), patch.size(3)).cuda()
                        patch = torch.cat((patch, zero_fill), dim=0)
                    z, nll, y_logits = graph(x=patch, reverse=False)
                    if pad > 0:
                        nll = nll[:-pad]
                    nll_vec[i * internal_batch:(i + 1) * internal_batch] = nll.cpu().numpy()

            filename = dataset.samples[idx][0]
            path_list = filename.split(os.sep)
            base_dir = os.path.join(args.output_path, 'heatmap_' + str(image_size))
            os.makedirs(base_dir, exist_ok=True)

            # PLOT 1 - likelihood map only
            if image.size(2) < image.size(3):  # landscape orientation
                new_image = nll_vec.reshape(image_size-patch_size, -1)
            else:  # portrait orientation
                new_image = nll_vec.reshape(-1, image_size - patch_size)
            new_image = 255 * (new_image - np.min(new_image)) / np.max(new_image)
            im_to_save = transforms.ToPILImage()(np.expand_dims(new_image, axis=2).astype(np.float32))
            im_to_save.convert('RGB').save(os.path.join(base_dir, 'gray_' + path_list[-1]))


            # PLOT 2 - image overlay
            plt.imshow(image.cpu().squeeze().permute(1, 2, 0)/2 + 0.5)
            if image.size(2) < image.size(3):  # landscape orientation
                plt.imshow(nll_vec.reshape(image_size - patch_size, -1), alpha=0.5, cmap='hot')
            else:  # portrait orientation
                plt.imshow(nll_vec.reshape(-1, image_size - patch_size), alpha=0.5, cmap='hot')
            plt.colorbar()
            plt.savefig(os.path.join(base_dir, 'overlay_' + path_list[-1]))
            plt.clf()









