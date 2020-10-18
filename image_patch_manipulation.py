import sys
import signal
import argparse
import os

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from tqdm import tqdm
import math
from PIL import Image

from misc import util
from network import Builder


def parse_args():
    parser = argparse.ArgumentParser(
        description='Image Manipulation after Training Patch Model"')
    parser.add_argument('profile', type=str,
                        default='profile/patch16_manipulation.json',
                        help='path to profile file')
    parser.add_argument('--mask_folder', type=str, required=True,
                        help='path to mask folder - mask should have the same name as the image')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to output file')
    parser.add_argument('--eta1', type=float, default=0.6,
                        help='positive eta')
    parser.add_argument('--eta2', type=float, default=-0.8,
                        help='negative eta')
    return parser.parse_args()


# Extracts patches in Row Stack
def extract_patches(x, patch_size):
    num_patches = (x.size(1)-patch_size) * (x.size(2)-patch_size)
    patches = torch.empty(num_patches, 3, patch_size, patch_size)
    for i in range(x.size(1)-patch_size):
        for j in range(x.size(2)-patch_size):
            patches[i*(x.size(2)-patch_size) + j, :, :, :] = x[:, i:i+patch_size, j:j+patch_size]
    return x[:, int(patch_size/2):x.size(1)-int(patch_size/2), int(patch_size/2):x.size(2)-int(patch_size/2)], patches


def get_mask(mask_path, w, h):
    mask = Image.open(mask_path)
    mask = mask.resize((w, h), resample=Image.NEAREST).convert('L')
    return mask


def mask_images(im0, im1, im2, mask):
    imA = Image.composite(im0, im1, mask)
    imB = Image.composite(im0, im2, mask)
    out = Image.new('RGB', (imA.width + imB.width, imA.height))
    out.paste(imA, (0, 0))
    out.paste(imB, (im1.width, 0))
    return out


if __name__ == '__main__':
    # this enables a Ctrl-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    # parse arguments
    args = parse_args()
    eta1 = args.eta1
    eta2 = args.eta2

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
    transform = transforms.Compose(
        [
            transforms.Resize(hps.dataset.resize),
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

    # create output dir
    os.makedirs(args.output_path, exist_ok=True)

    # start inference
    progress1 = tqdm(data_loader)
    for idx, input in enumerate(progress1):
        image = input[0][0].to(devices[0])
        patches = input[0][1].to(devices[0])
        prev_patches = patches.to(devices[0])

        # positive eta - gradually increase/decrease to fix inf values
        for target_eta in [eta1, eta2]:
            for eta in [0, target_eta/2, target_eta]:
                new_patches = torch.zeros_like(patches).to(devices[0])
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
                            z = z - eta * z * torch.exp(-0.5 * z ** 2)
                            if pad > 0:
                                new_patches[0, i * internal_batch:(i + 1) * internal_batch, :, :, :] = graph(z=z,
                                                                                        reverse=True)[:-pad, :, :, :]
                                nll = nll[:-pad]
                            else:
                                new_patches[0, i * internal_batch:(i + 1) * internal_batch, :, :, :] = graph(z=z,
                                                                                                             reverse=True)
                            # patches that are inf -> take from previous
                            p_patch = prev_patches[0, i * internal_batch:(i + 1) * internal_batch, :, :]
                            new_patches[0, i * internal_batch:(i + 1) * internal_batch, :, :, :][torch.isinf(new_patches[0, i * internal_batch:(i + 1) * internal_batch, :, :, :])] = p_patch[torch.isinf(new_patches[0, i * internal_batch:(i + 1) * internal_batch, :, :, :])]

                    prev_patches = new_patches
                    print("Finished eta = " + str(eta))
                    # compose the image - for final eta's
                    if eta == target_eta or eta == 0:
                        acc_image = torch.zeros(image.size(0), image.size(1), image.size(2) + patch_size - 1,
                                                image.size(3) + patch_size - 1)
                        counts = torch.zeros_like(acc_image)[:, 0, :, :]
                        new_patches = torch.clamp(new_patches, -1, 1)
                        for patch_num in range(new_patches.size(1)):
                            tr = int(patch_num / image.size(3))
                            br = tr + patch_size
                            lc = np.mod(patch_num, image.size(3))
                            rc = lc + patch_size
                            acc_image[0, :, tr:br, lc:rc] += new_patches[0, patch_num, :, :, :].cpu()
                            counts[0, tr:br, lc:rc] += 1
                        new_image = torch.div(acc_image, counts).cpu()
                        composed_im = transforms.ToPILImage()(new_image.squeeze()/2 + 0.5)
                        if eta == 0:
                            eta0_im = composed_im
                        elif eta == eta1:
                            eta1_im = composed_im
                        else:
                            eta2_im = composed_im

        filename = dataset.samples[idx][0]
        path_list = filename.split(os.sep)

        w, h = eta0_im.size
        mask = get_mask(os.path.join(args.mask_folder, path_list[-1]), w, h)

        out = mask_images(eta0_im, eta1_im, eta2_im, mask)
        out.save(os.path.join(args.output_path, 'manipulated_' + path_list[-1]))




