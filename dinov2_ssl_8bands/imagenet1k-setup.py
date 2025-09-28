# _*_ coding: utf-8 _*_

"""
Generate extra meta files for Imagenet-1k
"""

from dinov2.data.datasets import ImageNet

root = '/home/guoj5/Documents/datasets/imagenet1k'
extra = '/home/guoj5/Documents/datasets/imagenet1k_meta'

for split in ImageNet.Split:

    dataset = ImageNet(split=split, root=root, extra=extra)
    dataset.dump_extra()

