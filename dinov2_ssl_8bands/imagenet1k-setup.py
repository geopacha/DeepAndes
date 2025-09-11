# _*_ coding: utf-8 _*_

"""
Generate extra meta files for Imagenet-1k
"""

from dinov2.data.datasets import ImageNet

root = '/home/guoj5/Documents/datasets/imagenet1k'
extra = '/home/guoj5/Documents/datasets/imagenet1k_meta'

for split in ImageNet.Split:
    # SETUP
    # dataset = ImageNet(split=split, root=root, extra=extra)
    # dataset.dump_extra()

    # DEBUG
    debug_rgb = False
    if debug_rgb:
        dataset = ImageNet(split=split, root=root, extra=extra)
        sample = dataset[0]

    # CUSTOMIZED
    customized = True
    if customized:
        from dinov2.data.datasets import NLBDataset
        from dinov2.data.rs_augmentations import DataAugmentationDINO as DataAugmentationDINO_MS

        transforms = DataAugmentationDINO_MS(
            global_crops_size=224,
            local_crops_size=96,
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8
        )

        dataset = NLBDataset(root="/home/guoj5/Desktop/data_normalized_rename/train/combine/", 
                             transforms=transforms)

        sample = dataset[0]
        print()

        # cfg.crops.global_crops_scale,
        # cfg.crops.local_crops_scale,
        # cfg.crops.local_crops_number,
        # global_crops_size=cfg.crops.global_crops_size,
        # local_crops_size=cfg.crops.local_crops_size,