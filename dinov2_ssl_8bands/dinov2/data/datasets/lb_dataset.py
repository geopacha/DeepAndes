# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union, Any

import numpy as np
import pathlib
import torch

from .extended import ExtendedVisionDataset   # parent class of image_net

##fix: add
import albumentations as A
from torchvision import transforms

logger = logging.getLogger("dinov2")
_Target = int


class LBDataset(ExtendedVisionDataset):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:

        super().__init__(root, transforms, transform, target_transform)

        self.root = pathlib.Path(root)
        self.images_paths = list(self.root.rglob("*.npy")) # image saved as .npy format, glob al into a list

    def get_image_data(self, index: int):  # should return an image as an array

        image_path = self.images_paths[index]
        img = np.load(image_path)
        # img = Image.open(image_path).convert(mode="RGB")

        return img

    def get_target(self, index: int) -> Any:
        image_path = self.images_paths[index]
        target = int(image_path.parts[-2])  # it is pytorch like dataset, /parent path/class/samples, [-2] access the 'class'
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
            target = self.get_target(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        if self.transforms is not None:
            # the format for 'albumentations', [h w c] numpy array
            if isinstance(image, np.ndarray) and image.shape[0] == 8:
                image = image.transpose(1, 2, 0)
            image = self.transforms(image=image)["image"]


            # # last check of torch like tensor shape
            # if image.shape[0] != 8:
            #    image = image.transpose(2, 0, 1)

            # # convert to tensor
            # if not torch.is_tensor(image):
            #    image = torch.tensor(image)

            to_tensor = transforms.ToTensor()
            image  = to_tensor(image)



        return image, target

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.images_paths)


