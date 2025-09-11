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

from .extended import ExtendedVisionDataset   # parent class of image_net

##fix: add
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

import torch

logger = logging.getLogger("dinov2")
_Target = int


class NLBDataset(ExtendedVisionDataset):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:

        super().__init__(root, transforms, transform, target_transform)

        self.root = pathlib.Path(root)
        self.images_paths = list(self.root.iterdir())
        self.error_data = []

    def get_image_data(self, index: int):  # should return an image as an array

        image_path = self.images_paths[index]
        img = np.load(image_path)

        return img

    def get_target(self, index: int) -> Any:    # 0 for all data 
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            print(self.images_paths[index])
            self.error_data.append(self.images_paths[index])
            # return None, None
            raise RuntimeError(f"can not read image for sample {index}") from e
            # return None, None
        target = self.get_target(index)

        if self.transforms is not None:
            # the format for 'albumentations', [h w c] numpy ndarray
            # if isinstance(image, np.ndarray) and image.shape[0] == 8: ##fix: hard coded Channel dimension
            if isinstance(image, np.ndarray) and image.shape[0] ==8: # when first axis is channel dimension e.g., 3 channels, 8 channels, etc
                image = image.transpose(1, 2, 0)    # [c h w] -> [h w c]

            image = self.transforms(image)

        else:
            if isinstance(image, np.ndarray) and image.shape[0] != 8: 
                image = image.transpose(0, 1, 2) 
            image = torch.from_numpy(image)

        return image, target

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.images_paths)

