# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

# from .transforms import (
#     GaussianBlur,
#     make_normalize_transform,
# )

import albumentations as A
import numpy as np
import random
import cv2
import torch


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        num_channels=8, ##fix: add for flexible channels number
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        ##fix: add for flexible channels number
        self.num_channels = num_channels

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        self.geometric_augmentation_global = A.Compose(
            [
                A.RandomResizedCrop(
                height=global_crops_size,
                width=global_crops_size,
                scale=global_crops_scale,
                interpolation=1  # Using 1 for BICUBIC interpolation
            ),
                A.HorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = A.Compose(
            [
                A.RandomResizedCrop(
                height=local_crops_size,
                width=local_crops_size,
                scale=local_crops_scale,
                interpolation=1  # Using 1 for BICUBIC interpolation
            ),
                A.HorizontalFlip(p=0.5),
            ]
        )

        color_jittering = RandomColorJitterAndGrayscale(brightness=0.4, contrast=0.4, p_jitter=0.8, p_to_gray=0.2,
                                                        out_channels=self.num_channels)

        global_transfo1_extra = CustomGaussianBlur(p=1.0)


        # global_transfo2_extra: gaussian_blur + solarize
        global_transfo2_extra = self_func_compose(
            [CustomGaussianBlur(p=0.1),
             RandomSolarize(threshold=128.0, p=0.2)]
        )

        local_transfo_extra = CustomGaussianBlur(p=0.5)



        self.normalize = Normalize(bands=self.num_channels) # convert to tensor

        self.global_transfo1 = self_func_compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 =self_func_compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = self_func_compose([color_jittering, local_transfo_extra, self.normalize])



    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image=image)['image']
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image=image)['image']
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image=image)['image']) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

########## Customized augmentation class / methods 

# customize 8 channels transform (albumentation-like format )
##  class RandomBrightness(object):     # original implementation, effect is too random, washed out 
#     """Random Brightness"""

#     def __init__(self, brightness=0.4):
#         self.brightness = brightness

#     def __call__(self, sample):
#         image = sample["image"]
#         s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
#         image = image * s
#         return {'image':image}
    
class RandomBrightness(object):
    """Random Brightness using blending"""

    def __init__(self, brightness=0.4):
        self.brightness = brightness

    def __call__(self, sample):
        image = sample["image"]
        
        # Convert the image to float32 for accurate computation
        image = image.astype(np.float32)
        
        # Generate a random factor s
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        
        # Create a zero (black) image
        zero_image = np.zeros_like(image)
        
        # Blend the original image with the zero image
        image = image * s + zero_image * (1 - s)
        
        # Clip the values to be in the valid range [0, 255] and convert back to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return {'image': image}

class ToGray(object):
    def __init__(self, out_channels=8):
        self.out_channels = out_channels

    def __call__(self, sample):
        image = sample["image"]
        gray_img = np.mean(image, axis=-1)
        gray_img = np.tile(gray_img[..., np.newaxis], (1, 1, self.out_channels))
        
        return {'image': gray_img.astype(np.uint8)}



# original (random scale implementation)
# class RandomContrast(object):
#     """Random Contrast"""

#     def __init__(self, contrast=0.4):
#         self.contrast = contrast

#     def __call__(self, sample):
#         image = sample["image"]

#         s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
#         mean = np.mean(image, axis=(0, 1), keepdims=True)

#         img_contrasted = (image - mean) * s + mean
#         return {'image':img_contrasted.clip(0, 1)}
    
class RandomContrast(object):
    """Random Contrast using blending"""

    def __init__(self, contrast=0.4):
        self.contrast = contrast

    def __call__(self, sample):
        image = sample["image"]

        # Convert the image to float32 for accurate computation
        image = image.astype(np.float32)

        # Generate a random contrast factor
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)

        # Calculate the mean of the image (grayscale version)
        mean = np.mean(image, axis=(0, 1), keepdims=True)

        # Blend the original image with the mean image
        img_contrasted = image * s + mean * (1 - s)

        # Clip the values to be in the valid range [0, 255] and return
        return {'image': img_contrasted.clip(0, 255).astype(np.uint8)}


# class CustomGaussianBlur(object):
#     """
#     Apply Gaussian Blur with a given probability.
#     The blur is applied to each channel of an 8-channel numpy array image.
#     """

#     def __init__(self, p=0.5, blur_limit=(7, 7), sigma_limit=(0.1, 2.0)):
#         self.p = p
#         self.blur_limit = blur_limit  # This should ideally represent the kernel size
#         self.sigma_limit = sigma_limit

#     def __call__(self, image):
#         if random.random() > self.p:
#             return image  # Return the original image with probability 1 - p

#         # Apply GaussianBlur individually to each channel
#         # Note: Albumentations' GaussianBlur doesn't directly use sigma_limit for blurring.
#         # If direct control over sigma is needed, consider an alternative implementation.
#         blurred_channels = [A.GaussianBlur(blur_limit=self.blur_limit, always_apply=True)(image=image[:, :, i])['image']
#                             for i in range(image.shape[-1])]

#         # Stack the channels back together
#         blurred_image = np.stack(blurred_channels, axis=-1)
#         return blurred_image


class CustomGaussianBlur(object):
    """
    Apply Gaussian Blur with a given probability.
    The blur is applied to each channel of an 8-channel numpy array image.
    """

    def __init__(self, p=0.5, blur_limit=(7, 7), sigma_limit=(0.1, 2.0)):
        self.p = p
        self.blur_limit = blur_limit  # This should ideally represent the kernel size
        self.sigma_limit = sigma_limit

    def __call__(self, image):
        if random.random() > self.p:
            return image  # Return the original image with probability 1 - p

        # Apply GaussianBlur individually to each channel
        blurred_channels = []
        for i in range(image.shape[-1]):
            sigma = random.uniform(*self.sigma_limit)
            blurred_channel = cv2.GaussianBlur(image[:, :, i], self.blur_limit, sigma)
            blurred_channels.append(blurred_channel)

        # Stack the channels back together
        blurred_image = np.stack(blurred_channels, axis=-1)
        return blurred_image



class Solarize(object):

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, x):

        x = x.astype(np.float32)

        # Invert pixel values below the threshold
        x1 = x.copy()
        mask = x >= self.threshold

        # x normalize to [0, 1], use max_value = 1
        x1[mask] = 255 - x1[mask]

        return x1.astype(np.uint8)


class RandomSolarize(object):
    def __init__(self, p=0.5, threshold=128):
        self.p = p
        self.threshold = threshold
        self.solarize = Solarize(threshold=threshold)

    def __call__(self, image):
        if random.random() > self.p:
            return image

        return self.solarize(image)


class self_func_compose(object):

    def __init__(self, transforms):
        """
        Initializes the Compose object with a list of transforms.

        Parameters:
        - transforms (list): A list of transformations to compose.
        """
        self.transforms = transforms

    def __call__(self, image):
        """
        Applies each of the transformations in sequence.

        Parameters:
        - image: The input image to apply the transformations on.

        Returns:
        - The transformed image.
        """
        for transform in self.transforms:
            image = transform(image)
        return image





# class color_jittering(object):

class ColorJitter(object):
    """Applies Color \ging with adjustable probabilities for RandomBrightness, RandomContrast, and ToGray."""

    def __init__(self, brightness=0.4, contrast=0.4, out_channels=8, p_brightness=1.0, p_contrast=1.0, p_to_gray=1.0):
        self.random_brightness = RandomBrightness(brightness=brightness)
        self.random_contrast = RandomContrast(contrast=contrast)
        self.to_gray = ToGray(out_channels=out_channels)
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_to_gray = p_to_gray

    def __call__(self, sample):
        # Apply RandomBrightness with probability p_brightness
        if np.random.rand() < self.p_brightness:
            sample = self.random_brightness(sample)
        # Apply RandomContrast with probability p_contrast
        if np.random.rand() < self.p_contrast:
            sample = self.random_contrast(sample)
        # Convert to gray scale with multiple channels with probability p_to_gray
        if np.random.rand() < self.p_to_gray:
            sample = self.to_gray(sample)
        return sample


class RandomColorJitterAndGrayscale(object):
    def __init__(self, brightness=0.4, contrast=0.4, out_channels=8, p_to_gray=0.2, p_jitter=0.8):
        self.random_brightness = RandomBrightness(brightness=brightness)
        self.random_contrast = RandomContrast(contrast=contrast)
        self.to_gray = ToGray(out_channels=out_channels)

        self.p_to_gray = p_to_gray
        self.p_jitter = p_jitter

    def __call__(self, sample):
        # these sub functions all require the input to be a dict
        if not isinstance(sample, dict):
            sample = {'image': sample}

        # Apply color_jitter with probability p_jitter
        if np.random.rand() < self.p_jitter:
            sample = self.random_contrast(self.random_brightness(sample))

        # Convert to gray scale with multiple channels with probability p_to_gray
        if np.random.rand() < self.p_to_gray:
            sample = self.to_gray(sample)

        # return image (np array)
        return sample['image']

# class Normalize(object):
#     def __init__(self, bands=8):
#         self.bands = bands

#     def __call__(self, image):
#         if isinstance(image, np.ndarray) and image.shape[0] != self.bands:
#             # convert to pytorch-tensor like shape
#             image = image.transpose(2, 0, 1)

#         if not torch.is_tensor(image):
#             image = torch.tensor(image)

#         return image
    
class Normalize(object):
    def __init__(self, bands=8):
        self.bands = bands
        self.to_tensor = transforms.ToTensor()
        # self.mean=[0.6881, 0.6265, 0.5738, 0.5223, 0.4898, 0.5746, 0.5612, 0.5603]
        # self.std = [0.092, 0.1007, 0.1125, 0.1199, 0.1263, 0.1341, 0.1434, 0.1446]
        
        
        # image net 
        # self.mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456]
        # self.std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224]
        
        # self.to_normalize = transforms.Normalize(self.mean, self.std)

    def __call__(self, image):

        if isinstance(image, np.ndarray):
            image = self.to_tensor(image)
            # image = self.to_normalize(image)

        return image 

        

        


   

