import os
import random

import torch
import torchvision.transforms as transforms

from PIL import Image, ImageFont, ImageDraw

from helpers.utils import add_watermark


# from WMEmbeddedSystems
class RandomWatermark(object):
    # noinspection PyUnresolvedReferences
    """Normalize an tensor image with mean and standard deviation.
        Given mean: ``(M1,...,Mn)`` and std: ``(M1,..,Mn)`` for ``n`` channels, this transform
        will normalize each channel of the input ``torch.*Tensor`` i.e.
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
        """

    def __init__(self, watermark, probability=0.5):
        self.watermark = torch.from_numpy(watermark)
        self.probability = probability

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if random.random() < self.probability:
            return add_watermark(tensor, self.watermark)
        return tensor


class EmbedText(object):

    def __init__(self, text, pos, strength):
        self.text = text
        self.pos = pos
        self.strength = strength

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        image = transforms.ToPILImage()(tensor)

        draw = ImageDraw.Draw(image)

        font_path = os.path.join(os.getcwd(), "font", "sans_serif.ttf")
        font = ImageFont.truetype(font_path, 10)

        draw.text(self.pos, self.text, fill=int(255 * self.strength), font=font)
        # image.show()
        tensor = transforms.ToTensor()(image)

        return tensor


def set_pattern(image_channel, pixel_value):

    image_channel[24, 1] = pixel_value
    image_channel[24, 2] = pixel_value
    image_channel[24, 3] = pixel_value
    image_channel[25, 1] = pixel_value
    image_channel[26, 1] = pixel_value
    image_channel[26, 2] = pixel_value
    image_channel[26, 3] = pixel_value

    return image_channel


def set_pattern_contrast(image_channel):

    #select color by color of middle pixel
    color = 0 if image_channel[25, 2] > 0.5 else 1

    image_channel[24, 1] = color
    image_channel[24, 2] = color
    image_channel[24, 3] = color
    image_channel[25, 1] = color
    image_channel[26, 1] = color
    image_channel[26, 2] = color
    image_channel[26, 3] = color

    return image_channel


class EmbedPattern:

    def __init__(self, img_type='gray'):
        self.img_type = img_type

    def __call__(self, tensor):
        image = tensor

        if self.img_type == 'rgb':
            image[0] = set_pattern(image[0], 1)
            image[1] = set_pattern(image[1], 0)
            image[2] = set_pattern(image[2], 0)

        elif self.img_type == 'rgb_contrast':
            image[0] = set_pattern_contrast(image[0])
            image[1] = set_pattern_contrast(image[1])
            image[2] = set_pattern_contrast(image[2])

        elif self.img_type == 'gray':
            image[0] = set_pattern(image[0], 1)

        tensor = image

        return tensor
