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

        draw.text(self.pos, self.text, fill=int(255*self.strength), font=font)
        #image.show()
        tensor = transforms.ToTensor()(image)

        return tensor