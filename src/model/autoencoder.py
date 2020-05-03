# Install dependencies
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder structure
        self.enc_cnv_1 = nn.Conv2d(3, 32, (7, 7), padding=3)
        self.enc_cnv_2 = nn.Conv2d(32, 32, (5, 5), padding=2)
        self.enc_cnv_3 = nn.Conv2d(32, 16, (3, 3), padding=1)
        self.enc_cnv_4 = nn.Conv2d(16, 8, (3, 3), padding=1)

        # Decoder structure
        self.dec_cnv_1 = nn.Conv2d(8, 8, (3, 3), padding=1)
        self.dec_cnv_2 = nn.Conv2d(8, 16, (3, 3), padding=1)
        self.dec_cnv_3 = nn.Conv2d(16, 32, (5, 5), padding=2)
        self.dec_cnv_4 = nn.Conv2d(32, 3, (7, 7), padding=3)

    def encode(self, images):
        """
        :param images: mini-batch of input images
        :return: encoded representation of each image
        """
        code = F.leaky_relu(self.enc_cnv_1(images))
        code = F.max_pool2d(code, 2)
        code = F.leaky_relu(self.enc_cnv_2(code))
        code = F.max_pool2d(code, 2)
        code = F.leaky_relu(self.enc_cnv_3(code))
        code = F.max_pool2d(code, 2)
        code = F.relu(self.enc_cnv_4(code))
        return code

    def decode(self, code):
        """
        :param code:
        :return: reconstructed image
        """
        out = F.leaky_relu(self.dec_cnv_1(code))
        out = F.interpolate(out, scale_factor=2)
        out = F.leaky_relu(self.dec_cnv_2(out))
        out = F.interpolate(out, scale_factor=2)
        out = F.leaky_relu(self.dec_cnv_3(out))
        out = F.interpolate(out, scale_factor=2)
        out = th.sigmoid(self.dec_cnv_4(out))
        return out

    def forward(self, images):
        """
        Overrides method in nn.Module
        :param images: input images
        :return: reconstructed images
        """
        code = self.encode(images)
        return self.decode(code)


