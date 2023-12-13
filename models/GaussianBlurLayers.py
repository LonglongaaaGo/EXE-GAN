import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st


def gauss_kernel(size=21, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2,sigma+interval/2,size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, size, size))
    out_filter = np.tile(out_filter, [outchannels, inchannels, 1, 1])
    return out_filter



class GaussianBlurLayer(nn.Module):
    def __init__(self, size, sigma, in_channels=1, stride=1, pad=1):
        super(GaussianBlurLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.ch = in_channels
        self.stride = stride
        self.pad = nn.ReflectionPad2d(pad)

    def forward(self, x):
        kernel = gauss_kernel(self.size, self.sigma, self.ch, self.ch)
        kernel_tensor = torch.from_numpy(kernel)
        kernel_tensor = kernel_tensor.cuda()
        x = self.pad(x)
        blurred = F.conv2d(x, kernel_tensor, stride=self.stride)
        return blurred


class ConfidenceDrivenMaskLayer(nn.Module):
    def __init__(self, size=33, sigma=1.0/100, iters=7,pad=16):
    # def __init__(self, size=65, sigma=1.0 / 1, iters=1):

        super(ConfidenceDrivenMaskLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.iters = iters
        self.propagationLayer = GaussianBlurLayer(size, sigma, pad=pad)

    def forward(self, mask):
        # here mask 1 indicates missing pixels and 0 indicates the valid pixels
        init = 1 - mask
        mask_confidence = None
        for i in range(self.iters):
            mask_confidence = self.propagationLayer(init)
            mask_confidence = mask_confidence * mask
            init = mask_confidence + (1 - mask)
        return mask_confidence

    def show_mask_list(self, mask):
        # here mask 1 indicates missing pixels and 0 indicates the valid pixels
        init = 1 - mask
        mask_confidence = None
        out_list = []
        for i in range(self.iters):
            mask_confidence = self.propagationLayer(init)
            mask_confidence = mask_confidence * mask
            init = mask_confidence + (1 - mask)
            out_list.append(mask_confidence)
        return mask_confidence,out_list