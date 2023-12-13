import numpy as np
from PIL import Image
import torch
import glob
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def get_img_lists(path, suffix):
    out = list(glob.glob('{}/*{}'.format(path, suffix)))
    out.sort()
    return out

def load_img2tensor(path,size):
    """
    :param path: a image path
    :param size:  size : 256 ..
    :return: a mask tensor   [1,3,size,size]
    """
    original_image = Image.open(path).convert('RGB')
    original_image = TF.to_tensor(original_image).unsqueeze(0)
    original_image = F.interpolate(original_image, size=(size, size))
    original_image = (original_image - 0.5) * 2
    return original_image


def color_2mask(mask):
    if mask is None: return None
    # mask = mask.copy()
    # mask = np.array(mask)

    # mask = Image.fromarray(mask)
    out_ = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
    # mask = np.array(mask[:, :, 0], dtype=np.uint8)
    mask_ = out_.copy()
    mask_[mask_ <= 10] = 0
    mask_[mask_ > 10] = 1
    mask = np.expand_dims(mask_, axis=-1)
    return mask

def load_colorTensor(path,size):
    """
    :param path: a image path
    :param size:  size : 256 ..
    :return: a mask tensor   [1,3,size,size]
    """
    original_image = Image.open(path).convert('RGB')
    color_mask = color_2mask(np.array(original_image))
    color_mask = torch.tensor(color_mask).unsqueeze(0).unsqueeze(0).squeeze(-1)
    # original_image = np.array(original_image)
    # original_image[original_image<10] = 0
    original_image = TF.to_tensor(original_image).unsqueeze(0)
    original_image = F.interpolate(original_image, size=(size, size))
    color_mask = F.interpolate(color_mask, size=(size, size),mode = 'nearest')
    original_image = (original_image - 0.5) * 2 * color_mask

    return original_image


def load_semantic2tensor(path, size,one_hot_tag=True):

    semancti_map = Image.open(path).convert('L')
    semancti_map = semancti_map.resize((size, size), Image.NEAREST)
    semancti_map = np.array(semancti_map)
    #
    semancti_map = torch.from_numpy(semancti_map)
    if one_hot_tag:
        semancti_map = F.one_hot(semancti_map.to(torch.int64), num_classes=19)
        semancti_map = semancti_map.permute(2,0,1).unsqueeze(0)

    return semancti_map

def load_mask2tensor(path,size):
    """
    :param path: mask path
    :param size:  size : 256 ..
    :return: a mask tensor   [1,1,size,size]
    """
    mask_img = Image.open(path).convert("L")
    # mask dim and value
    mask_img = np.array(mask_img)
    if mask_img.ndim == 2:
        mask = np.expand_dims(mask_img, axis=0)
    else:
        mask_img = np.transpose(mask_img, (2, 0, 1))
        mask = mask_img[0:1, :, :]
    mask[mask <= 200] = 0
    mask[mask > 200] = 1.0
    masks = torch.from_numpy(mask).unsqueeze(0).float()
    masks = F.interpolate(masks, size=(size, size),mode="nearest")

    return masks


def sketch2tensor(mask_img,size):
    """
    :param path: mask path
    :param size:  size : 256 ..
    :return: a mask tensor   [1,1,size,size]
    """
    # mask_img = Image.open(path).convert("L")
    # mask dim and value
    mask_img = np.array(mask_img)
    if mask_img.ndim == 2:
        mask = np.expand_dims(mask_img, axis=0)
    else:
        mask_img = np.transpose(mask_img, (2, 0, 1))
        mask = mask_img[0:1, :, :]
    mask[mask <= 200] = 0
    mask[mask > 200] = 1.0
    masks = torch.from_numpy(mask).unsqueeze(0).float()
    masks = F.interpolate(masks, size=(size, size),mode="nearest")

    return masks



if __name__ == '__main__':

    masked_dir = "/home/k/EXE-GAN_cases/Image_Re-composition/mask"
    gt_dir = "/home/k/EXE-GAN_cases/Image_Re-composition/gt_img"
    exemplar_dir = "/home/k/EXE-GAN_cases/Image_Re-composition/exemplar"
    exe_post = "_exemplar.png"
    mask_post = "_mask.png"
    gt_post = "_real.png"

    gt_imgs = get_img_lists(gt_dir,gt_post)
    mask_imgs = get_img_lists(masked_dir,mask_post)

    exe_imgs = get_img_lists(exemplar_dir,exe_post)

    for i in range(len(exe_imgs)):
        exe_img_ = load_img2tensor(exe_imgs[i])
        gt_img_ = load_img2tensor(gt_imgs[i])
        mask_ = load_img2tensor(mask_imgs[i])


