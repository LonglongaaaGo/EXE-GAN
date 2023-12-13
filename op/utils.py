import numpy as np
import torch
import random
import torch.nn.functional as F
import os
import cv2
import shutil


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int_)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int_)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def np_free_form_mask_random(minVertex, maxVertex,minLength, maxLength,  minBrushWidth,maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(minVertex,maxVertex)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(minLength,maxLength + 1)
        brushWidth = np.random.randint(minBrushWidth, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_rect_mask(im_size, mask_size, margin=8, rand_mask=True):
    mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)
    if rand_mask:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
    else:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = (im_size[0] - sz0) // 2
        of1 = (im_size[1] - sz1) // 2
    mask[of0:of0+sz0, of1:of1+sz1] = 1
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    rect = np.array([[of0, sz0, of1, sz1]], dtype=int)
    return mask, rect


def generate_stroke_rect_mask(im_size,mask_size,max_large_rect_num=5,max_rect_num=10, margin=0, parts=10, maxVertex=20, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)

    rand_half_size0 = int(mask_size[0]//2)
    rand_half_size1 = int(mask_size[1]//2)
    rect_num = np.random.randint(0,max_rect_num)

    rand_large_size0 = int(mask_size[0])
    rand_large_size1 = int(mask_size[1])
    large_rect_num = np.random.randint(0,max_large_rect_num)

    # full size masks
    for i in range(large_rect_num):
        # random size
        sz0 = rand_large_size0 + np.random.randint(-rand_large_size0, 0)
        sz1 = rand_large_size1 + np.random.randint(-rand_large_size1, 0)
        # random location
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
        mask[of0:of0 + sz0, of1:of1 + sz1] = 1
    #half size masks
    for i in range(rect_num):
        #random size
        sz0 = rand_half_size0 + np.random.randint(-rand_half_size0,0)
        sz1 = rand_half_size1 + np.random.randint(-rand_half_size1,0)
        #random location
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
        mask[of0:of0 + sz0, of1:of1 + sz1] = 1

    mask = np.expand_dims(mask, axis=-1)
    # randVertex =  np.random.randint(int(maxVertex//2),maxVertex)
    # randLength =  np.random.randint(int(maxLength//2),maxLength)
    # randBrushWidth=  np.random.randint(10,maxBrushWidth)
    randVertex = maxVertex
    randLength = maxLength
    randBrushWidth = maxBrushWidth
    # stroke
    for i in range(parts):
        mask = mask + np_free_form_mask(randVertex, randLength, randBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.transpose(mask, [2, 0, 1])
    mask = np.expand_dims(mask, 0)
    return mask


def generate_stroke_mask(im_size, parts=10, maxVertex=20, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.transpose(mask, [2, 0, 1])
    mask = np.expand_dims(mask, 0)
    return mask

def generate_mask(type, im_size, mask_size):
    if type == 'rect':
        return generate_rect_mask(im_size, mask_size)
    elif type == 'center':
        return generate_rect_mask(im_size, mask_size,rand_mask=False)
    elif type == 'stroke_rect':
        return generate_stroke_rect_mask(im_size,mask_size), None
    else:
        return generate_stroke_mask(im_size), None




def get_mask(real_image,mask_type, im_size,mask_shapes):
    current_batch_size = real_image.shape[0]
    mask, rect = generate_mask(mask_type, [im_size, im_size], mask_shapes)
    mask_01 = torch.from_numpy(mask).cuda().repeat([current_batch_size, 1, 1, 1])
    if mask_type == 'rect':
        rect = [rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3]]
        gt_local = real_image[:, :, rect[0]:rect[0] + rect[1],
                   rect[2]:rect[2] + rect[3]]
    else:
        gt_local = real_image
    im_in = real_image * (1 - mask_01)
    #[data,mask]
    # gin = torch.cat((im_in, mask_01), 1)
    gin = torch.cat((im_in, mask_01-0.5), 1)

    # real_image = torch.cat((real_image, mask_01), 1)
    return gin, gt_local,mask,mask_01,im_in


def get_whole_mask(real_image,mask_type, im_size,mask_shapes):
    current_batch_size = real_image.shape[0]
    mask_01 = torch.ones([current_batch_size, 1,  real_image.shape[2],  real_image.shape[3]]).cuda()

    im_in = real_image * (1 - mask_01)
    #[data,mask]
    # gin = torch.cat((im_in, mask_01), 1)
    gin = torch.cat((im_in, mask_01-0.5), 1)

    # real_image = torch.cat((real_image, mask_01), 1)
    return gin, real_image,mask_01,mask_01,im_in




def get_real_mask(real_image,mask_type, im_size,mask_shapes):
    """
    只是在原图的基础上加上mask， 而不改变原图
    :param real_image:
    :param mask_type:
    :param im_size:
    :param mask_shapes:
    :return:
    """
    current_batch_size = real_image.shape[0]
    mask, rect = generate_mask(mask_type, [im_size, im_size], mask_shapes)
    mask_01 = torch.from_numpy(mask).cuda().repeat([current_batch_size, 1, 1, 1])
    if mask_type == 'rect':
        rect = [rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3]]
        gt_local = real_image[:, :, rect[0]:rect[0] + rect[1],
                   rect[2]:rect[2] + rect[3]]
    else:
        gt_local = real_image

    # im_in = real_image * (1 - mask_01)
    #[data,mask]
    # gin = torch.cat((im_in, mask_01), 1)
    gin = torch.cat((real_image, mask_01-0.5), 1)

    # real_image = torch.cat((real_image, mask_01), 1)
    return gin, gt_local,mask,mask_01,real_image


def dic_2_str(dics):
    out_str = "\n"
    for key in dics.keys():
        out_str += str(key)+":"+str(dics[key])+ " "
    out_str+="\n"
    return out_str

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def delete_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path,ignore_errors=True)
    return

def copy_dirs(patha,pathb):
    delete_dirs(pathb)
    #copytree 包含了mkdir
    shutil.copytree(patha,pathb)

def get_completion(pred,gt,mask_01):
    gt = F.interpolate(gt,(pred.shape[2],pred.shape[3]))
    mask_01 = F.interpolate(mask_01,(pred.shape[2],pred.shape[3]))
    completion = pred * mask_01 + gt * (1 - mask_01)
    return completion



