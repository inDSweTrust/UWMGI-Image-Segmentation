import os
import numpy as np
import random
import pandas as pd
import cupy as cp

import yaml
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import segmentation_models_pytorch as smp

import torch


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("> SEEDING DONE")


def id2mask(id_, df):
    idf = df[df["id"] == id_]
    wh = idf[["height", "width"]].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(["large_bowel", "small_bowel", "stomach"]):
        cdf = idf[idf["class"] == class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0, 0), (0, 0), (1, 0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask


def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)


def load_img(path):
    img = np.load(path)
    img = img.astype("float32")  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def load_imgs(CFG, img_paths):

    size = CFG["img_size"]

    imgs = np.zeros((*size, len(img_paths)), dtype=np.float32)
    for i, img_path in enumerate(img_paths):
        if i == 0:
            img, shape0 = load_img(img_path, size=size)
        else:
            img, _ = load_img(img_path, size=size)
        img = img.astype("float32")  # original is uint16
        mx = np.max(img)
        if mx:
            img /= mx  # scale image to [0, 1]
        imgs[..., i] += img
    return imgs, shape0


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    return msk


def show_img(img, mask=None):
    plt.imshow(img, cmap="bone")

    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [
            Rectangle((0, 0), 1, 1, color=_c)
            for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis("off")


# RLE
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def mask2rle(msk, thr=0.5):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    msk = cp.array(msk)
    pixels = msk.flatten()
    pad = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def masks2rles(msks, ids, heights, widths):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    for idx in range(msks.shape[0]):
        msk = msks[idx]
        height = heights[idx].item()
        width = widths[idx].item()
        shape0 = np.array([height, width])
        resize = np.array([320, 384])
        if np.any(shape0 != resize):
            diff = resize - shape0
            pad0 = diff[0]
            pad1 = diff[1]
            pady = [pad0 // 2, pad0 // 2 + pad0 % 2]
            padx = [pad1 // 2, pad1 // 2 + pad1 % 2]
            msk = msk[pady[0] : -pady[1], padx[0] : -padx[1], :]
            msk = msk.reshape((*shape0, 3))
        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[..., midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes


def get_metadata(row):
    data = row["id"].split("_")
    case = int(data[0].replace("case", ""))
    day = int(data[1].replace("day", ""))
    slice_ = int(data[-1])
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    return row


def path2info(row):
    path = row["image_path"]
    data = path.split("/")
    slice_ = int(data[-1].split("_")[1])
    case = int(data[-3].split("_")[0].replace("case", ""))
    day = int(data[-3].split("_")[1].replace("day", ""))
    width = int(data[-1].split("_")[2])
    height = int(data[-1].split("_")[3])
    row["height"] = height
    row["width"] = width
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    #     row['id'] = f'case{case}_day{day}_slice_{slice_}'
    return row


def load_model(path, backbone):
    model = build_model(backbone)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()
    return model


def build_model(CFG):
    model = smp.Unet(
        encoder_name=CFG["backbone"],
        encoder_weights="imagenet",
        in_channels=3,
        classes=CFG["num_classes"],
        activation=None,
    )
    model.to(CFG["device"])
    return model


def load_yaml(file_name):
    with open(file_name, "r") as stream:
        cfg = yaml.load(stream, Loader=yaml.SafeLoader)
    return cfg
