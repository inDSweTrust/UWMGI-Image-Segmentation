import torch
import os

import numpy as np
import pandas as pd

from pickle import FALSE
from glob import glob
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

from utils import load_img, load_msk
from transforms import augment_dict


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.msk_paths = df["mask_path"].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = []
        img = load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data["image"]
                msk = data["mask"]
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data["image"]
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


# Create Folds
def create_folds(CFG, input_path, data_path):

    df = create_dataframe_w_paths(CFG, data_path)
    skf = StratifiedGroupKFold(
        n_splits=CFG["n_fold"], shuffle=True, random_state=CFG["seed"]
    )

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df, df["empty"], groups=df["case"])
    ):
        df.loc[val_idx, "fold"] = fold
    df.to_csv(Path(input_path, "train_folds.csv"))


# Create dataframe with paths to images and masks
def create_dataframe_w_paths(CFG, data_path):

    path_df = pd.DataFrame(
        glob(data_path + "/uwmgi-2.5d-stride2-dataset/images/*"), columns=["image_path"]
    )
    path_df["mask_path"] = path_df.image_path.str.replace("image", "mask")
    path_df["id"] = path_df.image_path.map(
        lambda x: x.split("/")[-1].replace(".npy", "")
    )

    df = pd.read_csv(data_path + "/uwmgi-mask-dataset/train.csv")
    df["segmentation"] = df.segmentation.fillna("")
    df["rle_len"] = df.segmentation.map(len)  # length of each rle mask

    df2 = (
        df.groupby(["id"])["segmentation"].agg(list).to_frame().reset_index()
    )  # rle list of each id
    df2 = df2.merge(
        df.groupby(["id"])["rle_len"].agg(sum).to_frame().reset_index()
    )  # total length of all rles of each id

    df = df.drop(columns=["segmentation", "class", "rle_len"])
    df = df.groupby(["id"]).head(1).reset_index(drop=True)
    df = df.merge(df2, on=["id"])
    df["empty"] = df.rle_len == 0  # empty masks

    df = df.drop(columns=["image_path", "mask_path"])
    df = df.merge(path_df, on=["id"])

    # Remove faulty data
    fault1 = "case7_day0"
    fault2 = "case81_day30"
    df = df[
        ~df["id"].str.contains(fault1) & ~df["id"].str.contains(fault2)
    ].reset_index(drop=True)

    return df


def prepare_loaders(CFG, input_path, data_path, debug=False, fold=0):

    fold_csv_file = os.path.isfile(Path(input_path, "train_folds.csv"))

    if fold_csv_file == False:
        create_folds(CFG, input_path, data_path)

    df = pd.read_csv(Path(input_path, "train_folds.csv"))

    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    train_df = df

    if debug:
        train_df = train_df.head(32 * 5).query("empty==0")
        valid_df = valid_df.head(32 * 3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=augment_dict["train"])
    valid_dataset = BuildDataset(valid_df, transforms=augment_dict["valid"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["train_bs"] if not debug else 20,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["valid_bs"] if not debug else 20,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader
