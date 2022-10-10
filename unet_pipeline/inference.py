import gc
import pandas as pd
import argparse
import yaml
import glob

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

from utils import (
    build_model,
    load_model,
    masks2rles,
    argparser,
    load_yaml,
    get_metadata,
    path2info,
)
from data import BuildDataset

from tqdm import tqdm
from pathlib import Path


@torch.no_grad()
def infer(CFG, model_paths1, model_paths2, test_loader, num_log=1):
    debug = CFG["debug"]
    thr = CFG["thr"]

    msks = []
    imgs = []
    pred_strings = []
    pred_ids = []
    pred_classes = []

    for idx, (img, ids, heights, widths) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer ")
    ):

        img = img.to(CFG.device, dtype=torch.float)  # .squeeze(0)
        size = img.size()

        msk_all = []
        msk = torch.zeros(
            (size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32
        )
        for path in model_paths1:
            model = load_model(path, CFG.backbone[0])
            out = model(img)  # .squeeze(0) # removing batch axis
            out = nn.Sigmoid()(out)  # removing channel axis
            msk += out / len(model_paths1)
            if CFG.TTA:
                flipped_img = torch.flip(img, dims=(3,))
                flipped_out = model(flipped_img)
                flipped_out = nn.Sigmoid()(flipped_out)
                flipped_msk = torch.flip(flipped_out, dims=(3,))
                msk = (msk + flipped_msk) / 2
        msk_all.append(msk)

        msk = torch.zeros(
            (size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32
        )
        for path in model_paths2:
            model = load_model(path, CFG.backbone[1])
            out = model(img)  # .squeeze(0) # removing batch axis
            out = nn.Sigmoid()(out)  # removing channel axis
            msk += out / len(model_paths2)
            if CFG.TTA:
                flipped_img = torch.flip(img, dims=(3,))
                flipped_out = model(flipped_img)
                flipped_out = nn.Sigmoid()(flipped_out)
                flipped_msk = torch.flip(flipped_out, dims=(3,))
                msk = (msk + flipped_msk) / 2
        msk_all.append(msk)

        # Ensemble
        msk = torch.mean(torch.stack(msk_all), dim=0)
        msk = (
            (msk.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().detach().numpy()
        )  # shape: (n, h, w, c)
        result = masks2rles(msk, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

        if idx < num_log and debug:
            img = img.permute((0, 2, 3, 1)).cpu().detach().numpy()
            imgs.append(img[::5])
            msks.append(msk[::5])
        del img, msk, out, model, result
        gc.collect()
        torch.cuda.empty_cache()

    return pred_strings, pred_ids, pred_classes, imgs, msks


def main():

    args = argparser()
    config_file = Path(args.CFG_TRAIN)

    CFG = load_yaml(config_file)

    input_path = CFG["ROOT_DIR"] + "/input"
    data_path = CFG["DATA_DIR"]

    CFG["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Test
    sub_df = pd.read_csv(Path(input_path, "sample_submission.csv"))

    if not len(sub_df):
        debug = True
        sub_df = pd.read_csv(Path(input_path, "train.csv"))
        sub_df = sub_df[~sub_df.segmentation.isna()][: 1000 * 3]
        sub_df = sub_df.drop(columns=["class", "segmentation"]).drop_duplicates()
    else:
        debug = False
        sub_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()
    sub_df = sub_df.progress_apply(get_metadata, axis=1)

    if debug:
        paths = glob(
            f"/kaggle/input/uw-madison-gi-tract-image-segmentation/train/**/*png",
            recursive=True,
        )
    #     paths = sorted(paths)
    else:
        paths = glob(
            f"/kaggle/input/uw-madison-gi-tract-image-segmentation/test/**/*png",
            recursive=True,
        )
    #     paths = sorted(paths)
    path_df = pd.DataFrame(paths, columns=["image_path"])
    path_df = path_df.progress_apply(path2info, axis=1)

    test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")

    # Meta Data
    channels = 3
    stride = 2
    for i in range(channels):
        test_df[f"image_path_{i:02}"] = (
            test_df.groupby(["case", "day"])["image_path"]
            .shift(-i * stride)
            .fillna(method="ffill")
        )
    test_df["image_paths"] = test_df[
        [f"image_path_{i:02d}" for i in range(channels)]
    ].values.tolist()
    if debug:
        test_df = test_df.sample(frac=1.0)
    test_df.image_paths[0]

    test_dataset = BuildDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_bs,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
    )

    model_paths1 = glob(f"{CKPT_DIR}/final-effnet-b5.bin")
    model_paths2 = glob(f"{CKPT_DIR}/final-effnet-b7.bin")

    pred_strings, pred_ids, pred_classes, imgs, msks = infer(
        model_paths1, model_paths2, test_loader
    )

    pred_df = pd.DataFrame(
        {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
    )
    if not debug:
        sub_df = pd.read_csv(
            "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
        )
        del sub_df["predicted"]
    else:
        sub_df = pd.read_csv(
            "../input/uw-madison-gi-tract-image-segmentation/train.csv"
        )[: 1000 * 3]
        del sub_df["segmentation"]

    sub_df = sub_df.merge(pred_df, on=["id", "class"])
    sub_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
