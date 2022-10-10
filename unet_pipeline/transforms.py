import albumentations as A
from albumentations.augmentations.geometric.resize import Dict
from albumentations.pytorch import ToTensorV2


def augment_dict(CFG):

    data_transforms = {
        "train": A.Compose(
            [
                #         A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                #         A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.OpticalDistortion(
                            distort_limit=0.05, shift_limit=0.05, p=1.0
                        ),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=CFG["img_size"][0] // 20,
                    max_width=CFG["img_size"][1] // 20,
                    min_holes=5,
                    fill_value=0,
                    p=0.5,
                ),  # mask_fill_value
            ],
            p=1.0,
        ),
        "valid": A.Compose(
            [
                #         A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            ],
            p=1.0,
        ),
    }
    return augment_dict
