import os
import glob

import albumentations as A
import cv2
import lightning as L
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class CustomSegDatagen(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.is_test:
            mask = cv2.imread(self.mask_paths[idx], 0)
            mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]

            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            return image, mask.long().unsqueeze(0)
        else:
            # Test mode - no masks
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            return image, self.image_paths[idx]  # Return image and path for saving predictions


class CustomSegDataset(L.LightningDataModule):
    def __init__(
        self,
        batch_size=8,
        root_dir="./data",
        num_workers=2,
        img_size=(384, 384),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.img_size = img_size

    def get_train_transforms(self):
        return A.Compose(
            [
                A.Resize(*self.img_size, interpolation=cv2.INTER_LANCZOS4),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(
                    p=0.5, brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01
                ),
                A.Affine(
                    p=0.5,
                    scale=(0.5, 1.5),
                    translate_percent=0.125,
                    rotate=90,
                    interpolation=cv2.INTER_LANCZOS4,
                ),
                A.ElasticTransform(p=0.5, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def get_val_transforms(self):
        return A.Compose(
            [
                A.Resize(*self.img_size, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def get_test_transforms(self):
        return A.Compose(
            [
                A.Resize(*self.img_size, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def setup(self, stage=None):
        # Load train data
        train_images = sorted(glob.glob(os.path.join(self.root_dir, "Train/Image/*.jpg")))
        train_masks = sorted(glob.glob(os.path.join(self.root_dir, "Train/Mask/*.png")))

        # Load validation data
        val_images = sorted(glob.glob(os.path.join(self.root_dir, "Val/Image/*.jpg")))
        val_masks = sorted(glob.glob(os.path.join(self.root_dir, "Val/Mask/*.png")))

        # Load test data (no masks)
        test_images = sorted(glob.glob(os.path.join(self.root_dir, "Test/Image/*.jpg")))

        self.train_set = CustomSegDatagen(
            train_images, train_masks, transform=self.get_train_transforms()
        )
        self.val_set = CustomSegDatagen(
            val_images, val_masks, transform=self.get_val_transforms()
        )
        self.test_set = CustomSegDatagen(
            test_images, None, transform=self.get_test_transforms(), is_test=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )