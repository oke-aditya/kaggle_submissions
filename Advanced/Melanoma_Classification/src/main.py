import os
import pandas as pd

import numpy as np
import torch
import pretrainedmodels

import torch.nn as nn
import torch.nn.functional as F

# from apex import amp

from sklearn import metrics

import config
import albumentations

from wtfml.data_loaders.image import ClassificationLoader
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super().__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)
    
    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        
        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))

        return out, loss

def train(fold):
    train_images_path = "../input/images/"
    train_data_path = "../input/train_folds.csv"
    df = pd.read_csv(train_data_path)
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    epochs = config.EPOCHS
    train_batch_size = config.TRAIN_BATCH_SIZE
    valid_batch_size = config.VALID_BATCH_SIZE
    mean = config.MEAN
    std = config.STD

    df_train = df[df["kflold"] != fold].reset_index(drop=True)
    df_valid = df[df["kflold"] == fold].reset_index(drop=True)

    train_aug = albumentations.Compose([
        albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
    ])

    valid_aug = albumentations.Compose([
        albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
    ])

    train_images = df_train["image_name"].values.tolist()
    train_images = [os.path.join(train_images_path, i + ".jpg") for i in train_images]
    train_targets = df_train["target"].values

    valid_images = df_valid["image_name"].values.tolist()
    valid_images = [os.path.join(train_images_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid["target"].values
    
    train_dataset = ClassificationLoader(image_paths=train_images, targets=train_targets, 
                            resize=None, augmentations=train_aug)
    
    valid_dataset = ClassificationLoader(image_paths=valid_images, targets=valid_targets, 
                            resize=None, augmentations=valid_aug)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
    shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size,
    shuffle=False)

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=0.001, mode="max")

    es = EarlyStopping(patience=5, mode="max")

    for epoch in range(config.EPOCHS):
        training_loss = Engine.train(train_loader, model, optimizer, device)
        predictions, valid_loss = Engine.evaluate(valid_loader, model, device)
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"Epoch = {epoch}, auc = {auc} ")
        es(auc, model, config.MODEL_PATH)

        if es.early_stop:
            print("Early Stopping ... ")
            break

    # model, optimizer = 

def predict(fold):
    test_data_path = "../input/siic-isic-224x224-images/test/"
    df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
    device = "cuda"
    model_path = config.MODEL_PATH

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    images = df.image_name.values.tolist()
    images = [os.path.join(test_data_path, i + ".png") for i in images]
    targets = np.zeros(len(images))

    test_dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False,
    )

    model = SEResNext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions = Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions)).ravel()

    return predictions


if __name__ == "__main__":
    train(fold=0)
