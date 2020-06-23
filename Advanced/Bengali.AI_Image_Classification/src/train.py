import os
import torch
import torch.nn as nn
import engine
from dataset import BengaliDatasetTrain
import config
from model_dispatcher import MODEL_DISPATCHER
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    model = MODEL_DISPATCHER[config.BASE_MODEL](pretrained=True)
    model.to(device)

    train_dataset = BengaliDatasetTrain(
        folds=config.TRAINING_FOLDS,
        image_height=config.IMG_HEIGHT,
        image_width=config.IMG_WIDTH,
        mean=config.MODEL_MEAN,
        std=config.MODEL_STD,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True
    )

    valid_dataset = BengaliDatasetTrain(
        folds=config.VALIDATION_FOLDS,
        image_height=config.IMG_HEIGHT,
        image_width=config.IMG_WIDTH,
        mean=config.MODEL_MEAN,
        std=config.MODEL_STD,
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=5, factor=0.3, verbose=True
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(config.EPOCHS):
        engine.train(
            train_dataset, train_loader, model=model, optimizer=optimizer, device=device
        )
        val_score = engine.evaluate(
            valid_dataset, valid_loader, model=model, device=device
        )
        scheduler.step(val_score)
        torch.save(
            model.state_dict(),
            f"{config.BASE_MODEL}_fold{config.VALIDATION_FOLDS[0]}.pt",
        )


if __name__ == "__main__":
    run()
