import config
import dataset
import torch
import engine
from model import BERTBasedUncased
from sklearn import model_selection
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
import numpy as np


def run():
    print("---------- Starting Data Reading -------")
    df1 = pd.read_csv(
        "../input/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"]
    )
    df2 = pd.read_csv(
        "../input/jigsaw-unintended-bias-train.csv", usecols=["comment_text", "toxic"]
    )

    df_train = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    df_valid = pd.read_csv("../input/validation.csv")

    print("---- Data Read Sucessfully --- ")

    # # dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    # # dfx["sentiment"] = dfx["sentiment"].apply(
    # #     lambda x : 1 if x == "positive" else 0
    # # )

    # # df_train, df_valid = model_selection.train_test_split(
    # #     dfx,
    # #     test_size=0.1,
    # #     random_state=42,
    # #     stratify=dfx["sentiment"].values
    # # )

    # df_train = df_train.reset_index(drop=True)
    # df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        comment_text=df_train["comment_text"].values, target=df_train["toxic"].values
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4,
    )

    valid_dataset = dataset.BERTDataset(
        comment_text=df_valid["comment_text"].values, target=df_train["toxic"].values
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALIDATION_BATCH_SIZE, num_workers=1,
    )
    print("---- DataLoaders Created Sucessfully --- ")

    device = torch.device("cuda")

    model = BERTBasedUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = len(dfx) / (config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dataloader, model, optimizer, scheduler, device)
        outputs, targets = engine.eval_fn(valid_dataloader, model, device)
        targets = np.array(targets) >= 0.5
        accuracy = metrics.roc_auc_score(targets, outputs)
        print(f"AUC Score {accuracy}")

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
