import torch
import torch.nn as nn
from tqdm import tqdm


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)

    return (l1 + l2 + l3) / 3


def train(dataset, data_loader, model, optimizer, device):
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=len(dataset)):
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        optimizer.zero_grad()
        image = image.to(device, dtype=torch.float)
        grapheme_root = grapheme_root.to(device, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(device, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(device, dtype=torch.long)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

        outputs = model(image)
        # grahpeme, vowel, consonant
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(dataset, data_loader, model, device):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in tqdm(enumerate(data_loader), total=len(dataset)):
        counter += 1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        image = image.to(device, dtype=torch.float)
        grapheme_root = grapheme_root.to(device, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(device, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(device, dtype=torch.long)

        outputs = model(image)
        # grahpeme, vowel, consonant
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)
        final_loss += loss

    return final_loss
