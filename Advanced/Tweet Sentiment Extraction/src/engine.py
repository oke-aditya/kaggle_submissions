import utils
import string
import utils
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1 + l2


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    print("Pass 1")
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    print("Pass 2")
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        print("Pass 3")

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1, o2 = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def eval_fn(data_loader, model, device):
    with torch.no_grad():
        model.eval()
        print("Pass e1")
        # losses = utils.AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        print("Pass e2")

        fin_output_start = []
        fin_output_end = []
        fin_padding_lens = []
        fin_tweet_tokens = []
        fin_orig_sentiment = []
        fin_orig_selected = []
        fin_orig_tweet = []

        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            # targets_start = d["targets_start"]
            # targets_end = d["targets_end"]
            tweet_tokens = d["tweet_tokens"]
            padding_len = d["padding_len"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected_text"]
            orig_tweet = d["orig_tweet"]
            orig_sentiment = d["orig_sentiment"]

            print("Pass e3")

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            # targets_start = targets_start.to(device, dtype=torch.float)
            # targets_end = targets_end.to(device, dtype=torch.float)

            o1, o2 = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

            # loss = loss_fn(o1, o2, targets_start, targets_end)
            # loss.backward()
            # losses.update(loss.item(), ids.size(0))
            # tk0.set_postfix(loss=losses.avg)

            fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy())
            fin_output_end.append(torch.sigmoid(o2).cpu().detach().numpy())
            fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())

            fin_tweet_tokens.extend(tweet_tokens)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_selected)
            fin_orig_tweet.extend(orig_tweet)

        fin_output_start = np.vstack(fin_output_start)
        fin_output_end = np.vstack(fin_output_end)

        threshold = 0.2
        jaccards = []
        for j in range(len(fin_tweet_tokens)):
            target_string = fin_orig_selected[j]
            tweet_tokens = fin_tweet_tokens[j]
            padding_len = fin_padding_lens[j]
            orignal_tweet = fin_orig_tweet[j]
            sentiment = fin_orig_sentiment[j]

            if padding_len > 0:
                mask_start = fin_output_start[j, :][:-padding_len] >= threshold
                mask_end = fin_output_end[j, :][:-padding_len] >= threshold

            else:
                mask_start = fin_output_start[j, :] >= threshold
                mask_end = fin_output_end[j, :] >= threshold

            mask = [0] * len(mask_start)
            idx_start = np.nonzero(mask_start)[0]
            indx_end = np.nonzero(mask_end)[0]

            if len(idx_start) > 0:
                idx_start = idx_start[0]
                if len(idx_end) > 0:
                    idx_end = idx_end[0]
                else:
                    idx_end = idx_start
            else:
                idx_start = 0
                idx_end = 0

            for mj in range(idx_start, idx_start + 1):
                mask[mj] = 1

            output_tokens = [
                x for p, x in enumerate(tweet_tokens.splits()) if mask[p] == 1
            ]
            output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]

            final_output = ""
            for ot in output_tokens:
                if ot.startswith("##"):
                    final_output = final_output + ot[2:]
                elif len(ot) == 1 and ot in string.punctuation:
                    final_output = final_output + ot
                else:
                    final_output = final_output + " " + ot
            final_output.strip()
            if sentiment == "neutral" or len(orignal_tweet.split()) < 4:
                final_output = orignal_tweet

            jac = utils.jaccard(target_string.strip(), final_output.strip())
            jaccards.append(jac)

        mean_jac = np.mean(jac)
    return mean_jac
