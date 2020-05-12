# Fine tune the BERT model
import transformers
import config
import torch
import torch.nn as nn

class BERTBasedUncased(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768 * 2, 1)
        # 1 for binary classification
    
    def forward(self, ids, mask, token_type_ids):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # Vector of size 768 for each sample in batch.
        
        mean_pooling = torch.mean(o1, 1) # Both are vectors of size 768
        max_pooling = torch.max(o1, 1)
        cat = torch.cat((mean_pooling, max_pooling), 1)

        bo = self.bert_drop(cat)
        output = self.out(bo)
        return output

