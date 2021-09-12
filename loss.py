import torch
import torch.nn.functional as F
from torch import nn


# Generalized Cross Entropy Loss
class GCELoss(nn.Module):

    def __init__(self, q=0.7, ignore_index=-100):
        super(GCELoss, self).__init__()
        self.q = q
        self.ignore_index = ignore_index
             
    def forward(self, logits, targets, weights):
        valid_idx = targets != self.ignore_index
        logits = logits[valid_idx]
        targets = targets[valid_idx]
        weights = weights[valid_idx]
        # vanilla cross entropy when q = 0
        if self.q == 0:
            if logits.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(logits.view(-1), targets.float())
            else:
                ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
                loss = ce_loss(logits, targets)
        else:
            if logits.size(-1) == 1:
                pred = torch.sigmoid(logits)
                pred = torch.cat((1-pred, pred), dim=-1)
            else:
                pred = F.softmax(logits, dim=-1)
            pred = torch.gather(pred, dim=-1, index=torch.unsqueeze(targets, -1))
            loss = (1-pred**self.q) / self.q
        loss = (loss.view(-1)*weights).sum() / weights.sum()
        return loss
