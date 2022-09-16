from torch import nn
import torch.nn.functional as F
import torch


class MySimclrLoss():
    def __init__(self, temprature) -> None:
        self.temprature = temprature

    def __call__(self, x):
        b, _, d = x.shape
        # split into origin and augmented
        anchor = x[:, 0, :].squeeze()
        aug = x[:, 1, :].squeeze()
        all_features = torch.cat((anchor.T, aug.T), dim=1)
        logits = anchor @ all_features / self.temprature

        # minus max for numeric stability
        max_v, _ = torch.max(logits, dim=1)
        n_logits = logits - max_v.unsqueeze(1)
        
        # mask for positive sample and self
        pos_mask = torch.cat((torch.zeros((b, b)), torch.eye(b)), dim=1)
        self_mask = torch.cat([1 - torch.eye(b), torch.ones((b,b))], dim=1)

        # log softmax
        pos_logits = n_logits[pos_mask.bool()]
        log_sum = torch.log((torch.exp(n_logits) * self_mask).sum(dim=1))
        loss = - (pos_logits - log_sum).mean()
        return loss

