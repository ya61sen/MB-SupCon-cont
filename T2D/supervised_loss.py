"""
Refer to CMC: https://github.com/HobbitLong/CMC/blob/master/train_CMC.py
Refer to SupContrast: https://github.com/HobbitLong/SupContrast

Modified from Yonglong Tian's code on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
Reference: https://github.com/HobbitLong/SupContrast/blob/master/losses.py 
               - by Yonglong Tian (yonglong@mit.edu)
"""
from __future__ import print_function

import torch
import torch.nn as nn
from itertools import combinations

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, device="cuda:0"):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None, multi_core_index=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of samples in a batch')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        if multi_core_index is None:
            comb = combinations(range(contrast_count), 2)
        else:
            idx_list = list(range(contrast_count))
            idx_list.remove(multi_core_index)
            comb = [(multi_core_index, ii) for ii in idx_list]
        loss_all = torch.tensor(0.).to(device)
        for idx1, idx2 in comb:
            contrast_feature_g = features[:,idx1,:]
            contrast_feature_m = features[:,idx2,:]

            # compute logits
            anchor_dot_contrast_gm = torch.div(
                torch.matmul(contrast_feature_g, contrast_feature_m.T),
                self.temperature)
            # for numerical stability
            logits_max_gm, _ = torch.max(anchor_dot_contrast_gm, dim=1, keepdim=True)
            logits_gm = anchor_dot_contrast_gm - logits_max_gm.detach()

            # compute log_prob
            exp_logits_gm = torch.exp(logits_gm)
            log_prob_gm = logits_gm - torch.log(exp_logits_gm.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos_gm = (mask * log_prob_gm).sum(1) / mask.sum(1)

            # loss
            loss_gm = - (self.temperature / self.base_temperature) * mean_log_prob_pos_gm
            loss_gm = loss_gm.mean()

            ##################################################################################
            ##################################################################################

            # compute logits
            anchor_dot_contrast_mg = torch.div(
                torch.matmul(contrast_feature_m, contrast_feature_g.T),
                self.temperature)
            # for numerical stability
            logits_max_mg, _ = torch.max(anchor_dot_contrast_mg, dim=1, keepdim=True)
            logits_mg = anchor_dot_contrast_mg - logits_max_mg.detach()

            # compute log_prob
            exp_logits_mg = torch.exp(logits_mg)
            log_prob_mg = logits_mg - torch.log(exp_logits_mg.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos_mg = (mask * log_prob_mg).sum(1) / mask.sum(1)

            # loss
            loss_mg = - (self.temperature / self.base_temperature) * mean_log_prob_pos_mg
            loss_mg = loss_mg.mean()
            
            loss_all += loss_gm + loss_mg

        return loss_all