"""
Implementation of the CIGA algorithm from `"Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs"
<https://arxiv.org/abs/2202.05441>`_ paper
"""
from re import M
import torch
from torch.autograd import grad
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from typing import Tuple

@register.ood_alg_register
class CIGA(BaseOODAlg):
    r"""
    Implementation of the CIGA algorithm from `"Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs"
    <https://arxiv.org/abs/2202.05441>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(CIGA, self).__init__(config)
        self.rep_out = None
        self.causal_out = None
        self.spu_out = None
        self.env_id = None
        self.step = 0
        self.ratio = config.ood.ood_param
        print(config.ood.extra_param)
        if len(config.ood.extra_param)>=3 and type(config.ood.extra_param[-1])!=str:
            self.anneal_step = config.ood.extra_param[-1]
        elif len(config.ood.extra_param)>=4:
            self.anneal_step = config.ood.extra_param[-2]
        else:
            self.anneal_step = 0
        self.contrast_rep = "feat"
        if type(config.ood.extra_param[-1]) == str:
            self.contrast_rep = config.ood.extra_param[-1]
        # if type(self.anneal_step)==str:
        #     self.anneal_step = int(self.anneal_step)
        # print("############", self.anneal_step)
        # self.step=config.ood.extra_param[-1] if type(config.ood.extra_param[-1])!=str else config.ood.extra_param[-2]

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set input data format and preparations

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        if self.model.training:
            self.env_id = data.env_id
        return data, targets, mask, node_norm
    

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; apply the linear classifier

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions with the linear classifier applied

        """
        if isinstance(model_output, tuple):
            self.rep_out, self.causal_out, self.spu_out, self.edge_att = model_output
        else:
            self.causal_out = model_output
            self.rep_out, self.spu_out = None, None
        return self.causal_out
    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss based on Mixup algorithm

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func()}
                                   })


        Returns (Tensor):
            loss based on IRM algorithm

        """
        # for i in range(config.dataset.num_envs):
        #     env_idx = data.env_id == i
        self.step += 1
        if self.rep_out is not None:
            # print(f"#IN {mask.size(),mask.sum(),self.rep_out.size(),targets.size(),mask.size()}")
            # print(self.env_id.size(),len(targets.size()),targets.size()[-1])
            if len(targets.size()) >=2 and targets.size()[-1]>1:
                mask = mask
            else:
                mask = mask.view(-1)
            
            att = self.edge_att
            eps = 1e-6
            if config.ood.extra_param[0] <= 0:
                r = self.ratio
            else:
                r = self.get_r(40, 0.1, config.train.epoch, final_r=self.ratio)
            if self.contrast_rep == "raw":
                info_loss = (att * torch.log(att / r + eps) +
                        (1 - att) * torch.log((1 - att) / (1 - r + eps) + eps)).mean()
            else:
                info_loss = 0
            # print("#IN",self.rep_out[mask,:].size(),targets[mask].size())
            causal_loss = config.metric.loss_func(raw_pred, targets, reduction='none') 
            spu_loss = config.metric.loss_func(self.spu_out, targets, reduction='none')
            # print("#IN",causal_loss.sum(),spu_loss.sum())
            # print("#IN",mask.sum(),self.rep_out.size(),targets.size(),mask.size())
            # assert self.rep_out.size(0)==targets[mask].size(0), print(mask.sum(),self.rep_out.size(),targets.size(),mask.size())
                # exit()
            cls_loss = (causal_loss * mask).sum() / mask.sum()
            # contrast_loss = get_contrast_loss(self.rep_out[mask,:],targets[mask].view(-1),sampling="env",env_idx=self.env_id)
            contrast_loss = get_contrast_loss(self.rep_out[mask,:],targets[mask].view(-1))
            if len(config.ood.extra_param)>1:
                # hinge loss
                spu_loss_weight = torch.zeros(spu_loss.size()).to(raw_pred.device)
                spu_loss_weight[spu_loss > causal_loss] = 1.0
                spu_loss_weight = spu_loss_weight * mask
                spu_loss = (spu_loss * spu_loss_weight).sum() / (spu_loss_weight.sum() + 1e-6)
                hinge_loss = spu_loss
            else:
                hinge_loss = 0
            # print(cls_loss, contrast_loss)
            if self.step <= self.anneal_step:
                loss = cls_loss+info_loss
            else:
                loss = cls_loss+info_loss + config.ood.extra_param[0] * contrast_loss + \
                            (config.ood.extra_param[1] if len(config.ood.extra_param)>1 else 0) * hinge_loss
            self.mean_loss = cls_loss
            self.spec_loss = contrast_loss + hinge_loss
        else:
            cls_loss = (config.metric.loss_func(raw_pred, targets, reduction='none') * mask).sum() / mask.sum()

            loss = cls_loss
            self.mean_loss = cls_loss

        return loss

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        return loss
    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r



import copy
from email.policy import default
from enum import Enum
import torch
import argparse
from torch_geometric import data
from torch_geometric.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F


def get_irm_loss(causal_pred, labels, batch_env_idx, criterion=F.cross_entropy):
    device = causal_pred.device
    dummy_w = torch.tensor(1.).to(device).requires_grad_()
    loss_0 = criterion(causal_pred[batch_env_idx == 0] * dummy_w, labels[batch_env_idx == 0])
    loss_1 = criterion(causal_pred[batch_env_idx == 1] * dummy_w, labels[batch_env_idx == 1])
    grad_0 = torch.autograd.grad(loss_0, dummy_w, create_graph=True)[0]
    grad_1 = torch.autograd.grad(loss_1, dummy_w, create_graph=True)[0]
    irm_loss = torch.sum(grad_0 * grad_1)

    return irm_loss


def get_contrast_loss(causal_rep, labels, norm=F.normalize, contrast_t=1.0, sampling="mul",env_idx=None):
    if norm != None:
        causal_rep = norm(causal_rep)
    # TODO: float labels & multi-labels

    if sampling.lower() in ['mul']:
        # imitate https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        is_valid = mask.sum(1) != 0
        mean_log_prob_pos = (mask * log_prob).sum(1)[is_valid] / mask.sum(1)[is_valid]
        # some classes may not be sampled by more than 2
        # loss
        contrast_loss = -mean_log_prob_pos.mean()
    elif sampling.lower() == 'env':
        if env_idx == None:
            env_idx = labels
        # additional leverage the env id into the sampling
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        mask_env = torch.logical_not(torch.eq(env_idx.unsqueeze(1), env_idx.unsqueeze(1).T)).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        mask = mask * mask_env

        neg_mask = torch.logical_not(torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)).float().to(device)
        neg_env = torch.eq(env_idx.unsqueeze(1), env_idx.unsqueeze(1).T).float().to(device)
        neg_mask = neg_mask * neg_env
        
        overall_mask = torch.logical_or(mask,neg_mask)
        # compute log_prob
        exp_logits = torch.exp(logits) #* overall_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        is_valid = mask.sum(1) != 0
        # print(is_valid.sum(),mask.sum(),neg_mask.sum(),overall_mask.sum())
        mean_log_prob_pos = (mask * log_prob).sum(1)[is_valid] / mask.sum(1)[is_valid]
        # some classes may not be sampled by more than 2
        # loss
        contrast_loss = -mean_log_prob_pos.mean()
    return contrast_loss
