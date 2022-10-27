import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

class TSLoss(nn.Module):
    def __init__(self, weights=[1, 1]):
        super(TSLoss, self).__init__()
        self.weights = weights
        
    def forward(self, teacher_features, features, y_pred, labels):
        batch_size = teacher_features.size(0)
        # consistency_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)(F.log_softmax(teacher_features.view(batch_size,-1),1), F.log_softmax(features.view(batch_size,-1),1)) # kl散度loss待测试
        consistency_loss = nn.MSELoss()(teacher_features.view(-1), features.view(-1))
        cls_loss = nn.BCEWithLogitsLoss()(y_pred, labels)
        loss = self.weights[0] * consistency_loss + self.weights[1] * cls_loss
        return loss

from torch.nn.modules.loss import _WeightedLoss
## 继承_WeightedLoss类
class SmoothingBCELossWithLogits(_WeightedLoss):
	def __init__(self, weight=None, reduction='mean', smoothing=0.0):
		super(SmoothingBCELossWithLogits, self).__init__(weight=weight, reduction=reduction)
		self.smoothing = smoothing
		self.weight  = weight
		self.reduction = reduction
	# @staticmethod
	def _smooth(self, targets, smoothing=0.0):
		assert 0 <= smoothing < 1
		with torch.no_grad():
			targets = targets  * (1 - smoothing) + 0.5 * smoothing
        
		return targets

	def forward(self, inputs, targets):
		targets = self._smooth(targets, self.smoothing)
        
        # print(targets)
		loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)
		
		if self.reduction == 'sum':
			loss = loss.item()
		elif self.reduction == 'mean':
			loss = loss.mean()
		return loss

# 多标签分类的focal  loss
class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class DeepAUC(nn.Module):
    def __init__(self, phat):
        super(DeepAUC, self).__init__()
        self.p = phat
        self.margin = 0.7

    def forward(self, mod_out, labels, expt_a, expt_b, alpha):

        logits = torch.sigmoid(mod_out)
        neg_ind = torch.relu(-1*(labels-1))
        phat = torch.sum(labels, dim=0) / labels.shape[0]

        A1 = torch.mean((1-phat)*torch.pow(logits - expt_a, 2)*labels.float(), dim=0)
        A2 = torch.mean(phat*torch.pow(logits - expt_b, 2)*neg_ind.float(), dim=0)
        cross_term = phat*(1-phat)*torch.pow(alpha,2)
        margin_term_1 = 2*(alpha)*(phat*(1-phat)*self.margin + torch.mean(phat*logits*neg_ind.float() - (1-phat)*logits*labels.float(), dim=0))
        # margin_term_2 = 2*(1+alpha)*torch.mean((phat*logits*neg_ind.float() - (1-phat)*logits*labels.float()), dim=0)
        loss = torch.mean(A1 + A2 + margin_term_1 - cross_term)
        return loss
    
    
def info_nce_loss(features, device, n_views=2, temperature=0.07):
#     print(features.shape)
#     print([torch.arange(features.shape[0]) for i in range(n_views)])
    labels = torch.cat([torch.arange(features.shape[0]) for i in range(n_views)], dim=0)
    print('labels', labels.unsqueeze(1))
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    print('labels', labels)
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
#     print(features.shape, features.T.shape)

    similarity_matrix = torch.matmul(features, features.T)
    print('similarity_matrix', similarity_matrix.shape)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    print('mask', mask.shape)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels
        

# def roc_star_loss( _y_true, y_pred, gamma, _epoch_true, epoch_pred):
#         """
#         Nearly direct loss function for AUC.
#         See article,
#         C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
#         https://github.com/iridiumblue/articles/blob/master/roc_star.md
#             _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
#             y_pred: `Tensor` . Predictions.
#             gamma  : `Float` Gamma, as derived from last epoch.
#             _epoch_true: `Tensor`.  Targets (labels) from last epoch.
#             epoch_pred : `Tensor`.  Predicions from last epoch.
#         """
#         #convert labels to boolean
#         y_true = (_y_true>=0.50)
#         epoch_true = (_epoch_true>=0.50)

#         # if batch is either all true or false return small random stub value.
#         if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred)*1e-8

#         pos = y_pred[y_true]
#         neg = y_pred[~y_true]

#         epoch_pos = epoch_pred[epoch_true]
#         epoch_neg = epoch_pred[~epoch_true]

#         # Take random subsamples of the training set, both positive and negative.
#         max_pos = 1000 # Max number of positive training samples
#         max_neg = 1000 # Max number of positive training samples
#         cap_pos = epoch_pos.shape[0]
#         cap_neg = epoch_neg.shape[0]
#         epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
#         epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

#         ln_pos = pos.shape[0]
#         ln_neg = neg.shape[0]

#         # sum positive batch elements agaionst (subsampled) negative elements
#         if ln_pos>0 :
#             pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
#             neg_expand = epoch_neg.repeat(ln_pos)

#             diff2 = neg_expand - pos_expand + gamma
#             l2 = diff2[diff2>0]
#             m2 = l2 * l2
#             len2 = l2.shape[0]
#         else:
#             m2 = torch.tensor([0], dtype=torch.float).cuda()
#             len2 = 0

#         # Similarly, compare negative batch elements against (subsampled) positive elements
#         if ln_neg>0 :
#             pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
#             neg_expand = neg.repeat(epoch_pos.shape[0])

#             diff3 = neg_expand - pos_expand + gamma
#             l3 = diff3[diff3>0]
#             m3 = l3*l3
#             len3 = l3.shape[0]
#         else:
#             m3 = torch.tensor([0], dtype=torch.float).cuda()
#             len3=0

#         if (torch.sum(m2)+torch.sum(m3))!=0 :
#            res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
#            #code.interact(local=dict(globals(), **locals()))
#         else:
#            res2 = torch.sum(m2)+torch.sum(m3)

#         res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

#         return res2

class LovaszLoss(nn.Module):
    # 用于分类的Lovasz loss
    def __init__(self, ):
        super(LovaszLoss, self).__init__()

    def forward(self, logits, labels, ignore=None):

        loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels, ignore))
        return loss
    
    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
        logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss


    def flatten_binary_scores(self, logits, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = logits.view(-1)
        # scores = logits.sigmoid().view(-1) # 先sigmoid再flatten
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels
    
    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum().float()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard