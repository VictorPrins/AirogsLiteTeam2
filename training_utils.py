from transformers import Trainer, TrainingArguments
import torch
import torchmetrics.functional as torchmetrics
from sklearn import metrics
import sys
import numpy as np

# override the default trainer to use a weighted loss to compensate for our dataset imbalance
# refer to https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch
# and https://huggingface.co/docs/transformers/main/main_classes/trainer

class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        # this is needed to compute the class weights
        samples_per_class = torch.tensor(kwargs.pop('samples_per_class'))
        self.device = kwargs['args'].device

        super().__init__(*args, **kwargs)

        # this is a tensor of size two with the class weights to compensate for class imbalance
        # [1, 9]
        self.class_weights = torch.max(samples_per_class) / samples_per_class


    def compute_loss(self, model, inputs, return_outputs=False):

        # labels is size [batch_size]
        labels = inputs.get('labels').to(torch.float)

        # forward pass
        outputs = model(**inputs)

        # get logits and squeeze into size [batch_size]
        logits = outputs.get('logits').squeeze(dim=1)

        
        imbalance_weights = torch.zeros(labels.size()).to(self.device)

        for idx, label in enumerate(labels):
            imbalance_weights[idx] = self.class_weights[int(label)]

        loss_fn = torch.nn.BCEWithLogitsLoss(weight=imbalance_weights)

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# https://github.com/qurAI-amsterdam/airogs-evaluation/blob/33e6e9fb108a665b1f23ff2f1bc6b85584a98f74/evaluation.py
def partial_auc(y_pred, y_true, min_spec):    
    return metrics.roc_auc_score(y_true, y_pred, max_fpr=(1 - min_spec))


# https://github.com/qurAI-amsterdam/airogs-evaluation/blob/33e6e9fb108a665b1f23ff2f1bc6b85584a98f74/evaluation.py
def sens_at_spec(y_pred, y_true, at_spec, eps=sys.float_info.epsilon):

    fpr, tpr, threshes = metrics.roc_curve(y_true, y_pred, drop_intermediate=False)
    spec = 1 - fpr

    operating_points_with_good_spec = spec >= (at_spec - eps)
    max_tpr = tpr[operating_points_with_good_spec][-1]

    operating_point = np.argwhere(operating_points_with_good_spec).squeeze()[-1]
    operating_tpr = tpr[operating_point]

    assert max_tpr == operating_tpr or (np.isnan(max_tpr) and np.isnan(operating_tpr)), f'{max_tpr} != {operating_tpr}'
    assert max_tpr == max(tpr[operating_points_with_good_spec]) or (np.isnan(max_tpr) and max(tpr[operating_points_with_good_spec])), \
        f'{max_tpr} == {max(tpr[operating_points_with_good_spec])}'

    return max_tpr


def compute_metrics(p):
    # the argument is a transformers.EvalPrediction, with keys
    # predictions and label_ids (both numpy arrays)
    # Must return a dictionary with the metric name as key and its value as value

    preds = torch.tensor(p.predictions).squeeze(dim=1).to(torch.float)
    targets = torch.tensor(p.label_ids).to(torch.int32)
    
    precision, recall = torchmetrics.precision_recall(preds, targets)

    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    return {
        'kappa': torchmetrics.classification.binary_cohen_kappa(preds, targets),
        'f1_score': torchmetrics.classification.binary_f1_score(preds, targets),
        'jaccard': torchmetrics.classification.binary_jaccard_index(preds, targets),
        'precision': precision,
        'recall/tpr/sensitivity': recall,
        'specificity/tnr': torchmetrics.classification.binary_specificity(preds, targets),
        'sensitivity@95': sens_at_spec(preds_np, targets_np, 0.95),
        'AUROC@90': partial_auc(preds_np, targets_np, 0.9)
    }