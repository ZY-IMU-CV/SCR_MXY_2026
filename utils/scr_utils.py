"""SCR method utilities: one-time SDI and CLIP-guided, instance-level PCR.

These functions deliberately contain no LLM class-level semantic relations.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def semantic_decoupled_init(text_features, topk=2, alpha=0.7):
    """Decouple *normalized CLIP text prototypes* as in SCR Eqs. (1)-(2).

    The non-self top-k semantic neighbors are computed in the original text
    embedding space. Projection to a visual classifier, if needed, occurs later.
    """
    t = F.normalize(text_features.float(), dim=-1)
    num_classes = t.size(0)
    if num_classes <= 1:
        return t.to(dtype=text_features.dtype)
    k = max(1, min(int(topk), num_classes - 1))
    similarity = t @ t.t()
    similarity.fill_diagonal_(float('-inf'))
    hard_neighbors = similarity.topk(k, dim=-1).indices
    center = t[hard_neighbors].mean(dim=1)
    new_prototypes = F.normalize(t + float(alpha) * (t - center), dim=-1)
    return new_prototypes.to(dtype=text_features.dtype)


@torch.no_grad()
def build_pcr_targets(zs_model, images, labels, topk=5):
    """Form targets {ground truth} U {top-k *other* zero-shot CLIP classes}."""
    scores = zs_model(images).float()
    num_classes = scores.size(1)
    targets = torch.zeros_like(scores)
    labels = labels.long().view(-1, 1)
    targets.scatter_(1, labels, 1.)
    if num_classes > 1 and int(topk) > 0:
        # No ground truth class may appear among the confusing classes.
        scores.scatter_(1, labels, float('-inf'))
        indices = scores.topk(min(int(topk), num_classes - 1), dim=1).indices
        targets.scatter_(1, indices, 1.)
    return targets


def pcr_binary_la_loss(logits, targets, log_prior, tau_b=1.0):
    """Mean BCE-with-logits using binary logit-adjustment (SCR Eq. 5)."""
    adjusted_logits = logits.float() + float(tau_b) * log_prior
    return F.binary_cross_entropy_with_logits(adjusted_logits, targets.float())
