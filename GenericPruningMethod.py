import torch
from torch.nn.utils.prune import BasePruningMethod, _validate_pruning_amount_init, _compute_nparams_toprune, \
    _validate_pruning_amount

"""
 this module is a generic extension of the Pytorch pruning module. You can specify a pruning mask
 as you like (shape is one dimensional with length= nr. of weights in the respective layer)
 it only works with one layer
"""


class GenericUnstructured(BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, mask):
        self.mask = mask

    def compute_mask(self, t, default_mask):
        # mask is already provided with constructor
        return self.mask

    @classmethod
    def apply(cls, module, name, mask=None, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """
        return super(GenericUnstructured, cls).apply(
            module, name, mask=mask, importance_scores=importance_scores
        )


class HighestL1Unstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones
    with the highest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=True)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super(L1Unstructured, cls).apply(
            module, name, amount=amount, importance_scores=importance_scores
        )
