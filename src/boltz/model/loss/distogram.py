from typing import Dict, Tuple

import torch
from torch import Tensor
from boltz.data import const


def distogram_loss(
    output: Dict[str, Tensor],
    feats: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute the  distogram loss.

    Parameters
    ----------
    output : Dict[str, Tensor]
        Output of the model
    feats : Dict[str, Tensor]
        Input features

    Returns
    -------
    Tensor
        The globally averaged loss.
    Tensor
        Per example loss.

    """
    # Get predicted distograms
    pred = output["pdistogram"]

    # Compute target distogram
    target = feats["disto_target"]

    # Combine target mask and padding mask
    mask = feats["token_disto_mask"]
    mask = mask[:, None, :] * mask[:, :, None]
    mask = mask * (1 - torch.eye(mask.shape[1])[None]).to(pred)

    # Compute the distogram loss
    errors = -1 * torch.sum(
        target * torch.nn.functional.log_softmax(pred, dim=-1),
        dim=-1,
    )
    denom = 1e-5 + torch.sum(mask, dim=(-1, -2))
    mean = errors * mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    batch_loss = torch.sum(mean, dim=-1)
    global_loss = torch.mean(batch_loss)

    chain_id = feats["asym_id"]
    token_mask = feats["token_disto_mask"]

    token_type = feats["mol_type"]
    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()

    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()
    interface_mask = (
        token_mask[:, None, :] * token_mask[:, :, None]
        * (protein_mask[:, None, :] * protein_mask[:, :, None])
        * (1 - same_chain_mask)
    )

    interface_denom = 1e-5 + torch.sum(interface_mask, dim=(-1, -2))
    interface_loss = torch.sum(errors * interface_mask, dim=-1) / interface_denom[..., None]
    batch_interface_loss = torch.sum(interface_loss, dim=-1)
    global_interface_loss = torch.mean(batch_interface_loss)

    alpha = 0.0 #now interface loss is disabled
    total_batch_loss = batch_loss + alpha * batch_interface_loss
    total_global_loss = global_loss + alpha * global_interface_loss

    return total_global_loss, total_batch_loss
