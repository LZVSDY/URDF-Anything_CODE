import torch
import torch.nn.functional as F

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float = 1.0,
    eps: float = 1e-6,
):
    """
    Compute the DICE loss for binary masks.
    
    Args:
        inputs: [K, N] - predicted logits or probabilities (after sigmoid if needed)
        targets: [K, N] - ground truth binary masks (0 or 1)
        num_masks: number of valid masks (K) for normalization
    """

    assert inputs.shape == targets.shape, f"{inputs.shape} vs {targets.shape}"

    # inputs: [K, N], targets: [K, N]
    numerator = 2 * (inputs * targets).sum(dim=-1)      # [K]
    denominator = inputs.sum(dim=-1) + targets.sum(dim=-1)  # [K]

    loss_per_mask = 1 - (numerator + eps) / (denominator + eps)  # [K]
    total_loss = loss_per_mask.sum() / (num_masks + eps)

    return total_loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

if __name__ == "__main__":
    pred_mask, gt_mask = torch.randn((2, 2028)), torch.randn((2, 2028))
    loss = dice_loss(pred_mask, gt_mask)
    print(loss)