from typing import Tuple

import torch
import torch.nn as nn


def calc_vae_loss(distance: nn.Module, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                  target, options: dict) -> torch.Tensor:
    """
    Calculate the vae loss
    TODO: complete this function.
    :param options: Options
    :param distance: The distance to use for reconstruction loss
    :param input: tuple consisting of (decoded image, latent_vector, mu, log_var)
    :param target: target image (torch.Tensor)
    :rtype vae_loss: torch.Tensor
    """

    decoded_image, latent_vector, mu, log_var = input




    KLDiv_loss = -0.5 * torch.sum(1 + log_var - mu **2 - log_var.exp())
    KLDiv_loss = torch.mean(KLDiv_loss)
    return distance(decoded_image, target) + options["lambda"] * KLDiv_loss
