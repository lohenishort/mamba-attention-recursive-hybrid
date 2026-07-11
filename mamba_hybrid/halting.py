import torch
import torch.nn as nn
from mamba_hybrid.config import MambaHybridConfig


class ACTHaltingModule(nn.Module):
    def __init__(self, config: MambaHybridConfig) -> None:
        super().__init__()
        self.d_model: int = config.d_model
        self.bce_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )
        # Initialize bias of the last linear layer to -5.0 to start with low halting probability
        last_layer = self.bce_mlp[-1]
        assert isinstance(last_layer, nn.Linear)
        assert last_layer.bias is not None
        nn.init.constant_(last_layer.bias, -5.0)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the halting probability for each batch element.

        Args:
            z: Latent planning state of shape [batch_size, seq_len_z, d_model]
            y: Answer prediction state of shape [batch_size, seq_len_y, d_model]

        Returns:
            prob: Halting probabilities of shape [batch_size]
        """
        # z: [batch_size, seq_len_z, d_model]
        # y: [batch_size, seq_len_y, d_model]

        # Concatenate along the sequence dimension: [batch_size, seq_len_z + seq_len_y, d_model]
        concat_state = torch.cat([z, y], dim=1)

        # Global average pooling over the sequence dimension: [batch_size, d_model]
        # Detach to prevent gradients from flowing back into representations z and y
        s_t = concat_state.mean(dim=1).detach()

        # Pass through MLP: [batch_size, 1] -> [batch_size]
        bce_logit = self.bce_mlp(s_t).squeeze(-1)

        # Map to probability space [0, 1]
        prob = torch.sigmoid(bce_logit)  # [batch_size]

        return prob
