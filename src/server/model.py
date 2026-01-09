"""
Adapted from CHIMERA Task 2 baseline 
https://github.com/DIAGNijmegen/CHIMERA/tree/main/task2_baseline/prediction_model/Aggregators/training/mil_models
model_abmil_fusion.py
tabular_snn.py
"""
import torch
import torch.nn as nn

from embedRNA import RNANet
from embedCD import CDNet

class FusionNet(nn.Module):
    def __init__(
            self,
            rna_in_dim=19359,
            clinical_in_dim=14, 
            embedding_dim=512,
            gate=True, 
            dropout_CD=0.3,
            dropout_RNA=0.3
        ):
        super().__init__()
        self.gate = gate

        # === RNA embedding branch with configurable dropout ===
        self.rna_out_dim = embedding_dim - clinical_in_dim
        self.rna_embedding = RNANet(
            in_dim=rna_in_dim, 
            out_dim=self.rna_out_dim, 
            dropout_p=dropout_RNA
        )

        # === Clinical branch with LayerNorm (safe for batch size = 1) ===
        # TODO: why batch size = 1?
        self.tabular_net = CDNet(
            in_dim=clinical_in_dim, 
            dropout_p=0.3
        )

        # Interpretable per-modality importance weights
        # TODO: use nn.Parameter to create importance weights
        
        # Attention backbone
        self.attention = nn.Linear(512, 1)

        # === Safe dummy forward to get output dimension ===
        with torch.no_grad():
            self.tabular_net.eval()  # prevent any norm layers from training stats
            dummy_input = torch.zeros(1, clinical_in_dim)
            clinical_out_dim = self.tabular_net(dummy_input).shape[1]
            self.tabular_net.train()

        # === Fusion (binary) classifier ===
        self.classifier = nn.Linear(512 + clinical_out_dim, 1)

    def forward(self, x_clinical, x_rna=None):
        
        if x_rna is not None:
            # RNA embedding
            h = self.rna_embedding(x_rna)
            a = torch.softmax(self.attention(h), dim=1)
            z = torch.sum(a * h, dim=1)
        else:
            B = x_clinical.shape[0]
            z = torch.zeros((B, self.rna_out_dim))

        # Clinical embedding
        z_tab = self.tabular_net(x_clinical)

        # TODO: apply per-modality importance weights

        # Fusion
        z_fusion = torch.cat((z, z_tab), dim=-1)

        # TODO: apply attention backbone

        # Optional gating
        if self.gate:
            z_tab = 0.5 * z_tab  # less aggressive scaling than 0.2
        
        logits = self.classifier(z_fusion)

        return {'logits': logits, 'loss': None, 'attention': a}

    def attention_entropy_loss(self, attention_weights):
        """Optional regularization to avoid overconfident attention."""
        entropy = -torch.mean(
            torch.sum(attention_weights * torch.log(attention_weights + 1e-6), dim=1)
        )
        return entropy