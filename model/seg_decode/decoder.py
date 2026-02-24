import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_util import PointNetFeaturePropagation

class PointCrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        query_dim: int = 512,
        point_feat_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.point_proj = nn.Linear(point_feat_dim, hidden_dim)

        self.pos_enc = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
                ),
                "norm_cross": nn.LayerNorm(hidden_dim),

                "self_attn": nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
                ),
                "norm_self": nn.LayerNorm(hidden_dim),

                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
                ),
                "norm_ffn": nn.LayerNorm(hidden_dim),
            }))

        self.score_proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.score_proj_p = nn.Linear(hidden_dim, hidden_dim)

        init_temp = 1.0 / math.sqrt(hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def forward(
        self,
        queries: torch.Tensor,
        point_features: torch.Tensor,
        xyz: torch.Tensor = None,
    ) -> torch.Tensor:
        Q = self.query_proj(queries)
        P = self.point_proj(point_features)

        if xyz is not None:
            P = P + self.pos_enc(xyz)

        Q = Q.unsqueeze(0)
        P = P.unsqueeze(0)

        for layer in self.layers:
            Q_norm = layer["norm_cross"](Q)
            cross_out, _ = layer["cross_attn"](Q_norm, P, P)
            Q = Q + cross_out

            Q_norm = layer["norm_self"](Q)
            self_out, _ = layer["self_attn"](Q_norm, Q_norm, Q_norm)
            Q = Q + self_out

            Q_norm = layer["norm_ffn"](Q)
            Q = Q + layer["ffn"](Q_norm)

        Q = Q.squeeze(0)
        P = P.squeeze(0)

        Q_score = self.score_proj_q(Q)
        P_score = self.score_proj_p(P)

        Q_score = F.normalize(Q_score, p=2, dim=-1)
        P_score = F.normalize(P_score, p=2, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * torch.matmul(Q_score, P_score.T)

        return logits
    
class PartSegmentationEmbHead(nn.Module):
    def __init__(self, embed_dim=512, mlp=[512, 512]):
        super().__init__()
        in_channel = 3 * embed_dim
        self.propagation = PointNetFeaturePropagation(in_channel, mlp)

    def forward(self, xyz, centers, H4, H8, H12):
        B, N, _ = xyz.shape

        fused = torch.cat([H4, H8, H12], dim=-1) 

        fused = fused.permute(0, 2, 1)
        centers = centers.permute(0, 2, 1)
        xyz_t = xyz.permute(0, 2, 1)

        point_features = self.propagation(xyz_t, centers, None, fused)
        point_features = point_features.permute(0, 2, 1)
        return point_features
