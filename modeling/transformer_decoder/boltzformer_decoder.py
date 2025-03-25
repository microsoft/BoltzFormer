import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .position_encoding import PositionEmbeddingSine
from .modules import (
    SelfAttentionLayer,
    CrossAttentionLayer,
    FFNLayer,
    MLP,
)


class BoltzFormerDecoder(nn.Module):
    def __init__(
        self,
        language_encoder: nn.Module,
        in_channels=256,
        mask_classification=True,
        hidden_dim: int = 256,
        dim_proj: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 10,
        pre_norm: bool = False,
        mask_dim: int = 256,
        enforce_input_project: bool = False,
        boltzmann_sampling: dict = {
            "mask_threshold": 0.5,
            "do_boltzmann": False,
            "sample_ratio": 0.1,
            "base_temp": 1,
        },
        pre_self_attention: bool = False,
    ):
        super().__init__()

        self.language_encoder = language_encoder
        self.in_channels = in_channels
        self.mask_classification = mask_classification
        self.hidden_dim = hidden_dim
        self.dim_proj = dim_proj
        self.num_queries = num_queries
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.dec_layers = dec_layers
        self.pre_norm = pre_norm
        self.mask_dim = mask_dim
        self.enforce_input_project = enforce_input_project

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        if pre_self_attention:
            self.initial_self_attention_layer = SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )

            self.initial_ffn_layer = FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                normalize_before=pre_norm,
            )

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                )
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.query_index = {}

        self.language_encoder = language_encoder
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=0.02)

        self.boltzmann_sampling = boltzmann_sampling
        self.pre_self_attention = pre_self_attention

    def forward(
        self,
        x,  # x is a list of multi-scale features
        mask_features,
        extra={},
    ):
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos_embed = self.pe_layer(x[i], None).flatten(2).permute(2, 0, 1)
            pos.append(pos_embed)
            proj_src = (
                self.input_proj[i](x[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )
            src.append(proj_src.permute(2, 0, 1))

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        predictions_mask = []

        text_output = extra["grounding_tokens"]
        text_embed = text_output.detach().clone()

        # Pre Self Attention between queries and text tokens
        if self.pre_self_attention:
            combined_output = torch.cat([output, text_output], dim=0)
            combined_embed = torch.cat([query_embed, text_embed], dim=0)

            combined_output, avg_self = self.initial_self_attention_layer(
                combined_output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=combined_embed,
            )

            # Separate queries and text tokens again after self-attention
            output = combined_output[: self.num_queries]
            text_output = combined_output[self.num_queries :]

            # FFN Layer
            output = self.initial_ffn_layer(output)

        # Initial prediction heads
        outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )

        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # CROSS ATTENTION
            output, avg_cross = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # No masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            # Self Attention
            combined_output = torch.cat([output, text_output], dim=0)
            combined_embed = torch.cat([query_embed, text_embed], dim=0)

            combined_output, avg_self = self.transformer_self_attention_layers[i](
                combined_output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=combined_embed,
            )

            # Separate queries and text tokens again after self-attention
            output = combined_output[: self.num_queries]
            text_output = combined_output[self.num_queries :]

            # FFN Layer
            output = self.transformer_ffn_layers[i](output)

            outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                layer_id=i,
            )
            predictions_mask.append(outputs_mask)

        out = {
            "pred_gmasks": predictions_mask[-1],
        }
        return out

    def forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size, layer_id=-1
    ):
        decoder_output = self.decoder_norm(output)
        assert not torch.isnan(
            decoder_output
        ).any(), f"NaN detected after decoder_norm in layer {layer_id}"
        decoder_output = decoder_output.transpose(0, 1)

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
        ).detach()

        # Boltzman sampling on attention mask
        threshold = self.boltzmann_sampling[
            "mask_threshold"
        ]  # original threshold for masked attention
        do_boltzmann = self.boltzmann_sampling[
            "do_boltzmann"
        ]  # whether to do Boltzman sampling
        sample_ratio = self.boltzmann_sampling[
            "sample_ratio"
        ]  # number of iid samples as a ratio of total number of masked tokens
        base_temp = self.boltzmann_sampling[
            "base_temp"
        ]  # base temperature for Boltzman sampling
        
        if do_boltzmann:
            # probability of Boltzman sampling
            Temp = base_temp / (
                2 + layer_id
            )  # temperature decays with layer number (first layer from id -1)
            boltzmann_prob = torch.exp(attn_mask / Temp)
            boltzmann_prob = (
                boltzmann_prob * (attn_mask < threshold).float()
            )  # remove unmasked regions
            boltzmann_prob = boltzmann_prob / boltzmann_prob.sum(dim=-1, keepdim=True)

            # sample from Boltzman distribution n times
            n_samples = int(
                attn_mask.shape[-1] * sample_ratio
            )  # number of iid samples on the tokens
            masked_prob = (
                1 - boltzmann_prob
            ) ** n_samples  # probability that each token is still masked after n iid samples
            boltzmann_mask = (torch.rand_like(boltzmann_prob) < masked_prob).bool()

            # combine with original mask
            attn_mask = torch.logical_and(
                (attn_mask < threshold).bool(), boltzmann_mask
            )

        else:
            attn_mask = (attn_mask < threshold).bool()

        assert not torch.isnan(
            attn_mask
        ).any(), f"NaN detected in attn_mask in layer {layer_id}"

        return outputs_mask, attn_mask

    def override_in_channels(self, in_channels):
        # reinitialize the model with a new input shape but use all other remaining parameters
        # this is useful for model conversion
        self.__init__(
            language_encoder=self.language_encoder,
            in_channels=in_channels,
            mask_classification=self.mask_classification,
            hidden_dim=self.hidden_dim,
            dim_proj=self.dim_proj,
            num_queries=self.num_queries,
            nheads=self.nheads,
            dim_feedforward=self.dim_feedforward,
            dec_layers=self.dec_layers,
            pre_norm=self.pre_norm,
            mask_dim=self.mask_dim,
            enforce_input_project=self.enforce_input_project,
        )
