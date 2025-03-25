import torch
import torch.nn as nn
from hydra.utils import instantiate

# Assuming you already have Hiera, FpnNeck, and PositionEmbeddingSine imports


class HieraEncoder(nn.Module):
    def __init__(
        self,
        trunk,  # Trunk (Hiera)
        neck,  # Neck (FpnNeck)
        scalp: int = 1,  # Scalp parameter
        ckpt_path: str = None,  # Optional checkpoint path
    ):
        super().__init__()

        # Assigning the trunk, neck, and scalp
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp

        # Ensure trunk and neck have matching channel lists
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, Neck: {self.neck.backbone_channel_list}"

        # If a checkpoint path is provided, load the pretrained weights
        if ckpt_path is not None:
            self.load_pretrained_weights(ckpt_path)

    def forward(self, sample: torch.Tensor):
        # Forward pass through ImageEncoder components
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Optionally discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        # Return only 'vision_features'
        # for i, feature in enumerate(features):
        #     print(f"Feature {i}: {feature.shape}")
        #return features[-1]

        return {f'res{i+2}': features[i] for i in range(len(features))}

    def output_shape(self):
        backbone_feature_shape = dict()
        # for name in self._out_features:
        #     backbone_feature_shape[name] = dict(
        #         {
        #             "channel": self._out_feature_channels[name],
        #             "stride": self._out_feature_strides[name],
        #         }
        #     )
        backbone_feature_shape = {'res2': {'channel': 256, 'stride': 4}, 
                                  'res3': {'channel': 256, 'stride': 8}, 
                                  'res4': {'channel': 256, 'stride': 16}, 
                                  'res5': {'channel': 256, 'stride': 32}}
        return backbone_feature_shape

    def load_pretrained_weights(self, ckpt_path: str):
        """
        Loads pretrained weights from a checkpoint, extracting the image_encoder parts.
        """
        # Load the full checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        sam2_checkpoint = checkpoint["model"]

        image_encoder_state_dict = self.state_dict()
        model_keys = set(image_encoder_state_dict.keys())

        # Extract image_encoder weights from the checkpoint
        new_state_dict = {}
        for k, v in sam2_checkpoint.items():
            if (
                k.startswith("image_encoder.")
                and v.shape
                == image_encoder_state_dict[k.replace("image_encoder.", "")].shape
            ):
                new_state_dict[k.replace("image_encoder.", "")] = v
                print(f"Extracted {k} for image_encoder")

        # Update the image_encoder state_dict
        print("difference", model_keys - set(new_state_dict.keys()))

        # Load the extracted image_encoder weights into the model
        self.load_state_dict(new_state_dict, strict=True)
        print(f"Loaded pretrained weights from {ckpt_path}")
