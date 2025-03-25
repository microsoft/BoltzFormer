# Import required libraries
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import os
import numpy as np
from safetensors.torch import load_file

class MaskFormerHead(nn.Module):
    def __init__(self, pixel_decoder, predictor):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = predictor
        # self.classes = [
        #     "liver",
        #     "lung",
        #     "kidney",
        #     "pancreas",
        #     "heart anatomies",
        #     "brain anatomies",
        #     "eye anatomies",
        #     "vessel",
        #     "other organ",
        #     "tumor",
        #     "infection",
        #     "other lesion",
        #     "fluid disturbance",
        #     "other abnormality",
        #     "histology structure",
        #     "other",
        #     "background",
        # ]

    def forward(self, image_features, text=None):
        # if hasattr(self.predictor, "language_encoder"):
        #     self.predictor.language_encoder.get_text_embeddings(
        #         self.classes, is_eval=False
        #     )
        mask_features, transformer_encoder_features, multi_scale_features = (
            self.pixel_decoder.forward_features(image_features)
        )

        extra = {}
        class_emb = None
        logit_scale = None

        if hasattr(self.predictor, "language_encoder"):
            logit_scale = self.predictor.language_encoder.logit_scale
            if text is not None:
                gtext = self.predictor.language_encoder.get_text_token_embeddings(
                    text, name="grounding", token=False, norm=False
                )
                token_emb = gtext["token_emb"]
                tokens = gtext["tokens"]
                class_emb = gtext["class_emb"]
                query_emb = nn.utils.rnn.pad_sequence(
                    [
                        _token_emb[_tokens.bool()]
                        for _token_emb, _tokens in zip(
                            token_emb, tokens["attention_mask"]
                        )
                    ],
                    padding_value=-1,
                )

                non_zero_query_mask = query_emb.sum(dim=-1) == -query_emb.shape[-1]
                query_emb[non_zero_query_mask] = 0

                extra["grounding_tokens"] = query_emb
                extra["grounding_nonzero_mask"] = non_zero_query_mask.t()

        predictions = self.predictor(
            x=multi_scale_features, mask_features=mask_features, extra=extra
        )

        predictions["class_emb"] = class_emb
        predictions["logit_scale"] = logit_scale

        return predictions

    def forward_eval(
        self, image_features, text=None
    ):
        # if hasattr(self.predictor, "language_encoder"):
        #     self.predictor.language_encoder.get_text_embeddings(
        #         self.classes, is_eval=True
        #     )
        mask_features, transformer_encoder_features, multi_scale_features = (
            self.pixel_decoder.forward_features(image_features)
        )

        extra = {}
        class_emb = None
        logit_scale = None

        if hasattr(self.predictor, "language_encoder"):
            logit_scale = self.predictor.language_encoder.logit_scale
            if text is not None:
                gtext = self.predictor.language_encoder.get_text_token_embeddings(
                    text, name="grounding", token=False, norm=False
                )
                token_emb = gtext["token_emb"]
                tokens = gtext["tokens"]
                class_emb = gtext["class_emb"]
                query_emb = nn.utils.rnn.pad_sequence(
                    [
                        _token_emb[_tokens.bool()]
                        for _token_emb, _tokens in zip(
                            token_emb, tokens["attention_mask"]
                        )
                    ],
                    padding_value=-1,
                )

                non_zero_query_mask = torch.zeros(
                    query_emb.shape[:-1], dtype=torch.bool
                )
                extra["grounding_tokens"] = query_emb
                extra["grounding_nonzero_mask"] = non_zero_query_mask.t()

        predictions = self.predictor(
            x=multi_scale_features, mask_features=mask_features, extra=extra
        )

        predictions["class_emb"] = class_emb
        predictions["logit_scale"] = logit_scale
        return predictions

    def override_input_shape(self, input_shape):
        self.pixel_decoder.override_input_shape(input_shape)


class BoltzFormerModel(nn.Module):
    def __init__(
        self,
        backbone,
        sem_seg_head,
        pixel_mean=[123.675, 116.280, 103.530],
        pixel_std=[58.395, 57.120, 57.375],
        convolute_outputs=True,  # for upscaling the output of the model
        out_channels_1=10,  # Parameter for upscaling
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.sem_seg_head.override_input_shape(backbone.output_shape())
        self.pixel_mean = torch.tensor(pixel_mean).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(1, 3, 1, 1)
        self.convolute_outputs = convolute_outputs

        if self.convolute_outputs:
            self.output_deconv = nn.ConvTranspose2d(
                in_channels=self.sem_seg_head.predictor.num_queries + 3,
                out_channels=out_channels_1,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            # output convolution that doesn't change dimension but just channels
            self.output_conv = nn.Conv2d(
                in_channels=out_channels_1 + 3, out_channels=1, kernel_size=1
            )

    def convolution_procedure(self, image, pred_gmasks):
        """
        This function is for upscaling the output of the model to the original image size.
        """

        image_256 = F.interpolate(
            image, size=(256, 256), mode="bilinear", align_corners=False
        )  # bs, 3, 256, 256
        image_512 = F.interpolate(
            image, size=(512, 512), mode="bilinear", align_corners=False
        )  # bs, 3, 512, 512

        mean_mask = pred_gmasks.mean(dim=1, keepdim=True)  # bs, 1, 256, 256
        # bs, num_queries, 256, 256
        pred_gmasks_512 = F.interpolate(
            mean_mask,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )  # bs, 1, 512, 512

        stack_256 = torch.cat(
            (image_256, pred_gmasks), dim=1
        )  # bs , num_queries+3, 256, 256

        deconv_output = self.output_deconv(stack_256)  # bs, 10, 512, 512

        # concatenation with image_512
        stack_512 = torch.cat((image_512, deconv_output), dim=1)  # bs, 13, 512, 512
        outputs_1_channel = self.output_conv(stack_512)  # bs, 1, 512, 512

        stacked_tensor = torch.cat((outputs_1_channel, pred_gmasks_512), dim=1)
        averaged_results = stacked_tensor.mean(dim=1, keepdim=True)  # Take mean

        return averaged_results

    def forward_train(self, inputs):
        image = inputs["image"] if "image" in inputs else None
        text = inputs["text"] if "text" in inputs else None

        if image is None:
            raise ValueError("Image is required input")

        pixel_mean = self.pixel_mean.to(image.device)
        pixel_std = self.pixel_std.to(image.device)
        image = (image - pixel_mean) / pixel_std
        image_embedding = self.backbone(image)
        outputs = self.sem_seg_head.forward(
            image_features=image_embedding, text=text
        )

        if self.convolute_outputs:
            outputs["pred_gmasks"] = self.convolution_procedure(image, outputs["pred_gmasks"])
        else:
            outputs["pred_gmasks"] = outputs["pred_gmasks"].mean(dim=1, keepdim=True)

        results = {"predictions": outputs}

        return results

    def forward_eval(self, inputs):
        image = inputs["image"] if "image" in inputs else None
        text = inputs["text"] if "text" in inputs else None

        if image is None:
            raise ValueError("Image is required input")

        pixel_mean = self.pixel_mean.clone().detach().view(1, 3, 1, 1).to(image.device)
        pixel_std = self.pixel_std.clone().detach().view(1, 3, 1, 1).to(image.device)
        image = (image - pixel_mean) / pixel_std
        image_embedding = self.backbone(image)
        outputs = self.sem_seg_head.forward_eval(
            image_features=image_embedding, text=text
        )

        if self.convolute_outputs:
            outputs["pred_gmasks"] = self.convolution_procedure(image, outputs["pred_gmasks"])
        else:
            outputs["pred_gmasks"] = outputs["pred_gmasks"].mean(dim=1, keepdim=True)

        results = {"predictions": outputs}

        return results

    def forward(self, inputs, mode="train"):
        if mode == "train":
            return self.forward_train(inputs)
        elif mode == "eval":
            return self.forward_eval(inputs)
        else:
            raise ValueError(f"Unknown mode {mode}. Use 'train' or 'eval'.")

    def load_pretrained(self, checkpoint_path):
        """Loads a pretrained checkpoint into the model."""
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            state_dict = {
                k[6:] if k.startswith("model.") else k: v
                for k, v in state_dict.items()
            }
        self.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")
