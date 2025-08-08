from typing import Optional
import torch
from torch import nn

from .llm_model import RMSNorm, LlmModel
from .model_config import VLMConfig


class MultiModalProjector(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()

        self.input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_hidden_size, config.hidden_size)
        )

        self.vision_norm = RMSNorm(config.vision_hidden_size)

        self.patches_per_image = int(config.image_size // config.patch_size)
        self.tokens_per_side = int(config.tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        # (batch_size, patches_per_image*patches_per_image, vision_hidden_size)
        batch_size, _, vision_hidden_size = vision_outputs.shape

        # (batch_size, vision_hidden_size, patches_per_image*patches_per_image)
        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        # (batch_size, vision_hidden_size, patches_per_image, patches_per_image)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, vision_hidden_size, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        # (batch_size, vision_hidden_size, tokens_per_side, tokens_per_side)
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        # (batch_size, vision_hidden_size, tokens_per_side*tokens_per_side)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        # (batch_size, tokens_per_side*tokens_per_side, vision_hidden_size)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.vision_norm(pooled_vision_outputs)

        # (batch_size, tokens_per_side*tokens_per_side, hidden_size)
        projected_vision_outputs = torch.matmul(
            normed_vision_outputs,
            self.input_projection_weight.to(normed_vision_outputs.dtype)
        )
        return projected_vision_outputs.type_as(vision_outputs)


class VlmModel(LlmModel):
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.vlm_config = config
        self.vision_tower = config.vision_tower
        self.multi_modal_projector = MultiModalProjector(config)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_patches, vision_hidden_size)
        vision_outputs = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    def get_input_embeddings(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        # if input_ids is not None and self.vlm_config.image_tok >= self.config.vocab_size:
        #     special_image_mask = input_ids == self.vlm_config.image_tok
        #     llm_input_ids = input_ids.clone()
        #     llm_input_ids[special_image_mask] = self.vlm_config.padding_tok
        # else:
        #     llm_input_ids = input_ids

        inputs_embeds = self.embed_tokens(input_ids)
        # (batch_size, channels, height, width)
        pixel_values: Optional[torch.Tensor] = kwargs.get('pixel_values', None)

        if pixel_values is not None:
            # (batch_size, tokens_per_side*tokens_per_side, hidden_size)
            image_features = self.get_image_features(pixel_values)

            # (batch_size, seq_len, 1)
            special_image_mask = (input_ids == self.vlm_config.image_tok).unsqueeze(-1)
            # (batch_size, seq_len, hidden_size)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            # (batch_size, seq_len, hidden_size)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds