# -*- coding: utf-8 -*-


from .modelBase import ModelBase
from transformers import UperNetForSemanticSegmentation
from torchio.transforms import CropOrPad


class UPerNet(ModelBase):
    def __init__(
        self,
        parameters: dict,
    ):
        super(UPerNet, self).__init__(parameters)
        # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-tiny"
        )

        if self.n_dimensions == 3:
            self.model = self.converter(self.model)

    def forward(self, x):
        original_shape = x.shape
        size_to_transform = self.model.config.__getattribute__(
            "backbone_config"
        ).__getattribute__("image_size")

        x_transformed = CropOrPad(
            (original_shape[1], size_to_transform, size_to_transform)
        )(x)

        output_from_model = self.model(pixel_values=x_transformed)
        output_original_shape = CropOrPad(tuple(original_shape[1:]))(
            output_from_model.logits
        )

        return self.apply_final_layer(output_original_shape)
