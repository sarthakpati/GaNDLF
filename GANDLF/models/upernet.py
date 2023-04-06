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
        self.model_2d = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-tiny"
        )

        if self.n_dimensions == 3:
            self.model = self.converter(self.model_2d)
        else:
            self.model = self.model_2d

        # set the number of classes and channels
        self.model.decode_head.classifier.out_channels = self.n_classes
        self.model.backbone.embeddings.patch_embeddings.in_channels = self.n_channels

        self.model.config.__setattr__("num_labels", self.n_classes)
        backbone_config = self.model.config.__getattribute__("backbone_config")
        backbone_config.__setattr__("num_channels", self.n_channels)
        self.model.config.__setattr__("backbone_config", backbone_config)

    def forward(self, x):
        # original_shape = x.shape
        # size_to_transform = self.model.config.__getattribute__(
        #     "backbone_config"
        # ).__getattribute__("image_size")

        # size_to_transform = self.model.config.__getattribute__(
        #     "hidden_size"
        # )

        # if len(original_shape) == 5:
        #     x_transformed = CropOrPad(
        #         (original_shape[1], size_to_transform, size_to_transform, size_to_transform)
        #     )(x)
        # else:
        #     x_transformed = CropOrPad(
        #         (original_shape[1], size_to_transform, size_to_transform)
        #     )(x)

        output_from_model = self.model(x)

        # output_original_shape = CropOrPad(tuple(original_shape[1:]))(
        #     output_from_model.logits
        # )

        return self.apply_final_layer(output_from_model)
