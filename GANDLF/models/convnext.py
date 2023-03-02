# -*- coding: utf-8 -*-

from .modelBase import ModelBase
from transformers import ConvNextConfig, ConvNextModel

class ConvNeXT(ModelBase):
    def __init__(
        self,
        parameters: dict,
        ):

        # translate parameters to ConvNextConfig

        # Initializing a ConvNext convnext-tiny-224 style configuration
        configuration = ConvNextConfig()
        """ default config
        ConvNextConfig {
            "depths": [
                3,
                3,
                9,
                3
            ],
            "drop_path_rate": 0.0,
            "hidden_act": "gelu",
            "hidden_sizes": [
                96,
                192,
                384,
                768
            ],
            "image_size": 224,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "layer_scale_init_value": 1e-06,
            "model_type": "convnext",
            "num_channels": 3,
            "num_stages": 4,
            "out_features": null,
            "patch_size": 4,
            "stage_names": [
                "stem",
                "stage1",
                "stage2",
                "stage3",
                "stage4"
            ],
            "transformers_version": "4.26.1"
        }
        """

        # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        self.model = ConvNextModel(configuration)

        if self.n_dimensions == 3:
            self.model = self.converter(self.model)
    
    def forward(self, x):

        return self.model(x)