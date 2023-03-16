# -*- coding: utf-8 -*-

from .modelBase import ModelBase
from transformers import ConvNextConfig, ConvNextModel


class ConvNeXT(ModelBase):
    def __init__(
        self,
        parameters: dict,
    ):
        super(ConvNeXT, self).__init__(parameters)
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

        # translate parameters to ConvNextConfig
        configuration.__setattr__("depths", parameters["model"].get("depths", [3, 3]))
        configuration.__setattr__(
            "hidden_sizes", parameters["model"].get("hidden_sizes", [96, 192])
        )
        configuration.__setattr__(
            "num_stages", parameters["model"].get("num_stages", 2)
        )
        configuration.__setattr__(
            "stage_names", parameters["model"].get("stage_names", ["stem", "stage1"])
        )
        # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        self.model = ConvNextModel(configuration)

        if self.n_dimensions == 3:
            self.model = self.converter(self.model)

    def forward(self, x):
        output_from_convnext = self.model(pixel_values=x)
        # output_from_convnext.keys()
        # odict_keys(['last_hidden_state', 'pooler_output'])
        # none of these are corresponding to the size of the input image, which is [128,128]

        pooler_output_after_final_layer = self.final_convolution_layer(
            output_from_convnext.pooler_output
        )
        last_hidden_state_after_final_layer = self.final_convolution_layer(
            output_from_convnext.last_hidden_state
        )

        return output_from_convnext.pooler_output
