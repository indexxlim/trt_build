#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse

from collections import namedtuple, OrderedDict
from itertools import product
from typing import Dict

# TRT-HuggingFace
from NNDF.networks import Precision, NetworkMetadata, NNConfig, speachDims as Dims
from NNDF.interface import MetadataArgparseInteropMixin

# Limitation of namedtuples. You must declare namedtuples in module scope and not in classes.
# Otherwise pickle doesn't work.
# See: https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
_WhisperMetadata = namedtuple("WhisperMetadata", ["kv_cache"])


class WhisperMetadata(_WhisperMetadata, MetadataArgparseInteropMixin):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add commandline interface parser."""
        network_group = parser.add_argument_group("Whisper network")
        network_group.add_argument(
            "--variant",
            help="Whisper variant to generate",
            choices=WhisperModelTRTConfig.TARGET_MODELS,
            required=True,
        )
        network_group.add_argument(
            "--enable-kv-cache",
            help="Whisper enable KV cache",
            action="store_true",
            default=False,
        )
        network_group.add_argument(
            "--num-beams",
            type=int,
            default=1,
            help="Enables beam search during decoding.",
        )

    @staticmethod
    def from_args(args: argparse.Namespace):
        return NetworkMetadata(
            variant=args.variant,
            precision=Precision(fp16=False),
            other=WhisperMetadata(kv_cache=args.enable_kv_cache),
        )

    @staticmethod
    def add_inference_args(parser: argparse.ArgumentParser) -> None:
        WhisperMetadata.add_args(parser)
        inference_group = parser.add_argument_group("inference group")
        inference_group.add_argument(
            "--fp16", action="store_true", help="Enables fp16 TensorRT tactics."
        )

    @staticmethod
    def from_inference_args(args: argparse.Namespace):
        base_metadata = WhisperMetadata.from_args(args)
        return base_metadata._replace(precision=Precision(fp16=False))

    @staticmethod
    def add_benchmarking_args(parser: argparse.ArgumentParser) -> None:
        benchmarking_group = parser.add_argument_group("benchmarking group")
        benchmarking_group.add_argument(
            "--input-seq-len",
            type=int,
            help="Specify fixed input sequence length for perf benchmarking. Required for benchmark except when both input_profile_max and output_profile_max are provided for trt",
        )
        benchmarking_group.add_argument(
            "--output-seq-len",
            type=int,
            help="Specify fixed output sequence length for perf benchmarking. Required for benchmark except when both input_profile_max and output_profile_max are provided for trt",
        )


WhisperBenchmarkingArgs = namedtuple(
    "WhisperBenchmarkingArgs", ["input_seq_len", "output_seq_len"]
)

# trt has more benchmarking arguments
WhisperTRTBenchmarkingArgs = namedtuple(
    "WhisperTRTBenchmarkingArgs",
    [
        "input_seq_len",
        "output_seq_len",
        "input_profile_max_len",
        "output_profile_max_len",
    ],
)


class WhisperModelTRTConfig(NNConfig):
    # choices: openai/whisper-tiny | openai/whisper-base | openai/whisper-small | openai/whisper-medium | openai/whisper-large-v2
    TARGET_MODELS = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v2",
    ]

    # TensorRT maximum workspace size for each model variant. Set by TensorRT memory_pool_limits API
    MAX_ENCODER_WORKSPACE_MB = {
        TARGET_MODELS[0]: 512,
        TARGET_MODELS[1]: 512,
        TARGET_MODELS[2]: 1024,
        TARGET_MODELS[3]: 2048,
        TARGET_MODELS[4]: 3072,
    }

    MAX_DECODER_WORKSPACE_MB = {
        TARGET_MODELS[0]: 1024,
        TARGET_MODELS[1]: 1024,
        TARGET_MODELS[2]: 2048,
        TARGET_MODELS[3]: 3072,
        TARGET_MODELS[4]: 4096,
    }

    MAX_SEQUENCE_LENGTH = {
        TARGET_MODELS[0]: 384,
        TARGET_MODELS[1]: 448,
        TARGET_MODELS[2]: 448,
        TARGET_MODELS[3]: 448,
        TARGET_MODELS[4]: 448,
    }

    ENCODER_HIDDEN_SIZE = {
        TARGET_MODELS[0]: 384,
        TARGET_MODELS[1]: 512,
        TARGET_MODELS[2]: 768,
        TARGET_MODELS[3]: 1024,
        TARGET_MODELS[3]: 1280,
    }

    # To achieve identical results with original HuggingFace implementation, the min_length in model config should be consistent with each model variant
    # see task-specific params in config.json of each variant model
    MIN_OUTPUT_LENGTH = {
        TARGET_MODELS[0]: 0,
        TARGET_MODELS[1]: 0,
        TARGET_MODELS[2]: 0,
        TARGET_MODELS[3]: 0,
        TARGET_MODELS[4]: 0,
    }

    # TODO: this might better be an inference time input like the `max_length` arg in generate() and greedy_search(). The change needed is in NNDF/interface.py:__call__ so it's a fundamental change affecting GPT2 and Whisper code. Here I just put this option in Whisper model config for now. But it's also reasonable to treat this as a model config, because the TRT engine building may need this to have fixed dimension (e.g., to enable KV-cache)
    # see task-specific params in config.json of each variant model
    MAX_OUTPUT_LENGTH = {
        TARGET_MODELS[0]: 448,
        TARGET_MODELS[1]: 448,
        TARGET_MODELS[2]: 448,
        TARGET_MODELS[3]: 448,
        TARGET_MODELS[4]: 448,
    }

    # This parameter should be using HuggingFace config, but this file is locked by test and cannot import transformers, so hardcoded here
    NUM_DECODER_LAYERS = {
        TARGET_MODELS[0]: 4,
        TARGET_MODELS[1]: 6,
        TARGET_MODELS[2]: 12,
        TARGET_MODELS[3]: 24,
        TARGET_MODELS[4]: 32,
    }

    NUMBER_OF_HEADS = {
        TARGET_MODELS[0]: 6,
        TARGET_MODELS[1]: 8,
        TARGET_MODELS[2]: 12,
        TARGET_MODELS[3]: 16,
        TARGET_MODELS[4]: 20,
    }

    NO_REPEAT_NGRAM_SIZE = 3
    DECODER_START_TOKEN_ID = 50258
    EOS_TOKEN_ID = 50257

    VOCAB_SIZE = {
        TARGET_MODELS[0]: 51865,
        TARGET_MODELS[1]: 51865,
        TARGET_MODELS[2]: 51865,
        TARGET_MODELS[3]: 51865,
    }

    SUPPRESS_TOKENS = [
        1,
        2,
        7,
        8,
        9,
        10,
        14,
        25,
        26,
        27,
        28,
        29,
        31,
        58,
        59,
        60,
        61,
        62,
        63,
        90,
        91,
        92,
        93,
        359,
        503,
        522,
        542,
        873,
        893,
        902,
        918,
        922,
        931,
        1350,
        1853,
        1982,
        2460,
        2627,
        3246,
        3253,
        3268,
        3536,
        3846,
        3961,
        4183,
        4667,
        6585,
        6647,
        7273,
        9061,
        9383,
        10428,
        10929,
        11938,
        12033,
        12331,
        12562,
        13793,
        14157,
        14635,
        15265,
        15618,
        16553,
        16604,
        18362,
        18956,
        20075,
        21675,
        22520,
        26130,
        26161,
        26435,
        28279,
        29464,
        31650,
        32302,
        32470,
        36865,
        42863,
        47425,
        49870,
        50254,
        50258,
        50358,
        50359,
        50360,
        50361,
        50362,
    ]

    BEGIN_SUPPRESS_TOKENS = [220, 50257]

    FORCED_DECODER_IDS = [[1, 50264], [2, 50359], [3, 50363]]

    NETWORK_FULL_NAME = "full"
    NETWORK_DECODER_SEGMENT_NAME = "decoder"
    NETWORK_ENCODER_SEGMENT_NAME = "encoder"
    NETWORK_SEGMENTS = [NETWORK_DECODER_SEGMENT_NAME, NETWORK_ENCODER_SEGMENT_NAME]

    def __init__(self):
        precision_fp16 = [False, True]
        kv_caches = [False, True]

        variants = []
        for variant, fp16, kv_cache in product(
            WhisperModelTRTConfig.TARGET_MODELS, precision_fp16, kv_caches
        ):
            variants.append(
                NetworkMetadata(
                    variant=variant,
                    precision=Precision(fp16=fp16),
                    other=WhisperMetadata(kv_cache=kv_cache),
                )
            )

        super().__init__("Whisper", variants=variants)

    def get_python_requirements(self):
        base_requirements = super().get_python_requirements()
        base_requirements.append("transformers==4.23.0")
        return base_requirements

    def get_network_segments(self):
        """
        Returns exportable segments for the given network.
        Used in the case where a single network needs to
        be exported into multiple parts.
        """
        return WhisperModelTRTConfig.NETWORK_SEGMENTS

    def get_metadata_string(self, metadata: NetworkMetadata) -> str:
        # Remove redundant t5 name
        metadata = metadata._replace(variant=metadata.variant.lstrip("openai/whisper-"))
        return super().get_metadata_string(metadata)

    @staticmethod
    def get_input_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of input dimensions.
        Keys will be equal to get_model_segments()

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_inputs_dict = OrderedDict(
            {
                "input_ids": (Dims.BATCH, Dims.SEQUENCE),
                "encoder_hidden_states": (
                    Dims.BATCH,
                    Dims.create_new_sequence_dim("encoder_hidden_length"),
                    WhisperModelTRTConfig.ENCODER_HIDDEN_SIZE[
                        metadata.variant
                    ],  # dim not containing string 'Dims.BATCH' or 'Dims.SEQUENCE' will be non-dynamic axis
                ),
            }
        )
        if metadata.other.kv_cache:
            # for KV cache version, we need add per-layer KV cache inputs. `past_key_values` at each layer is (self-attention K, self-attention V, cross-attention K, cross-attention V)
            for i in range(WhisperModelTRTConfig.NUM_DECODER_LAYERS[metadata.variant]):
                # decoder self-attention KV cache (dim[0] & dim[2] are dynamic, and dim[2] varies at each decoding timestep)
                self_attention_past_kv_dims = (
                    Dims.BATCH,
                    "num_heads",
                    Dims.create_new_sequence_dim("past_decoder_length"),
                    "embedding_size_per_head",
                )
                decoder_inputs_dict[
                    f"past_key_values.{i}.decoder.key"
                ] = self_attention_past_kv_dims
                decoder_inputs_dict[
                    f"past_key_values.{i}.decoder.value"
                ] = self_attention_past_kv_dims

                # encoder-decoder cross-attention KV cache (dim[0] & dim[2] are dynamic, but dim[2] is constant at each decoding timestep)
                cross_attention_past_kv_dims = (
                    Dims.BATCH,
                    "num_heads",
                    Dims.create_new_sequence_dim("encoder_length"),
                    "embedding_size_per_head",
                )
                decoder_inputs_dict[
                    f"past_key_values.{i}.encoder.key"
                ] = cross_attention_past_kv_dims
                decoder_inputs_dict[
                    f"past_key_values.{i}.encoder.value"
                ] = cross_attention_past_kv_dims

        decoder_inputs = Dims(decoder_inputs_dict)

        encoder_inputs = Dims(
            OrderedDict({"input_features": (Dims.BATCH, Dims.FEATURE, Dims.SEQUENCE)})
        )

        return {
            WhisperModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_inputs,
            WhisperModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME: encoder_inputs,
        }

    @staticmethod
    def get_output_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of output dimensions.
        Keys will be equal to get_model_segments()

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_outputs_dict = OrderedDict(
            {"hidden_states": (Dims.BATCH, Dims.SEQUENCE)}
        )

        if metadata.other.kv_cache:
            # for KV cache version, we need add per-layer KV cache inputs. `past_key_values` at each layer is (self-attention K, self-attention V, cross-attention K, cross-attention V)

            # for all BART variants, # encoder layers = # decoder layers, so just divide total # layers by 2
            for i in range(WhisperModelTRTConfig.NUM_DECODER_LAYERS[metadata.variant]):
                # decoder self-attention KV cache (dim[0] & dim[2] are dynamic, and dim[2] varies at each decoding timestep)
                self_attention_present_kv_dims = (
                    Dims.BATCH,
                    "num_heads",
                    Dims.create_new_sequence_dim("decoder_length"),
                    "embedding_size_per_head",
                )
                decoder_outputs_dict[
                    f"present_key_values.{i}.decoder.key"
                ] = self_attention_present_kv_dims
                decoder_outputs_dict[
                    f"present_key_values.{i}.decoder.value"
                ] = self_attention_present_kv_dims

                # encoder-decoder cross-attention KV cache (dim[0] & dim[2] are dynamic, but dim[2] is constant at each decoding timestep)
                cross_attention_present_kv_dims = (
                    Dims.BATCH,
                    "num_heads",
                    Dims.create_new_sequence_dim("encoder_length"),
                    "embedding_size_per_head",
                )
                decoder_outputs_dict[
                    f"present_key_values.{i}.encoder.key"
                ] = cross_attention_present_kv_dims
                decoder_outputs_dict[
                    f"present_key_values.{i}.encoder.value"
                ] = cross_attention_present_kv_dims

        decoder_outputs = Dims(decoder_outputs_dict)

        encoder_outputs = Dims(
            OrderedDict(
                {
                    "hidden_states": (
                        Dims.BATCH,
                        Dims.SEQUENCE,
                        WhisperModelTRTConfig.ENCODER_HIDDEN_SIZE[metadata.variant],
                    )
                }
            )
        )

        return {
            WhisperModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_outputs,
            WhisperModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME: encoder_outputs,
        }
