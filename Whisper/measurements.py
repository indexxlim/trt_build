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

"""
Utils specific to Whisper network.
"""
import timeit
from types import MethodType

# torch
import torch

# TRT-HuggingFace
from NNDF.general_utils import measure_python_inference_code
from NNDF.torch_utils import use_cuda, expand_inputs_for_beam_search
from NNDF.tensorrt_utils import TRTNativeRunner
from NNDF.logger import G_LOGGER

from Whisper.WhisperModelConfig import WhisperModelTRTConfig

# from HuggingFace transformers
from transformers.generation_logits_process import (
    NoRepeatNGramLogitsProcessor,
    MinLengthLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    LogitsProcessorList,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.generation_beam_search import (
    BeamSearchScorer,
)


@use_cuda
def decoder_inference(
    whisper_decoder,
    input_ids,
    encoder_last_hidden_state,
    timing_profile,
    use_cuda=True,
    use_cache=False,
    past_key_values=None,
):
    # This implementation is a bit ugly. Moving implementation of the model to check HFRunner would be cleaner.
    if isinstance(whisper_decoder, TRTNativeRunner):
        # Function is technically in WhisperTRTDecoder however due to circular import, TRTNativeRunner in this module scope
        # implies the existence of this function.
        whisper_decoder.set_return_device("cuda" if use_cuda else "cpu")

    def decoder_stmt():
        whisper_decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_last_hidden_state,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

    decoder_e2e_time = measure_python_inference_code(decoder_stmt, timing_profile)

    return (decoder_stmt(), decoder_e2e_time)


@use_cuda
def encoder_inference(whisper_encoder, input_features, timing_profile, use_cuda=True):
    encoder_stmt = lambda: whisper_encoder(input_features=input_features)
    encoder_e2e_time = measure_python_inference_code(encoder_stmt, timing_profile)

    return (encoder_stmt(), encoder_e2e_time)


@use_cuda
def full_inference(
    whisper_encoder,
    whisper_decoder,
    input_features,
    tokenizer,
    timing_profile,
    max_length,
    min_length=0,
    num_beams=1,
    batch_size=1,
    use_cuda=True,
    early_stopping=True,
    use_cache=False,
):
    G_LOGGER.info(f"Running full inference...")

    # encoder_last_hidden_state = whisper_encoder(input_features=input_features)
    def get_encoder(self):
        return whisper_encoder

    whisper_decoder.get_encoder = MethodType(get_encoder, whisper_decoder)

    def _e2e():
        start = timeit.default_timer()
        with torch.no_grad():
            decoder_output = whisper_decoder.generate(
                input_features,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                eos_token_id=whisper_decoder.config.eos_token_id,
                pad_token_id=whisper_decoder.config.pad_token_id,
                use_cache=use_cache,
            )
        stop = timeit.default_timer()
        print(stop - start)

        return decoder_output

    if isinstance(whisper_decoder, TRTNativeRunner):
        whisper_decoder.set_return_device("cuda" if use_cuda else "cpu")

    measurement_function = _e2e

    full_e2e_time = measure_python_inference_code(measurement_function, timing_profile)

    return (measurement_function(), full_e2e_time)


@use_cuda
def full_inference_greedy(
    Whisper_encoder,
    Whisper_decoder,
    input_features,
    tokenizer,
    timing_profile,
    max_length,
    min_length=0,
    batch_size=1,
    use_cuda=True,
    early_stopping=False,
    use_cache=False,
):
    G_LOGGER.info("Running full inference with greedy decoding...")

    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)])
    no_repeat_ngram_size = WhisperModelTRTConfig.NO_REPEAT_NGRAM_SIZE
    logits_processor = LogitsProcessorList(
        [
            NoRepeatNGramLogitsProcessor(no_repeat_ngram_size),
            MinLengthLogitsProcessor(
                min_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            ),
            ForcedBOSTokenLogitsProcessor(
                WhisperModelTRTConfig.DECODER_START_TOKEN_ID
            ),
            ForcedEOSTokenLogitsProcessor(
                max_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            ),
        ]
    )  # by checking HuggingFace's generate() implementation carefully, the default logits processor for BART has no_repeat_ngram_size = 3 and forced_eos_token_id = 2. In this way we can get identical results with raw HuggingFace

    decoder_input_ids = torch.full(
        (batch_size, 1),
        tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
        dtype=torch.int32,
    )

    if use_cuda:
        decoder_input_ids = decoder_input_ids.to("cuda")
    else:
        decoder_input_ids = decoder_input_ids.to("cpu")

    def _e2e():
        with torch.no_grad():
            encoder_last_hidden_state = Whisper_encoder(input_features=input_features)
            decoder_output_greedy = Whisper_decoder.greedy_search(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache,
            )
        return decoder_output_greedy
    
    # With e2e we can opt to bind inputs only once for hidden states for optimization
    def _e2e_trt():
        with torch.no_grad():
            encoder_last_hidden_state = Whisper_encoder(input_features=input_features)
            Whisper_decoder.set_encoder_hidden_states_for_inference_cycle(
                encoder_last_hidden_state
            )
            decoder_output_greedy = Whisper_decoder.greedy_search(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache,
            )
        return decoder_output_greedy

    measurement_function = _e2e
    if isinstance(Whisper_decoder, TRTNativeRunner):
        Whisper_decoder.set_return_device("cuda" if use_cuda else "cpu")
        measurement_function = _e2e_trt

    full_e2e_time = measure_python_inference_code(measurement_function, timing_profile)

    return (measurement_function(), full_e2e_time)


@use_cuda
def full_inference_beam(
    Whisper_encoder,
    Whisper_decoder,
    input_features,
    tokenizer,
    timing_profile,
    num_beams,
    max_length,
    min_length=0,
    batch_size=1,
    use_cuda=True,
    early_stopping=False,  # Now used to control beam search early_stopping to have the same meaning as HuggingFace
    use_cache=False,
):
    G_LOGGER.info(
        f"Running full inference with beam search (num_beams = {num_beams}) decoding..."
    )

    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)])
    no_repeat_ngram_size = WhisperModelTRTConfig.NO_REPEAT_NGRAM_SIZE
    logits_processor = LogitsProcessorList(
        [
            NoRepeatNGramLogitsProcessor(no_repeat_ngram_size),
            MinLengthLogitsProcessor(
                min_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            ),
            ForcedBOSTokenLogitsProcessor(
                WhisperModelTRTConfig.DECODER_START_TOKEN_ID
            ),
            ForcedEOSTokenLogitsProcessor(
                max_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            ),
        ]
    )  # by checking HuggingFace's generate() implementation carefully, the default logits processor for BART has no_repeat_ngram_size = 3 and forced_eos_token_id = 2. In this way we can get identical results with raw HuggingFace

    decoder_input_ids = torch.full(
        (batch_size, 1),
        tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
        dtype=torch.int32,
    )
    decoder_input_ids = expand_inputs_for_beam_search(
        decoder_input_ids, expand_size=num_beams
    )

    if use_cuda:
        decoder_input_ids = decoder_input_ids.to("cuda")
    else:
        decoder_input_ids = decoder_input_ids.to("cpu")

    def _e2e():
        with torch.no_grad():
            # beam scorer must be reset before each beam search run, otherwise beam search will be skipped due to scorer cache
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device="cuda" if use_cuda else "cpu",
                do_early_stopping=early_stopping,
            )

            encoder_last_hidden_state = Whisper_encoder(
                input_idsinput_features=input_features
            )

            encoder_last_hidden_state = expand_inputs_for_beam_search(
                encoder_last_hidden_state, expand_size=num_beams
            )

            decoder_output_beam = Whisper_decoder.beam_search(
                input_ids=decoder_input_ids,
                beam_scorer=beam_scorer,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache,
            )
        return decoder_output_beam

    # With e2e we can opt to bind inputs only once for hidden states for optimization
    def _e2e_trt():
        with torch.no_grad():
            # beam scorer must be reset before each beam search run, otherwise beam search will be skipped due to scorer cache
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device="cuda" if use_cuda else "cpu",
                do_early_stopping=early_stopping,
            )

            encoder_last_hidden_state = Whisper_encoder(input_features=input_features)

            encoder_last_hidden_state = expand_inputs_for_beam_search(
                encoder_last_hidden_state, expand_size=num_beams
            )

            Whisper_decoder.set_encoder_hidden_states_for_inference_cycle(
                encoder_last_hidden_state
            )
            decoder_output_beam = Whisper_decoder.beam_search(
                input_ids=decoder_input_ids,
                beam_scorer=beam_scorer,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache,
            )
        return decoder_output_beam

    measurement_function = _e2e
    if isinstance(Whisper_decoder, TRTNativeRunner):
        Whisper_decoder.set_return_device("cuda" if use_cuda else "cpu")
        measurement_function = _e2e_trt

    full_e2e_time = measure_python_inference_code(measurement_function, timing_profile)

    return (measurement_function(), full_e2e_time)


@use_cuda
def calculate_perplexity(
    whisper_encoder,
    whisper_decoder,
    tokenizer,
    input_features,
    decoder_input_ids,
    max_seq_len=None,
    use_cuda=True,
):
    encoder_last_hidden_state = whisper_encoder(input_features=input_features)
    if isinstance(whisper_decoder, TRTNativeRunner):
        whisper_decoder.set_return_device("cuda" if use_cuda else "cpu")

    # Set the first token to be pad token
    decoder_input_ids_padded = torch.full(
        decoder_input_ids.size()[:-1] + (decoder_input_ids.size()[-1] + 1,),
        tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
        dtype=decoder_input_ids.dtype,
    )
    decoder_input_ids_padded[..., 1:] = decoder_input_ids

    if use_cuda:
        encoder_last_hidden_state = encoder_last_hidden_state.to("cuda")
        decoder_input_ids_padded = decoder_input_ids_padded.to("cuda")

    with torch.no_grad():
        if max_seq_len is not None:
            decoder_input_ids_padded = decoder_input_ids_padded[:, :max_seq_len]
        logits = whisper_decoder(
            decoder_input_ids_padded, encoder_last_hidden_state, return_dict=True
        ).logits
        # Truncate the last prediction
        logits = logits[:, :-1, :]
        loss = torch.nn.CrossEntropyLoss()(logits.permute((0, 2, 1)), decoder_input_ids)
        return torch.exp(loss).item()
