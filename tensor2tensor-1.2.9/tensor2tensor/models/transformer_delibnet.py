# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from .transformer import Transformer
from tensor2tensor.utils import beam_search

from .transformer import transformer_decoder, transformer_prepare_decoder, transformer_ffn_layer, transformer_base_v1

import tensorflow as tf

from tensorflow.python.util import nest

@registry.register_model
class Transformer_Delib(Transformer):
  """Attention net.  See file docstring."""

  def transformer_prepare_delibdecoder(self, inputs, hparams):
    """Prepare one shard of the model for the encoder.
    Args:
    inputs: a Tensor.
    hparams: run hyperparameters
    Returns:
    """
    firstPdecoder_input = inputs
    firstPdecoder_padding = common_attention.embedding_to_padding(firstPdecoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(firstPdecoder_padding)
    firstP_delib_attention_bias = ignore_padding
    if hparams.pos == "timing":
      firstPdecoder_input = common_attention.add_timing_signal_1d(firstPdecoder_input)

    return (firstPdecoder_input, firstP_delib_attention_bias)

  def model_fn_body(self, features):
    hparams = self._hparams
    inputs = features.get("inputs")
    firstP = features.get("firstP")
    firstP = common_layers.flatten4d3d(firstP)
    targets = features["targets"]
    targets = common_layers.flatten4d3d(targets)

    encoder_output, encoder_decoder_attention_bias = (None, None)
    if inputs is not None:
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(inputs, target_space, hparams)

    # used to extract hidden states
    (decoder_input, decoder_self_attention_bias) = transformer_prepare_decoder(firstP, hparams)
    # the conventional `targets` used for the second-pass decoder, i.e., delib-decoder
    (delibdecoder_input, delibdecoder_self_attention_bias) = transformer_prepare_decoder(targets, hparams)
    # the `delibctx` used for the second-pass decoder
    firstP_input, firstP_self_attention_bias = self.transformer_prepare_delibdecoder(firstP, hparams)

    # add dropout to the two decoders
    decoder_input = tf.nn.dropout(decoder_input, 1.0 - hparams.layer_prepostprocess_dropout)
    delibdecoder_input = tf.nn.dropout(delibdecoder_input, 1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder(decoder_input,
                                         encoder_output,
                                         decoder_self_attention_bias,
                                         encoder_decoder_attention_bias,
                                         hparams,
                                         cache=None)

    firstP_input = tf.concat(values=[firstP_input, decoder_output], axis=-1)

    delibdecoder_output = transformer_delibdecoder(
        delibdecoder_input, encoder_output, firstP_input,
        delibdecoder_self_attention_bias, encoder_decoder_attention_bias, firstP_self_attention_bias,
        hparams, cache=None, name="delib_decoder")
    return delibdecoder_output

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
        raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams

    inputs = features["inputs"]
    firstP = features["firstP"]
    batch_size = tf.shape(inputs)[0]
    target_modality = self._problem_hparams.target_modality
    if t2t_model.is_class_modality(target_modality):
        decode_length = 1
    else:
        decode_length = tf.shape(inputs)[1] + decode_length

    # TODO(llion): Clean up this reshaping logic.
    # @authors: what are U doing ?
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
    s = tf.shape(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])

    firstP = tf.expand_dims(firstP, axis=1)
    if len(firstP.shape) < 5:
        firstP = tf.expand_dims(firstP, axis=4)
    z = tf.shape(firstP)
    firstP = tf.reshape(firstP, [z[0] * z[1], z[2], z[3], z[4]])

    # _shard_features called to ensure that the variable names match
    inputs = self._shard_features({"inputs": inputs})["inputs"]

    # deal with the encoder
    input_modality = self._problem_hparams.input_modality["inputs"]
    with tf.variable_scope(input_modality.name):
        inputs = input_modality.bottom_sharded(inputs, dp)
    with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode, inputs, features["target_space_id"], hparams)
    encoder_output = encoder_output[0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

    # deal with the first pass decoder
    def preprocess_firstP(firstP):
        firstP = self._shard_features({"firstP": firstP})["firstP"]
        firstP_modality = self._problem_hparams.input_modality["firstP"]
        with tf.variable_scope(firstP_modality.name):
            firstP = firstP_modality.targets_bottom_sharded(firstP, dp)[0]
        firstP = common_layers.flatten4d3d(firstP)
        if hparams.pos == "timing":
            firstP = common_attention.add_timing_signal_1d(firstP)
        return firstP

    firstP = preprocess_firstP(firstP)
    firstPdecoder_padding = common_attention.embedding_to_padding(firstP)
    firstP_delib_attention_bias = common_attention.attention_bias_ignore_padding(firstPdecoder_padding)
    firstP_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(tf.shape(firstP)[1]))
    if hparams.proximity_bias:
        firstP_self_attention_bias += common_attention.attention_bias_proximal(tf.shape(firstP)[1])

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
        """Performs preprocessing steps on the targets to prepare for the decoder.

        This includes:
          - Embedding the ids.
          - Flattening to 3D tensor.
          - Optionally adding timing signals.

        Args:
          targets: inputs ids to the decoder. [batch_size, 1]
          i: scalar, Step number of the decoding loop.

        Returns:
          Processed targets [batch_size, 1, hidden_dim]
        """
        # _shard_features called to ensure that the variable names match
        targets = self._shard_features({"targets": targets})["targets"]
        with tf.variable_scope(target_modality.name,reuse=True):
            targets = target_modality.targets_bottom_sharded(targets, dp)[0]
        targets = common_layers.flatten4d3d(targets)

        # TODO(llion): Explain! Is this even needed?
        targets = tf.cond(
            tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

        if hparams.pos == "timing":
            targets += timing_signal[:, i:i + 1]
        return targets

    # this is actually for the delib-decoder, i.e., the 2nd-pass decoder
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
        decoder_self_attention_bias += common_attention.attention_bias_proximal(decode_length)

    key_channels = hparams.attention_key_channels or hparams.hidden_size
    value_channels = hparams.attention_value_channels or hparams.hidden_size
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

    cache = {
        "layer_%d" % layer: {
            "k": tf.zeros([batch_size, 0, key_channels]),
            "v": tf.zeros([batch_size, 0, value_channels]),
        }
        for layer in range(num_layers)
        }

    # Set 2nd dim to None since it's not invariant in the tf.while_loop
    # Note: Tensor.set_shape() does not work here since it merges shape info.
    # TODO(llion); Find a more robust solution.
    # pylint: disable=protected-access
    for layer in cache:
        cache[layer]["k"]._shape = tf.TensorShape([None, None, key_channels])
        cache[layer]["v"]._shape = tf.TensorShape([None, None, value_channels])
    # pylint: enable=protected-access
    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    with tf.variable_scope("body"):
        firstP_hidden = dp(transformer_decoder, firstP, encoder_output, firstP_self_attention_bias,
                            encoder_decoder_attention_bias, hparams)
    firstP_input = tf.concat(values=[firstP, firstP_hidden[0]], axis=-1)
    cache["firstP_input"] = firstP_input
    cache["firstP_self_attention_bias"] = firstP_delib_attention_bias

    def symbols_to_logits_fn(ids, i, cache):
        """Go from ids to logits for next symbol."""
        ids = ids[:, -1:]
        targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
        targets = preprocess_targets(targets, i)

        bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

        with tf.variable_scope("body"):
            body_outputs = dp(transformer_delibdecoder,
                targets, cache["encoder_output"], cache["firstP_input"],
                bias, cache["encoder_decoder_attention_bias"],
                cache["firstP_self_attention_bias"], hparams, cache)

        with tf.variable_scope(target_modality.name):
            logits = target_modality.top_sharded(body_outputs, None, dp)[0]

        return tf.squeeze(logits, axis=[1, 2, 3]), cache



    if beam_size > 1:  # Beam Search
        target_modality = (
            self._hparams.problems[self._problem_idx].target_modality)
        vocab_size = target_modality.top_dimensionality
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)
        decoded_ids, scores = beam_search.beam_search(
            symbols_to_logits_fn, initial_ids, beam_size, decode_length,
            vocab_size, alpha, states=cache, stop_early=(top_beams == 1))
        
        if top_beams == 1:
            decoded_ids = decoded_ids[:, 0, 1:]
        else:
            decoded_ids = decoded_ids[:, :top_beams, 1:]
    else:  # Greedy

        def inner_loop(i, next_id, decoded_ids, cache):
            logits, cache = symbols_to_logits_fn(next_id, i, cache)
            temperature = (0.0 if hparams.sampling_method == "argmax"
                           else hparams.sampling_temp)
            next_id = tf.expand_dims(
                common_layers.sample_with_temperature(logits, temperature), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i + 1, next_id, decoded_ids, cache

        decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
        scores = None
        next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
        _, _, decoded_ids, _ = tf.while_loop(
            # TODO(llion): Early stopping.
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), next_id, decoded_ids, cache],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                nest.map_structure(lambda t: tf.TensorShape(t.shape), cache),
            ])

    return decoded_ids, scores


def transformer_delibdecoder(decoder_input,
                             encoder_output,
                             firstP_input,
                             decoder_self_attention_bias,
                             encoder_decoder_attention_bias,
                             firstP_self_attention_bias,
                             hparams,
                             cache=None,
                             name="delib_decoder"):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    firstP_input: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    firstP_self_attention_bias: bias Tensor for delibctx attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    name: a string

  Returns:
    y: a Tensors
  """
  def _one_attn_unit(x):
    if encoder_output is not None:
      with tf.variable_scope("delibctx_attention"):
        # TODO(llion): Add caching.
        y = common_attention.multihead_attention(
            common_layers.layer_preprocess(x, hparams),
            encoder_output, encoder_decoder_attention_bias,
            hparams.attention_key_channels or hparams.hidden_size,
            hparams.attention_value_channels or hparams.hidden_size,
            hparams.hidden_size, hparams.num_heads,
            hparams.attention_dropout,
            name="encdec_attention"
        )
        x = common_layers.layer_postprocess(x, y, hparams)
        return x
    return x

  def _two_attn_unit(x):
    with tf.variable_scope("delibctx_attention"):
      preprocess_x = common_layers.layer_preprocess(x, hparams)
      y = common_attention.multihead_attention(
              preprocess_x,
              encoder_output, encoder_decoder_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size, hparams.num_heads,
              hparams.attention_dropout,
              name="encdec_attention") + \
          common_attention.multihead_attention(
              preprocess_x,
              firstP_input, firstP_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size, hparams.num_heads,
              hparams.attention_dropout,
              name="decdec_attention"
          )
      x = common_layers.layer_postprocess(x, y, hparams)
      return x

  if hparams.delib_layers is "":
    delib_layers = range(hparams.num_hidden_layers)
  else:
    delib_layers = hparams.delib_layers.split(";")
    delib_layers = [int(x.strip()) for x in delib_layers]

  x = decoder_input
  with tf.variable_scope(name):
    for layer in xrange(hparams.num_decoder_layers or
                        hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              cache=layer_cache)
          x = common_layers.layer_postprocess(x, y, hparams)
        if layer in delib_layers:
          x = _two_attn_unit(x)
        else:
          x = _one_attn_unit(x)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    delibdecoder_output = common_layers.layer_preprocess(x, hparams)
    return tf.expand_dims(delibdecoder_output, axis=2)


@registry.register_hparams
def transformer_delib_big():
    """HParams for transfomer big delibnet model on WMT."""
    hparams = transformer_base_v1()
    hparams.add_hparam("delib_layers", "")
    hparams.add_hparam("update_delib_only", True)
    hparams.hidden_size = 1024
    hparams.filter_size = 4096
    hparams.num_heads = 16
    hparams.learning_rate_warmup_steps = 1
    hparams.shared_embedding_and_softmax_weights = int(False)
    return hparams
