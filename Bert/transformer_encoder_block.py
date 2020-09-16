import tensorflow as tf
from tensorflow.keras import utils,layers,initializers,regularizers,constraints

@utils.register_keras_serializable(package="Text")
class TransformerEncoderBlock(layers.Layer):

    def __init__(self,
                 num_attention_heads,
               inner_dim,
               inner_activation,
               output_range=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_bias=True,
               norm_first=False,
               norm_epsilon=1e-12,
               output_dropout=0.0,
               attention_dropout=0.0,
               inner_dropout=0.0,
               attention_initializer=None,
               **kwargs):
        super().__init__(**kwargs)

        self._num_heads = num_attention_heads
        self._inner_dim = inner_dim
        self._inner_activation = inner_activation
        self._attention_dropout = attention_dropout
        self._output_dropout = output_dropout
        self._output_range = output_range
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._inner_dropout = inner_dropout
        if attention_initializer:
          self._attention_initializer = initializers.get(
              attention_initializer)
        else:
          self._attention_initializer = self._kernel_initializer

    def build(self,input_shape):
