import tensorflow as tf
import tensorflow.keras as keras 
# Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
import json
import numpy as np


# CONSTANTS
KEY_DIM = 64
NUM_HEADS = 8
BATCH_SIZE = 32

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
)


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder Layer of a Transformer, consisting of MultiHeadAttention and
    Feed Forward Neural Network.
    Recall after the positional Encoding Part
    This will be 'Nx'ed
    (k, v, q) -> MhA - (skip with inp) -> AN -> FF -> (skip) -> AN -> EncoderLayer2 or EncoderBlock2
    """

    def __init__(self, d_feedforward, 
                 d_model=KEY_DIM, num_heads=NUM_HEADS, 
                 dropout_rate=0.1):
        """
        Parameters:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_feedforward (int): Dimension of the feed forward network.
            dropout_rate (float): Dropout dropout_rate.
        """
        super(EncoderLayer, self).__init__()
        self.key_dim = d_model
        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(key_dim=self.key_dim, 
                                      num_heads=self.num_heads)
        self.dropout1 = Dropout(dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)

        # Recall after this -> add-norm 2 of these
        # Also, FF -> simple layer
        self.ffn = keras.Sequential([
            Dense(units=d_feedforward, activation='relu'),
            Dense(units=self.key_dim)
        ])
        self.dropout2 = Dropout(dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)




    def call(self, x):
        '''
        Similar to forward() in torch
        (key, value, query) to mha
        '''
        all_head_attention_scores = self.mha(x, x, x)
        attention_scores = self.dropout1(all_head_attention_scores)
        skip_concat_out = x + attention_scores
        add_norm_out1 = self.layernorm1(skip_concat_out)

        ffn_out = self.ffn(add_norm_out1)
        ffn_out = self.dropout1(ffn_out)
        skip_concat_out2 = add_norm_out1 + ffn_out 
        add_norm_out2 = self.layernorm2(skip_concat_out2)
        return add_norm_out2




if __name__ == "__main__":

    x = tf.random.normal([BATCH_SIZE, 3, KEY_DIM])
    el = EncoderLayer(d_feedforward=128)
    out = el(x)

    print( out.shape ) # (32, 3, 64)

