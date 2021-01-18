import tensorflow as tf

class NT_Xent(tf.keras.layers.Layer):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """
    def __init__(self, tau=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.similarity = lambda x,y: tf.reduce_sum(tf.multiply(x, y), axis=-1)
        #self.similarity = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
    def build(self, input_shape):
        batch_size = input_shape[0]
        self.mask = 1-tf.eye(batch_size, dtype=tf.float32)
    def call(self, zizj):
        """ zizj is [B,N] tensor with order zi1 zj1 zi2 zj2 zi3 zj3 ... 
            batch_size is twice the original batch_size
        """
        sim = self.similarity(tf.expand_dims(zizj, 1), tf.expand_dims(zizj, 0))
        neg = tf.reduce_sum(tf.exp(sim/self.tau)*self.mask, axis=-1)
        sim_i_j = tf.exp(self.similarity(zizj[0::2], zizj[1::2])/self.tau)
        pos = tf.repeat(sim_i_j, repeats=2)
        return -tf.reduce_mean(tf.math.log(pos/neg))
