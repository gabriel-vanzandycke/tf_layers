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
    def get_config(self):
        return {"tau": self.tau}
    def call(self, zizj):
        """ zizj is [B,N] tensor with order zi1 zj1 zi2 zj2 zi3 zj3 ... 
            batch_size is twice the original batch_size
        """
        batch_size = tf.shape(zizj)[0]
        mask = 1-tf.eye(batch_size, dtype=tf.float32)
        sim = self.similarity(tf.expand_dims(zizj, 1), tf.expand_dims(zizj, 0))
        neg = tf.reduce_sum(tf.exp(sim/self.tau)*mask, axis=-1)
        sim_i_j = tf.exp(self.similarity(zizj[0::2], zizj[1::2])/self.tau)
        pos = tf.repeat(sim_i_j, repeats=2)
        return -tf.reduce_mean(tf.math.log(pos/neg))


class AvgSPP(tf.keras.layers.Layer):
    def __init__(self, scale, name=None):
        super().__init__(name=name)
        self.scale = scale
    def get_config(self):
        return {"scale": self.scale}
    def build(self, input_shape):
        self.shape = input_shape
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        shape = tf.shape(inputs)[1:3]
        eye = tf.eye(self.scale**2, batch_shape=(batch_size,))                              # identity matrix   [B, s*s, s*s]
        mask = tf.reshape(eye, (-1, self.scale, self.scale, self.scale**2))                 # simple mask       [B, s, s, s*s]
        mask = tf.image.resize(mask, shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # full mask         [B, H, W, s*s]
        spp = tf.multiply(tf.expand_dims(inputs, 4), tf.expand_dims(mask, 3))               # splitted image    [B, H, W, C, s*s]
        spp = tf.reduce_mean(spp, axis=[1,2])*self.scale**2                                 # average           [B, 1, 1, C, s*s]
        spp = tf.reshape(spp, (-1, self.shape[3], self.scale, self.scale))                  # reshaping         [B, C, s, s]
        spp = tf.transpose(spp, [0,2,3,1])                                                  # transposing       [B, s, s, C]
        return tf.image.resize(spp, shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)   # recover size      [B, H, W, C]


# import torch
# zi = torch.from_numpy(x[0::2])
# zj = torch.from_numpy(x[1::2])
# print(zi.shape)
# print(zj.shape)
# class NT_Xent(torch.nn.Module):
#     def __init__(self, tau=1):
#         super(NT_Xent, self).__init__()
#         self.tau = tau

#     def forward(self, z_i, z_j):
#         """
#         We do not sample negative examples explicitly.
#         Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
#         """
#         out = torch.cat((z_i, z_j), dim=0)
#         n_samples = len(out)
#         cov = torch.mm(out, out.t().contiguous())
#         print(cov)
#         sim = torch.exp(cov/self.tau)
        
#         mask = ~torch.eye(n_samples).bool()
#         neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
#         print(neg)
#         pos = torch.exp(torch.sum(z_i*z_j, dim=-1)/ self.tau)
#         pos = torch.cat([pos, pos], dim=0)
#         print(pos)
#         loss = -torch.log(pos/neg).mean()
#         return loss
# print(NT_Xent()(zi,zj))