import tensorflow as tf
import numpy as np

class GammaColorAugmentation(tf.keras.layers.Layer):
    def __init__(self, stddev, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.depth = len(stddev)
        self.seed = seed
        self.stddev = stddev
    def get_config(self):
        return {"stddev": self.stddev, "seed": self.seed}
    def call(self, input_tensor):
        batch_size = tf.shape(input_tensor)[0]
        # Random values are sampled at each call
        gammas = tf.random.normal(shape=(batch_size, self.depth), mean=0., stddev=1., seed=self.seed)*self.stddev+tf.constant([1.]*self.depth)
        return tf.pow(input_tensor, 1/tf.cast(gammas, input_tensor.dtype)[:,tf.newaxis, tf.newaxis,:])



class PeakLocalMax(tf.keras.layers.Layer):
    def __init__(self, min_distance=20, threshold_abs=0.5, *args, **kwargs):
        """ Find peaks in a batch of images as boolean mask. Peaks are the local
            maxima in a region of 2 * min_distance + 1 (i.e. peaks are separated
            by at least min_distance).

            If there are multiple local maxima with identical pixel intensities
            inside the region defined with min_distance, the coordinates of all
            such pixels are returned.

            Arguments:
                - min_distance (int): Minimum number of pixels separating peaks
                in a region of 2 * min_distance + 1 (i.e. peaks are separated by
                at least min_distance). To find all the local maxima, use
                min_distance=1).
                - threshold_abs (float): Minimum intensity of peaks.
        """
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        super().__init__(*args, **kwargs)
    def get_config(self):
        return {"min_distance": self.min_distance, "threshold_abs": self.threshold_abs}
    def build(self, input_shape):
        assert len(input_shape) == 4, "Expecting a 4 dimensional tensor. Received {}".format(input_shape)

    def call(self, batch_heatmap):
        """ Performs the peak-local-max operation on batch_heatmap.

            Arguments:
                - batch_heatmap: a float32 tensor of shape [B,H,W,C] in [0,1]
                containing B images of width W and height H with C channels.
            Returns:
                Returns a boolean tensor of shape [B,H,W,C] with
                - True: on local maxima of each channel
                - False: elsewhere.
        """
        max_pooled = tf.keras.layers.MaxPool2D(pool_size=2*self.min_distance+1, strides=1, padding="SAME")(batch_heatmap)
        return tf.logical_and(tf.equal(batch_heatmap, max_pooled), tf.greater(batch_heatmap, self.threshold_abs))


class AvoidLocalEqualities(tf.keras.layers.Layer):
    def get_config(self):
        return {}
    def build(self, input_shape):
        self.random_tensor = tf.expand_dims(tf.random.normal(input_shape[1:], mean=0, stddev=0.001), 0)
    def call(self, input_tensor):
        return self.random_tensor+input_tensor

class SingleKeypointDetectionMetricsLayer(tf.keras.layers.Layer):
    """ Computes true and false positives and negatives for single keypoint detection task
    """
    def __init__(self, detection_threshold, min_distance, target_enlargment_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_local_max = PeakLocalMax(min_distance=min_distance, threshold_abs=detection_threshold)
        self.avoid_local_eq = AvoidLocalEqualities()
        self.enlarge_target = tf.keras.layers.MaxPool2D(target_enlargment_size, strides=1, padding="same")
    def get_config(self):
        peak_local_max_config = self.peak_local_max.get_config()
        return {
            "detection_threshold": peak_local_max_config["threshold_abs"],
            "min_distance": peak_local_max_config["min_distance"],
            "target_enlargment_size": self.enlarge_target.get_config()["pool_size"][0]
        }
    def call(self, batch_target, batch_output):
        """ Arguments:
                batch_target - a [B,H,W,C] tensor
                batch_output - a [B,H,W,C] tensor
            Returns:
                A dictionary of elementary metrics of shape [B,C]
        """
        batch_output = self.avoid_local_eq(batch_output)
        batch_output = self.peak_local_max(batch_output)
        batch_output = tf.cast(batch_output, tf.int32)
        batch_target = self.enlarge_target(batch_target)
        batch_target = tf.cast(batch_target, tf.int32)

        """                  1 FN        1 FP + 1 FN        1 TP           1 FP          1 TN
            output        ___________    ________|__    ______|____    ______|____    ___________
            target        _______█___    ____█______    ______█____    ___________    ___________
            -------------------------------------------------------------------------------------------
            TP_map        ___________    ___________    ______|____    ___________    ___________
            FP_map        ___________    ________|__    ___________    ______|____    ___________
            max(output)        0              1              1              1              0
            max(target)        1              1              1              0              0
            -------------------------------------------------------------------------------------------
            batch_TP           0              0              1              0              0
            batch_FP           0              1              0              1              0
            batch_TN     (1-0)*(1-1)=0  (1-1)*(1-1)=0  (1-1)*(1-1)=0  (1-1)*(1-0)*0  (1-0)*(1-0)=1
            batch_FN         1-0=1          1-0=1          1-1=0          0-0=0          0-0=0
        """
        TP_map = tf.multiply(batch_target, batch_output)
        FP_map = tf.multiply(1-batch_target, batch_output)
        batch_TP = tf.reduce_sum(TP_map, axis=[1,2])
        batch_FP = tf.reduce_sum(FP_map, axis=[1,2])

        max_output = tf.reduce_max(batch_output, axis=[1,2])
        max_target = tf.reduce_max(batch_target, axis=[1,2])
        batch_TN = (1-max_output) * (1-max_target)
        batch_FN = max_target - tf.reduce_max(TP_map, axis=[1,2])

        return {
            "batch_TP": batch_TP,
            "batch_FP": batch_FP,
            "batch_TN": batch_TN,
            "batch_FN": batch_FN,
        }


class SingleKeypointDetectionMetrics(tf.keras.metrics.Metric):
    def __init__(self, detection_threshold, min_distance, target_enlargment_size, name='keypoint-detection-accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.metrics_computation = SingleKeypointDetectionMetricsLayer(detection_threshold, min_distance, target_enlargment_size)

    def update_state(self, batch_target, batch_output, sample_weight=None):
        metrics = self.metrics_computation(batch_target=batch_target, batch_output=batch_output)
        # TODO: split the different channels if any
        self.true_positives.assign_add(tf.reduce_sum(metrics["batch_TP"]))
        self.false_positives.assign_add(tf.reduce_sum(metrics["batch_FP"]))
        self.true_negatives.assign_add(tf.reduce_sum(metrics["batch_TN"]))
        self.false_negatives.assign_add(tf.reduce_sum(metrics["batch_FN"]))

    def result(self):
        precision = self.true_positives/(self.true_positives+self.false_positives)
        recall = self.true_positives/(self.true_positives+self.false_negatives)
        return precision, recall


# https://github.com/margokhokhlova/NT_Xent_loss_tensorflow/blob/master/contrastive_loss.py
# https://amitness.com/2020/03/illustrated-simclr/
class NT_Xent(tf.keras.layers.Layer):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """
    def __init__(self, tau=1, compute_accuracy=False, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.similarity = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        self.criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.compute_accuracy = compute_accuracy
    def get_config(self):
        return {"tau": self.tau, "compute_accuracy": self.compute_accuracy}
    def call(self, zizj):
        """ zizj is [B,N] tensor with order z_i1 z_j1 z_i2 z_j2 z_i3 z_j3 ...
            batch_size is twice the original batch_size
        """
        batch_size = tf.shape(zizj)[0]
        mask = tf.repeat(tf.repeat(~tf.eye(batch_size/2, dtype=tf.bool), 2, axis=0), 2, axis=1)

        sim = -1*self.similarity(tf.expand_dims(zizj, 1), tf.expand_dims(zizj, 0))/self.tau
        sim_i_j = -1*self.similarity(zizj[0::2], zizj[1::2])/self.tau

        pos = tf.reshape(tf.repeat(sim_i_j, repeats=2), (batch_size, -1))
        neg = tf.reshape(sim[mask], (batch_size, -1))

        logits = tf.concat((pos, neg), axis=-1)
        labels = tf.one_hot(tf.zeros((batch_size,), dtype=tf.int32), depth=batch_size-1)

        loss = self.criterion(labels, logits)
        accuracy = tf.equal(tf.math.argmax(logits, axis=-1), tf.math.argmax(labels, axis=-1))

        return (loss, accuracy) if self.compute_accuracy else loss


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


class Dilation2D(tf.keras.layers.Layer):
    def __init__(self, kernel: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.kernel = tf.constant(kernel[:,:,np.newaxis])
    def get_config(self):
        return {"kernel": self.kernel[:,:,0].numpy}
    def build(self, input_shape):
        assert len(input_shape) == 4, "Invalid shape for input. Expected a 4 dimentional tensor. Recieved {}".format(input_shape)
    def call(self, input_tensor):
        kernel = tf.cast(self.kernel, input_tensor.dtype)
        output = tf.nn.dilation2d(input_tensor, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format="NHWC")
        return output - tf.ones_like(output)



# class FocalLoss(ChunkProcessor):
#     def __init__(self, alpha=0.25, gamma=2):
#         """Compute focal loss for predictions.
#             Multi-labels Focal loss formula:
#                 FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
#                         ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
#         Args:
#             alpha: A scalar tensor for focal loss alpha hyper-parameter
#             gamma: A scalar tensor for focal loss gamma hyper-parameter
#         Returns:
#             loss: A (scalar) tensor representing the value of the loss function
#         """
#         self.alpha = alpha
#         self.gamma = gamma

#     def __call__(self, chunk):
#         """
#         prediction_tensor: A float tensor of shape [batch_size, num_anchors, num_classes] representing the predicted logits for each class
#         target_tensor: A float tensor of shape [batch_size, num_anchors, num_classes] representing one-hot encoded classification targets
#         """
#         prediction_tensor = chunk["batch_logits"]
#         target_tensor = chunk["one-hot"]#tf.one_hot(chunk["batch_target"], chunk["classes"])

#         sigmoid_p = tf.nn.sigmoid(prediction_tensor)
#         zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

#         # For poitive prediction, only need consider front part loss, back part is 0;
#         # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
#         pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

#         # For negative prediction, only need consider back part loss, front part is 0;
#         # target_tensor > zeros <=> z=1, so negative coefficient = 0.
#         neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
#         per_entry_cross_ent = - self.alpha * (pos_p_sub ** self.gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
#                             - (1 - self.alpha) * (neg_p_sub ** self.gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
#         return tf.reduce_sum(per_entry_cross_ent)



# def gaussian_kernel(size: int, mean: float, std: float):
#     vals = tfp.distributions.Normal(mean, std).prob(tf.range(start=-size, limit=size+1, dtype=tf.float32))
#     kernel = tf.einsum('i,j->ij', vals, vals)
#     return kernel/tf.reduce_sum(kernel)



