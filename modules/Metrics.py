from keras import backend as K
import tensorflow as tf
import math

from NumericalTools import rangeTransform

def scale_metric(y_true, y_pred):
    """custom metric for scale"""
    y_true = tf.unstack(y_true, axis=1, num=1)[0]
    #mu = tf.unstack(y_pred, axis=1)[0]
    mu    = rangeTransform( tf.unstack(y_pred, axis=1)[0], -3, 3)
    return K.std(mu - y_true)


def dir_metric(y_true, y_pred):
    """custom metric for direction"""
    y_true = tf.unstack(y_true, axis=1, num=1)[0]
    mu = rangeTransform( tf.unstack(y_pred, axis=1)[0], -math.pi, math.pi)
    
    aux = tf.where(tf.greater_equal(y_true-mu, math.pi), y_true-mu - 2*math.pi,
                   tf.where(tf.greater_equal(-math.pi, y_true-mu), y_true-mu + 2*math.pi,
                            y_true-mu)
                        )
    return K.std(aux)

#list of all metrics
global_metrics_list={'scale_metric':scale_metric,
                     'dir_metric':dir_metric}
