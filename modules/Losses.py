import tensorflow as tf
from keras import backend as K
import math
from numpy import sqrt,arange
from scipy.special import erf
from NumericalTools import *

def ahuber_generator(dm=0.5,dp=1.0):

    """generator for an asymmetric Huber loss function. For standard Huber set dm=dp"""

    def aux_fun(y_true,y_pred):
        e = y_true[:,0] - y_pred[:,0]
        aux = tf.where( tf.greater_equal(e,dp),
                        dp*e-0.5*dp**2,
                        tf.where( tf.greater_equal(e,-dm),
                                  0.5*K.square(e),
                                  -dm*e-0.5*dm**2) 
                        )
        return K.mean(aux)

    aux_fun.__name__=('ahuber_%3.2f_%3.2f_loss'%(dm,dp)).replace('.','p')
    return aux_fun

def quantile_loss_generator(q):

    """generator for a q-quantile loss"""

    def aux_fun(y_true, y_pred):
        e = y_true[:,0] - y_pred[:,0]        
        aux = tf.where(tf.greater_equal(e, 0.), q*e, (q-1)*e)
        return K.mean(aux)

    aux_fun.__name__ = ('quantile_%d_loss'%int(100*q))
    return aux_fun


def gd_offset_loss_generator(noOffset=True):

    """generator for a gd_loss function with fixed offset"""

    def aux_fun(y_true, y_pred):

        y_true = tf.unstack(y_true, axis=1, num=1)[0]
        p_pred = tf.unstack(y_pred, axis=1)
        

        if noOffset:
            mu     = rangeTransform( p_pred[0], -3,   3)
            sigma  = rangeTransform( p_pred[1], 1e-3, 5)
            aL     = rangeTransform( p_pred[2], 1e-3, 5)
            aR     = rangeTransform( p_pred[3], 1e-3, 5)
            aux    = gdLikelihood(y_true,mu,sigma,aL,aR)
            #aux = tf.Print(aux, [y_pred], "y_pred")
                    
        else:
            mu     = rangeTransform( p_pred[0], -math.pi, math.pi)
            sigma  = rangeTransform( p_pred[1],  1e-3, math.pi)
            aL     = rangeTransform( p_pred[2],  1e-3, math.pi)
            aR     = rangeTransform( p_pred[3],  1e-3, math.pi)
            offset = rangeTransform( p_pred[4],  1e-5, 1)
            aux    = gdOffsetLikelihood(y_true,mu,sigma,aL,aR,offset)            
            #aux    = tf.Print(aux, [aux,K.sum(aux),mu,sigma,aL,aR,offset], "params")
                       
        return K.sum(aux)

    aux_fun.__name__ = 'gd_%soffset_loss'%('no' if noOffset else '')
    return aux_fun

def bgsum_loss_generator(unbound):
    """a sum of two truncated normal distibutions with a core gaussian"""

    def aux_fun(y_true, y_pred):

        y_true = tf.unstack(y_true, axis=1, num=1)[0]
        p_pred = tf.unstack(y_pred, axis=1)

        if unbound:
            mu     = rangeTransform( p_pred[0], -3,   3)
            sigma  = rangeTransform( p_pred[1], 1e-3, 5)
            sigmaL = rangeTransform( p_pred[2], 1e-3, 10)
            fL     = rangeTransform( p_pred[3], 1e-6, 1)
            sigmaR = rangeTransform( p_pred[4], 1e-3, 10)
            fR     = rangeTransform( p_pred[5], 1e-6, 1)
        else:
            mu     = rangeTransform( p_pred[0], -0.5*math.pi, 0.5*math.pi)
            sigma  = rangeTransform( p_pred[1], 1e-3, math.pi)
            sigmaL = rangeTransform( p_pred[2], 1e-3, 1e4)
            fL     = rangeTransform( p_pred[3], 1e-6, 1.0)
            sigmaR = rangeTransform( p_pred[4], 1e-3, 1e4)
            fR     = rangeTransform( p_pred[5], 1e-6, 1.0)

        aux    = bgsumLikelihood(y_true,mu,sigma,sigmaL,fL,sigmaR,fR,unbound)
        return K.sum(aux)

    aux_fun.__name__ = 'bgsum_{0}_loss'.format('unbound' if unbound else 'mpi2pi')
    return aux_fun


# list of all the losses
global_loss_list={'gd_nooffset_loss'      : gd_offset_loss_generator(noOffset=True),
                  'gd_offset_loss'        : gd_offset_loss_generator(noOffset=False),
                  'bgsum_unbound_loss'    : bgsum_loss_generator(unbound=True),
                  'bgsum_mpi2pi_loss'    : bgsum_loss_generator(unbound=False),
                  'ahuber_1p00_1p00_loss' : ahuber_generator(dm=1.0,dp=1.0),
                  'ahuber_0p50_1p00_loss' : ahuber_generator(dm=0.5,dp=1.0),
                  'quantile_16_loss'      : quantile_loss_generator(0.16),
                  'quantile_50_loss'      : quantile_loss_generator(0.50),
                  'quantile_84_loss'      : quantile_loss_generator(0.84)}
