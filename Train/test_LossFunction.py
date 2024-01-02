from Losses import *
import math
import numpy as np
import tensorflow as tf

ntest=10


with tf.Session() as sess:

    print('='*50)
    print('gd_offset_loss')
    y_true=tf.stack( [[(2.0*x/float(ntest)-1)*math.pi for x in range(0,ntest+1)]],axis=1 )
    y_pred=tf.stack([[0.0,0.1,1.0,1.0,0.1]])
    print('Truth set to')
    print(sess.run(y_true))
    print('Prediction set to')
    print(sess.run(y_pred))

    print('Evaluating loss function')
    loss=gd_offset_loss(y_true,y_pred,sess)
    print('Final result')
    print(sess.run(loss))

    print('='*50)
    print('gd_loss')
    y_true=tf.stack( [[0.25*x for x in range(0,ntest+1)]],axis=1 )
    y_pred=tf.stack([[0.0,0.1,1.0,1.0]])
    print('Truth set to')
    print(sess.run(y_true))
    print('Prediction set to')
    print(sess.run(y_pred))

    print('Evaluating loss function')
    loss=gd_loss(y_true,y_pred,sess)
    print('Final result')
    print(sess.run(loss))



