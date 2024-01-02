import tensorflow as tf
from keras import backend as K
import math
from numpy import sqrt,arange

def rangeTransform(x,a,b,inverse=False):

    """transform to a limited range as Minuit does (cf. https://root.cern.ch/download/minuit.pdf Sec. 1.2.1)"""

    return tf.asin(2*(x-a)/(b-a)-1.0) if inverse else a+0.5*(b-a)*(tf.sin(x)+1.0)

def phi_mpi_pi(phi):
    
    """ if phi<-pi : phi+2*pi if phi>pi : phi-2*pi """
    
    return tf.where( K.greater_equal(phi,math.pi),
                     phi-2*math.pi,
                     tf.where( K.less(phi,-math.pi),
                               phi+2*math.pi,
                               phi) 
                     )

def gdLikelihood(y_true,mu,sigma,aL,aR,gausConstraints={}):

    """ implements the gaussian-double exponential likelihood """

    #reduce the variable
    t = (y_true - mu)/sigma

    #normalization term    
    Norm = sqrt(math.pi/2)*sigma*(tf.erf(aR/sqrt(2)) + tf.erf(aL/sqrt(2))) 
    Norm += K.exp(-0.5*K.pow(aL,2))*sigma/aL 
    Norm += K.exp(-0.5*K.pow(aR,2))*sigma/aR

    #add gaussian constraints
    aux_gaus=0.
    if 'mu' in gausConstraints:
        Norm     +=  sqrt(0.5/math.pi)/gausConstraints['mu'][1]
        aux_gaus +=  0.5*K.pow((aL-gausConstraints['mu'][0])/gausConstraints['mu'][1],2)
    if 'sigma' in gausConstraints:
        Norm     +=  sqrt(0.5/math.pi)/gausConstraints['sigma'][1]
        aux_gaus +=  0.5*K.pow((aL-gausConstraints['sigma'][0])/gausConstraints['sigma'][1],2)
    if 'aL' in gausConstraints:
        Norm     +=  sqrt(0.5/math.pi)/gausConstraints['aL'][1]
        aux_gaus +=  0.5*K.pow((aL-gausConstraints['aL'][0])/gausConstraints['aL'][1],2)
    if 'aR' in gausConstraints:
        Norm     +=  sqrt(0.5/math.pi)/gausConstraints['aR'][1]
        aux_gaus +=  0.5*K.pow((aL-gausConstraints['aR'][0])/gausConstraints['aR'][1],2)

        
    #make sure the normalization is not 0
    Norm  = tf.clip_by_value(Norm,1e-5,9e12)
    
    #negative log-likelihood
    nll = tf.where(K.greater_equal(t, aR),
                   K.log(Norm) -0.5*K.pow(aR, 2) + aR*t + aux_gaus,
                   tf.where(K.greater_equal(t, -aL),
                            K.log(Norm) + 0.5*K.pow(t,2) + aux_gaus,
                            K.log(Norm) -0.5*K.pow(aL, 2) - aL*t + aux_gaus
                            )
                   )
    
    return nll

def gdOffsetLikelihood(y_true,e2,sigma_e2,a1_e2,a2_e2,n_e2,gausConstraints={}):

    """ implements the gaussian-double exponential+offset likelihood """    

    #reduce the variable
    #y_true=phi_mpi_pi(y_true)
    t=(y_true - e2)/sigma_e2

    #validity range of the pdf
    t1 = (-math.pi-e2)/sigma_e2
    t2 = (math.pi-e2)/sigma_e2

    #normalization: take care of cases where the exponents are larger than integration range
    n1 = tf.where(tf.less(t1,-a1_e2),
                  (K.exp(0.5*tf.pow(a1_e2,2))/a1_e2)*(K.exp(-tf.pow(a1_e2,2)) - K.exp(a1_e2*t1)),
                  tf.zeros_like(t1))
    n2 = tf.where(tf.greater(t2,a2_e2),
                  (K.exp(0.5*tf.pow(a2_e2,2))/a2_e2)*(K.exp(-tf.pow(a2_e2,2)) - K.exp(-a2_e2*t2)),
                  tf.zeros_like(t2))
    n0 = tf.where(tf.logical_and( tf.less(t1,-a1_e2), tf.greater(t2,a2_e2) ),
                  (tf.erf(a2_e2/sqrt(2))+tf.erf(a1_e2/sqrt(2))),
                  tf.where( tf.logical_and( tf.greater(t1,-a1_e2), tf.greater(t2,a2_e2) ),
                            (tf.erf(a2_e2/sqrt(2))-tf.erf(t1/sqrt(2))),
                            tf.where( tf.logical_and( tf.greater(t1,-a1_e2), tf.less(t2,a2_e2) ),
                                      (tf.erf(t2/sqrt(2))-tf.erf(t1/sqrt(2))),
                                      (tf.erf(t2/sqrt(2))+tf.erf(a1_e2/sqrt(2))))
                            )
                  )    
    n0  *= sqrt(math.pi/2)
    noff = n_e2*(t2-t1)
    N    = sigma_e2*(n0+n1+n2+noff)
    N    = tf.clip_by_value(N,1e-5,9e12)    

    #evaluate PDF and likelilhood
    f = tf.where(tf.greater_equal(t, a2_e2),
                 K.exp(0.5*tf.pow(a2_e2, 2)-a2_e2*t),
                 tf.where(tf.greater_equal(t,-a1_e2),
                          K.exp(-0.5*tf.pow(t,2)),
                          K.exp(0.5*tf.pow(a1_e2, 2)+a1_e2*t)
                          )
                 )
    nll = K.log(N)-K.log(f+n_e2)

    #nll=tf.Print(nll,[nll], 'nll')
    #nll=tf.Print(nll,[e2,sigma_e2,a1_e2,a2_e2,n_e2],'params')
    #arg_extremes=[tf.argmin(nll, axis=0),    tf.argmax(nll, axis=0)]
    #extremes=[tf.gather(nll,arg) for arg in arg_extremes]
    #nll=tf.Print(nll,[nll,extremes], 'nll, extremes')
    #N=tf.check_numerics(N,'CHECK NUMERICS: N')
    #f=tf.check_numerics(f,'CHECK NUMERICS: f')
    #nll=tf.check_numerics(nll,'CHECK NUMERICS: nll')
    
    return nll


def bgsumLikelihood(y_true,mu,sigma,sigmaL,fL,sigmaR,fR,unbound):

    """ implements the gaussian-double exponential likelihood """

    #normalization term    
    if unbound:
        Norm  = sigma*(1+(1+fL)/(1+fR))
        Norm += fL*sigmaL
        Norm += fR*sigmaR*((1+fL)/(1+fR))
    else:
        Norm  = sigma*tf.erf((math.pi+mu)/(sqrt(2)*sigma))
        Norm += sigma*tf.erf((math.pi-mu)/(sqrt(2)*sigma))*(1+fL)/(1+fR)
        Norm += fL*sigmaL*tf.erf((math.pi+mu)/(sqrt(2)*sigmaL))
        Norm += fR*sigmaR*tf.erf((math.pi-mu)/(sqrt(2)*sigmaR))*(1+fL)/(1+fR)
    Norm *= sqrt(math.pi/2)
    Norm  = tf.clip_by_value(Norm,1e-5,9e12)
    
    #negative log-likelihood
    func = tf.where(K.less_equal(y_true, mu),
                    (K.exp(-0.5*K.pow((y_true-mu)/sigma,2))+fL*K.exp(-0.5*K.pow((y_true-mu)/sigmaL,2))),
                    (K.exp(-0.5*K.pow((y_true-mu)/sigma,2))+fR*K.exp(-0.5*K.pow((y_true-mu)/sigmaR,2)))*((1+fL)/(1+fR)),
                    )
    func = tf.clip_by_value(func,1e-5,9e12)
    nll = K.log(Norm)-K.log(func)
    
    return nll
