from keras.models import Model
from keras.layers import Input, Dense, Dropout,BatchNormalization,Concatenate, concatenate, Add
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras import backend as K
from Losses import *

def oneParamRegDNN(Inputs,nclasses,nregclasses,dropoutRate=None,batchNorm=True,recursiveInputs=True,arch='30x15x5'):
    """mean regression with a DNN"""

    nodes=arch.split('x')
    arch=[(i+1, int(nodes[i]), 'lrelu') for i in xrange(0,len(nodes))]

    print '[oneParamRegDNN] configuring',arch,'parameter-specific nodes'
    print '%d (%d) regression classes (parameters)'%(nregclasses,nregclasses)
    print 'Inputs for nested networks will be','recursive' if recursiveInputs else 'independent'

    Outputs=[]
    for ic in xrange(0,nregclasses):

        x=Inputs[0]
        if len(Outputs)>0 and recursiveInputs:
            x=concatenate(Outputs+[x],axis=1,name='in_%d'%ic)

        if batchNorm:
            x=BatchNormalization(name='bninputs_c%d'%ic)(x)

        for ilayer,isize,iact in arch:                          
            x = Dense(isize,  
                      kernel_initializer='glorot_normal', 
                      bias_initializer='glorot_uniform', 
                      name='dense%d_c%d'%(ilayer,ic))(x)
            if batchNorm:
                x = BatchNormalization(name='bn%d_c%d'%(ilayer,ic))(x)
            if dropoutRate: 
                x = Dropout(dropoutRate)(x)
            if iact=='lrelu':
                x = LeakyReLU(0.2)(x)
            if iact=='prelu':
                x =  PReLU()(x)
        
        Outputs.append( Dense(1, use_bias=True, name='out_c%d'%ic)(x) )
            
    return Model(inputs=Inputs, outputs=Outputs)

def semiParamSingleRegDNN(Inputs,nclasses,nregclasses,dropoutRate=None,batchNorm=True,recursiveInputs=True,arch='32x16x6'):
    """model for the regression of the recoil and direction scale, with only one DNN"""

    #parse the architecture
    nodes=arch.split('x')
    arch=[(i+1, int(nodes[i]), 'lrelu') for i in xrange(0,len(nodes))]

    nparams=nregclasses
    if nparams%5==0 :
        nregclasses/=5
        pList=['mu','sigma','aL','aR','offset']
    else            :
        nregclasses/=6
        pList=['mu','sigma','sigmaL','fL','sigmaR','fR']

    print '[semiParamSingleRegDNN] configuring',arch,'parameter-specific nodes'
    print '%d (%d) regression classes (parameters)'%(nregclasses,nparams)
    print 'Inputs for nested networks will be','recursive' if recursiveInputs else 'independent'

    Outputs=[]
    for ic in xrange(0,nregclasses):

        x=Inputs[0]

        if len(Outputs)>0 and recursiveInputs:
            #x=concatenate(Outputs+[x],axis=1,name='in_%d'%ic)
            x=concatenate(Outputs+[x] )

        if batchNorm:
            x=BatchNormalization(name='bninputs_c%d'%ic)(x)

        for ilayer,isize,iact in arch:
            x = Dense(isize,
                      kernel_initializer='glorot_normal',
                      bias_initializer='glorot_uniform',
                      name='dense%d_c%d'%(ilayer,ic))(x)
            if batchNorm:
                x = BatchNormalization(name='bn%d_c%d'%(ilayer,ic))(x)
            if dropoutRate:
                x = Dropout(dropoutRate)(x)
            if iact=='lrelu':
                x = LeakyReLU(0.2)(x)
            if iact=='prelu':
                x =  PReLU()(x)

        # Outputs have the same size as pList
        Outputs.append( Dense(len(pList),  kernel_initializer='glorot_normal', bias_initializer='glorot_uniform', name='out_c%d'%ic)(x) )

    return Model(inputs=Inputs, outputs=Outputs)


def semiParamRegDNN(Inputs,nclasses,nregclasses,dropoutRate=None,batchNorm=True,recursiveInputs=True,arch=':32x16x4'):
    """model for the regression of the recoil and direction scale"""

    #inputs
    x=Inputs[0]

    #parse the architecture                                                                     
    nodesCommon,nodesInd=arch.split(':')

    nparams=nregclasses
    if nparams%5==0 : 
        nregclasses/=5
        pList=['mu','sigma','aL','aR','offset']
    else            : 
        nregclasses/=6
        pList=['mu','sigma','sigmaL','fL','sigmaR','fR']

    print '[semiParamRegDNN] configuring',nodesCommon,'common nodes and',nodesInd,'parameter-specific nodes'
    print '%d (%d) regression classes (parameters)'%(nregclasses,nparams)
    print 'Inputs for nested networks will be','recursive' if recursiveInputs else 'independent'

    #common nodes
    archCommon=[]
    if len(nodesCommon)>0 :
        nodesCommon=nodesCommon.split('x')
        archCommon=[(i+1, int(nodesCommon[i]), 'lrelu') for i in xrange(0,len(nodesCommon))]
    for ilayer,isize,iact in archCommon:
        x = Dense(isize,
                  kernel_initializer='glorot_normal',
                  bias_initializer='glorot_uniform')(x)
        if batchNorm:
            x=BatchNormalization()(x)
        if dropoutRate:
            x = Dropout(dropoutRate)(x)
        if iact=='lrelu':
            x = LeakyReLU(0.2)(x)


    def getRegDNNForParameter(p,inputs,arch,batchNorm,dropoutRate,pfix):
        dnn=inputs
        for ilayer,isize,iact in arch:
            dnn = Dense(isize, 
                        kernel_initializer='glorot_normal', 
                        bias_initializer='glorot_uniform',
                        name='dense_%s_%d_%s'%(p,ilayer,pfix))(dnn)
            if batchNorm:
                dnn=BatchNormalization(name='bn_%s_%d_%s'%(p,ilayer,pfix))(dnn)
            if dropoutRate: 
                dnn=Dropout(dropoutRate)(dnn)
            if iact=='lrelu':
                dnn = LeakyReLU(0.2)(dnn)
        dnn=Dense(1,
                  kernel_initializer='glorot_normal', 
                  bias_initializer='glorot_uniform',
                  activation='linear',
                  name='out_%s_%s'%(p,pfix))(dnn)
        return dnn


    nodesInd=nodesInd.split('x')
    archInd=[(i+1, int(nodesInd[i]), 'lrelu') for i in xrange(0,len(nodesInd))]    
    Outputs=[]
    for ic in xrange(0,nregclasses):

        if len(Outputs)>0 and recursiveInputs:
            iin=concatenate( [x] + Outputs )
        else:
            iin=x

        iout=[]
        for p in pList:
            iout.append( getRegDNNForParameter(p=p,
                                               inputs=iin,
                                               arch=archInd,
                                               batchNorm=batchNorm,
                                               dropoutRate=dropoutRate,
                                               pfix='c%d'%ic) 
                         )            
        Outputs.append( concatenate(iout,name='out_%d'%ic) )

    #build the final model
    return Model(inputs=Inputs, outputs=Outputs)


MODELLIST= { 
    #non-parametric: mean
    0:   { 'predLabels':['mu'],     "loss":['ahuber_0p50_1p00_loss'],
           "method":oneParamRegDNN, 'arch':'32x16x4', "dropout":None, "batchNorm": True, "learningrate":0.0001, "nepochs":50,  "batchsize":1024 },

    #non-parametric: mean+quantiles
    100:   { 'predLabels':['mu','qm','qp'], "loss":['quantile_50_loss','quantile_16_loss','quantile_84_loss'],
             "method":oneParamRegDNN, 'arch':'32x16x4', "dropout":None, "batchNorm": True, "learningrate":0.0001, "nepochs":50, "batchsize":1024 },

    #semi-parametric (original)
    200:  {  'predLabels':['mu','sigma','aL','aR','offset'], 'loss':[],
             'method':semiParamRegDNN, 'arch':':32x16x4',   "dropout":None, "batchNorm": True, "learningrate":0.0001,  "nepochs":100,  "batchsize":1024 },

    #semi-parametric (v2)
    300:  {  'predLabels':['mu','sigma','sigmaL','fL','sigmaR','fR'], 'loss':[],
             'method':semiParamRegDNN, 'arch':':32x16x4',   "dropout":None, "batchNorm": True, "learningrate":0.0001,  "nepochs":100,  "batchsize":1024 },

    #semi-parametric v2, with only one DNN for six parameters
    400:  {  'predLabels':['mu','sigma','sigmaL','fL','sigmaR','fR'], 'loss':[],
             'method':semiParamSingleRegDNN, 'arch':'120x96x24',   "dropout":None, "batchNorm": True, "learningrate":0.0001,  "nepochs":50,  "batchsize":1024 },
    }

#add variations of the main models
import copy
_archScan=['32x16x4','64x32x16','128x64x32','64:32x16x4','128:64x32x16','64x16:32x16x4','128x16:32x16x4','64x32x16x4','128:64x32x16x4','128x32:64x32x16x4']
_dropoutScan=[0.1,0.2]
_modelScan=[(arch,dropout) for arch in _archScan for dropout in _dropoutScan]
for _ims in xrange(0,len(_modelScan)):
    arch,dropout=_modelScan[_ims]
    for _m in [0,100,200,300, 400]:

        if _m<200 and ':' in arch : continue
        MODELLIST[_ims+1+_m]=copy.deepcopy(MODELLIST[_m])

        MODELLIST[_ims+1+_m]['arch']=arch
        if _m in [200,300] and not ':' in arch:
            MODELLIST[_ims+1+_m]['arch']=':'+arch
        MODELLIST[_ims+1+_m]['dropout']=dropout
        MODELLIST[_ims+1+_m]['nepochs']=int(MODELLIST[_m]['nepochs']/(1-dropout)**2)
#import pprint
#pprint.pprint(MODELLIST)
