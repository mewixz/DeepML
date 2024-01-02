from DeepJetCore.training.training_base import training_base
from RecoilModels import *
from Metrics import *
import copy

#start a training base class (also does the parsing)    
#if running locally for >24h you may want to set renewtokens=True
train=training_base(testrun=False,renewtokens=False)
model,targets=train.keras_model_method.split(':')
model=int(model)
targets=targets.split(',')
if not model in MODELLIST:
    raise ValueError('Unknown model %d'%model)
print 'Preparing model',model
print MODELLIST[model]
print 'Targets set to',targets
      
#configure model
metricsList = ['mae','mse']
if 'lne1' in targets : metricsList.append( scale_metric )
if 'e2'   in targets:  metricsList.append( dir_metric )
predictionLabels  = [x+'_'+y 
                     for x in MODELLIST[model]['predLabels'] 
                     for y in targets]

train.setModel( model=MODELLIST[model]['method'],                    
                arch=MODELLIST[model]['arch'],
                dropoutRate=MODELLIST[model]['dropout'],
                batchNorm=MODELLIST[model]['batchNorm'] )

#build the list of loss functions to use
losses=[]
predictionLabels=[]
for i in xrange(0,len(targets)):
    for j in xrange(0,len(MODELLIST[model]['predLabels'])):
        predictionLabels.append( MODELLIST[model]['predLabels'][j]+'_'+targets[i] )

        #for simple models a loss function needs to be declared per prediction label
        if model<200:
            lossName=MODELLIST[model]['loss'][j]
            losses.append( global_loss_list[lossName] )

    if model>=200 and model<300:
        lossName='gd_nooffset_loss' if targets[i]=='lne1' else 'gd_offset_loss'
        losses.append( global_loss_list[lossName] )
    if model>=300:
        lossName='bgsum_unbound_loss' if targets[i]=='lne1' else 'bgsum_mpi2pi_loss'
        losses.append( global_loss_list[lossName] )



train.defineCustomPredictionLabels(predictionLabels)
print(train.keras_model.summary())
print 'Loss functions'
print losses
print 'Output labels'
print predictionLabels


nepochs=MODELLIST[model]['nepochs']
for i in xrange(0,len(losses)):

    print 'Traing starting for target set #%d'%i

    #assign weight 1 only for the loss being trained
    loss_weights=[1 if j==i else 0 for j in xrange(0,len(losses))]
    train.compileModel(learningrate=MODELLIST[model]['learningrate'],
                       loss=losses,
                       loss_weights=loss_weights,
                       metrics=metricsList)
    
    curmodel,curhistory = train.trainModel(nepochs=nepochs*(i+1),
                                           batchsize=MODELLIST[model]['batchsize'],
                                           stop_patience=300, #stop after N-epochs if validation loss increases
                                           lr_factor=0.5,     #adapt learning rate if validation loss flattens
                                           lr_patience=15, 
                                           lr_epsilon=0.0001, 
                                           lr_cooldown=2, 
                                           lr_minimum=0.0001, 
                                           maxqsize=100       #play if file system is unstable
                                           )


    matchKey='_c%d'%i
    for layer in train.keras_model.layers:        
        if matchKey in layer.name:
            layer.trainable=False 
            print 'Froze training for',layer.name
    
