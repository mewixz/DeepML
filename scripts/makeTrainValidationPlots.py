import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parsePlotsFrom(data):
    """parse the json file and get the data"""
    x=[i+1 for i in xrange(0,len(data))]
    y={}
    for i in xrange(0,len(data)):
        epochData=data[i]
        for key in epochData.keys():
            key=str(key)
            if not key in y: y[key]=[]
            y[key].append(epochData[key])
        
    return x,y

def makeTrainValidationPlot(key,labels,xy_train,xy_val):
    """plot a figure of merit"""
    fig,ax=plt.subplots(num=key,figsize=(6,6))
    plt.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.1)

    colors=['r','g','b','c','m','k']
    ymin,ymax=1e9,-1e9
    for i in xrange(0,len(labels)):
        ax.plot(xy_train[i][0], xy_train[i][1], color=colors[i], linestyle=':', label='train')
        ax.plot(xy_val[i][0],   xy_val[i][1],   color=colors[i], linestyle='-', label='validation')
        ymin=min([ymin,min(xy_val[i][1][2:-1]),min(xy_train[i][1][2:-1])])
        ymax=max([ymax,max(xy_val[i][1][2:-1]),max(xy_train[i][1][2:-1])])
    plt.xlabel('Epoch')
    ytitle=key
    #if key=='loss': ytitle='Loss / min Loss'
    plt.ylabel(ytitle)
    plt.ylim(ymin=ymin,ymax=ymax)
    plt.legend(framealpha=0.0, fontsize=10,loc='upper right')    
    ax.text(0,1.02,'CMS preliminary simulation', transform=ax.transAxes)
    ax.text(1.0,1.02,r'W$\rightarrow\ell\nu$ (13 TeV)', transform=ax.transAxes,horizontalalignment='right')
    for i in xrange(0,len(labels)):
        ax.text(0.73, 0.95-i*0.09, labels[i],horizontalalignment='right', transform=ax.transAxes,fontsize=8)

    # if >2 orders of magnitude make it logscale
    #if key=='loss':
    #    plt.ylim(ymin=0.9,ymax=3)
    if np.log10(abs(ymax-ymin))>2: 
        if ymin<=0: plt.ylim(ymin=1e-2,ymax=ymax)
        ax.set_yscale('log')

    #fig.tight_layout()
    extList=['png','pdf']
    for ext in extList:
        fig.savefig('%s.%s'%(key,ext))
    print 'Saved %s.[%s]'%(key,','.join(extList))

allKeys=[]
modelData=[]
modelList=[x.split(':')[1] for x in sys.argv[1].split(',') if len(x)>0]
titleList=[x.split(':')[0] for x in sys.argv[1].split(',') if len(x)>0]
print titleList
print modelList
for model in modelList:
    data=None
    fname='%s/train/model/full_info.log'%model
    if os.path.isfile(fname):
        with open(fname) as f:
            data=json.loads(f.read())
        modelData.append( parsePlotsFrom(data) )
        
        #normalize to minimum loss
        #for key in ['loss','val_loss']:
        #    if key in modelData[-1][1]:
        #        miny=min(modelData[-1][1][key])
        #        modelData[-1][1][key]=[val/miny for val in modelData[-1][1][key]]

        allKeys += list(modelData[-1][1].keys())
    else:
        modelData.append(None)

allKeys=list(set(allKeys))
for key in allKeys:
    if key.find('val')==0 : continue

    titles=[]
    xy_train=[]
    xy_val=[]
    for i in xrange(0,len(modelList)):
        if not modelData[i]: continue
        if not key in modelData[i][1]: continue        
        if not 'val_'+key in modelData[i][1]: continue
        titles.append( titleList[i] )
        xy_train.append( (modelData[i][0],modelData[i][1][key]) )
        xy_val.append( (modelData[i][0],modelData[i][1]['val_'+key]) )

    if len(xy_train)<1 : continue
    makeTrainValidationPlot(key,titles,xy_train,xy_val)
    
