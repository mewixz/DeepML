import root_pandas as rpd
from   root_pandas import read_root
import pickle
import gzip
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import os
import sys
import time
import re

defaultQuantileRegressorConfig={
   'truthLabels':['SingleMuon_Run2016C'],
   'simLabels':['DYJetsToLL_M50'],
   'inputDir':'/eos/cms/store/cmst3/user/psilva/Wmass/RecoilTest_regress-v2/Chunks/',
   'tree':'data',
   'selBranch':'isZ',
   'selVal':1,
   'evtBranches':["nvert", "rho", "vis_pt"],
   'varBranches':["tkmet_pt", "tkmet_phi"],
   'wgtBranch':"wgt"
   }

class QuantileRegressor:
   """wraps the procedure of regressing N-quantiles based on original code by P. Musella et al"""

   def __init__(self, label,config=defaultQuantileRegressorConfig):

      self.label   = label
      self.outDir='%s/train'%self.label
      os.system('mkdir -p %s'%self.outDir)
      print 'QuantileRegressor outputs will be stored in',self.outDir
      
      #configure the regressor
      print 'Configuration to be used is'
      for key in config:
         setattr(self,key,config[key])
         print key,'=',config[key]

      #branches to extract from the tree
      self.branches = [self.selBranch,self.wgtBranch]+self.evtBranches+self.varBranches
      
      #data to use
      self.truth=None
      self.sim=None

   def loadDF(self,truthF=None,mcF=None):
      """if pickle files are already available use them, otherwise load from ROOT again"""
      if truthF and mcF:
         self.truth = pickle.load(gzip.open(truthF))
         self.sim  = pickle.load(gzip.open(mcF))
      else:
         for b in [False, True]: 
            self.loadDFFromROOT(isTruth=b)
      ntruth,nsim=len(self.truth.index),len(self.sim.index)
      print 'Events available for truth:%d sim:%d'%(ntruth,nsim)
      if ntruth==0 or nsim==0:
         raise ValueError('Invalid number of events')

   def loadDFFromROOT(self,isTruth,savePickle=True):
      """load dataframes from the input files"""

      #Build the list of files
      tagList=self.truthLabels if isTruth else self.simLabels
      fileList=[]
      for f in os.listdir(self.inputDir):
         for t in tagList:
            if not t in f : continue
            fileList.append( os.path.join(self.inputDir,f) )
      print len(fileList),'tree files found matching',tagList

      #add trees to a data frame
      for f in fileList:
         try:
            df = pd.concat([df, rpd.read_root(f,self.ttree,columns=self.branches)])
         except:
            df = rpd.read_root(f,self.tree,columns=self.branches)
      print "Events read:", len(df.index)

      #select
      df = df.loc[ df[self.selBranch]== self.selVal ]
      df = df.drop(columns=[self.selBranch],axis=1)
      df = df.reset_index()
      print "Events selected:",len(df.index)

      #assign selected dataframe
      if isTruth: self.truth=df
      else:      self.sim=df

      #save summary
      if not savePickle: return 
      outputName='%s/qreg_%s.pck'%(self.outDir,'truth' if isTruth else 'sim')
      pickle.dump(df, 
                  gzip.open(outputName, 'wb'), 
                  protocol=pickle.HIGHEST_PROTOCOL)
      print 'Pandas dataframe stored in',outputName

   def trainQuantile(self, var, featureList, q, isTruth, useWeights = False, maxDepth = 3, minLeaf = 9):
      """ train the quantiles"""

      df = self.truth if isTruth else self.sim

      # quantile regressions features
      X     = df.loc[:,featureList]
      # target
      Y     = df[var]      
      #event weight
      w     = df[self.wgtBranch]

      #start a BDT to train the quantiles
      print 'Train q=%f for %s'%(q,"truth" if isTruth else "sim")
      clf = GradientBoostingRegressor(loss='quantile', 
                                      alpha=q,
                                      n_estimators=250,
                                      max_depth=maxDepth,
                                      learning_rate=.1, 
                                      min_samples_leaf=minLeaf,
                                      min_samples_split=minLeaf)
      #fit
      t0 = time.time()
      if (useWeights) :
         print 'Applying weights in the fit'
         w=w.abs()
         clf.fit(X, Y, w)
      else:
         clf.fit(X, Y)         
      t1 = time.time()
      print "Fit took = ", t1-t0,'s'

      #save results
      outputName='%s/%s_weights_%s_q%3.3f_%s.pck'%(self.outDir,self.label,
                                                   "truth" if isTruth else "sim",
                                                   q,
                                                   var)
      with gzip.open(outputName, 'wb') as cache:
         pickle.dump((var,featureList,self.wgtBranch,useWeights), cache, protocol=pickle.HIGHEST_PROTOCOL)
         pickle.dump(clf,cache, protocol=pickle.HIGHEST_PROTOCOL)
      print 'Weights and information to use them were saved in',outputName      

   def runTrainQuantilesLoop(self,qList,usePrevVarAsFeature=True):
      """wrap up the training of quantiles"""

      #build a task list: loop over variables, quantiles, truth and sim
      task_list=[]
      for iv in xrange(0,len(self.varBranches)):
         var=self.varBranches[iv]
         featureList=self.evtBranches
         if usePrevVarAsFeature:
            featureList += self.varBranches[0:iv]
            for q in qList:
               for isTruth,useWeights in [(False,True),(True,False)]:
                  task_list.append( (var,featureList,q,isTruth,useWeights) )

      #run jobs
      print 'Starting %d tasks, this may take a while'%len(task_list)
      for args in task_list:
         var,featureList,q,isTruth,useWeights=args
         self.trainQuantile(var=var,
                            featureList=featureList,
                            q=q,
                            isTruth=isTruth,
                            useWeights=useWeights)

   def readQuantileRegressors(self,inputDir):

      regList={}
      featureList={}

      for f in os.listdir(inputDir):

         #filter weights file for simulation requiring truth is also available
         if not 'weights' in f : continue
         if not 'sim' in f : continue
         simF=os.path.join(inputDir,f)
         truthF=simF.replace('_sim_','_truth_')
         if not os.path.isfile(truthF) : continue

         #parse quantile
         q=re.findall("\d+\.\d+",f)[0]
      
         #read info
         with gzip.open(simF,'rb') as cache:
            var,features,_,_=pickle.load(cache)
            sim_clf=pickle.load(cache)      
         with gzip.open(truthF,'rb') as cache:
            pickle.load(cache)
            truth_clf=pickle.load(cache)      

         #store
         if not var in regList: 
            regList[var]=[]
            featureList[var]=features
         regList[var].append( (q,sim_clf,truth_clf) )

      #sort quantiles
      for var in regList:
         regList[var]=sorted(regList[var], key=lambda q: q[0])
   
      return regList,featureList

