import numpy as np
import os
import ROOT
import array as array

class QuantileCorrector:
   
    """helper class to perfom quantile-based corrections based on the code by P. Musella et al"""

    def __init__(self,label,df,regList,featureList):

       #prepare the output directory
       os.system('mkdir -p %s/predict'%label)
       self.label=label

       #store regressors and features
       self.regList=regList
       self.featureList=featureList

       print 'Preparing quantile predictions (this may take a while)'
       self.qtls={}
       self.Y={}
       for var in self.regList:

           features=featureList[var]
           print 'Starting with',var
           print 'Features to use are',features

           #project features and run predictions
           self.qtls[var]=[]
           X=df.loc[:,features]
           for i in xrange(0,2):
               print '...running quantile predictions for',('sim' if i==0 else 'truth')
               self.qtls[var].append( np.array( [clf[i+1].predict(X) for clf in regList[var]] ) )
               
           #target variables
           self.Y[var]=df[var]
       
    def saveCorrections(self,outName):

        """loop over the events and correct the target variables"""

        #variables to regress
        varnames=self.Y.keys()

        #open an output file
        fOutName="%s/predict/%s"%(self.label,outName)
        fOut = ROOT.TFile(fOutName, "RECREATE")
        fOut.cd()
        outvarnames=['%s_%s'%(x,self.label) for x in varnames]
        t=ROOT.TNtuple('tree','tree',':'.join(outvarnames))

        #loop over events and fill tuple with corrections
        nevts=len(self.Y[ varnames[0] ])
        nq=len(self.qtls[varnames[0]][0])
        print 'Looping over %d to save corrected variables'%nevts,varnames
        print nq,'quantiles will be used in the correction'
        for iev in xrange(0,nevts):            
            ycorr=[
                self.correctEvent(y=self.Y[var][iev],
                                  simqtls=[self.qtls[var][0][iq][iev] for iq in xrange(0,nq)],
                                  truthqtls=[self.qtls[var][1][iq][iev] for iq in xrange(0,nq)])
                for var in self.Y
                ]
            t.Fill(array.array('f',ycorr))
        
        #save file
        fOut.cd()
        t.Write()
        fOut.Close()
        print 'Summary tree saved @',fOutName


    def correctEvent(self,y,simqtls,truthqtls):

        """the actual correction algorithm"""

        #find the sim quantile
        qsim =0
        while qsim < len(simqtls): # while + if, to avoid bumping the range
            if simqtls[qsim] < y:
                qsim+=1
            else:
                break

        #determine neighouring quantiles
        if qsim == 0:
            qsim_low,qtruth_low   = 0,0    #lower bound... check this
            qsim_high,qtruth_high = simqtls[qsim],truthqtls[qsim]
        elif qsim < len(simqtls):
            qsim_low,qtruth_low   = simqtls[qsim-1],truthqtls[qsim-1]
            qsim_high,qtruth_high = simqtls[qsim],truthqtls[qsim]
        else:
            qsim_low,qtruth_low   = simqtls[qsim-1],truthqtls[qsim-1]
            qsim_high,qtruth_high = simqtls[len(simqtls)-1]*1.2,truthqtls[len(truthqtls)-1]*1.2
            #sets the value for the highest quantile 20% higher
            
        #linear correction
        return (qtruth_high-qtruth_low)/(qsim_high-qsim_low) * (y - qsim_low) + qtruth_low
