from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad
from DeepJetCore.stopwatch import stopwatch
from RecoilModels import MODELLIST
from argparse import ArgumentParser
from ParametricRecoilFunction import evalSemiParametricCorrections
import numpy

def fileTimeOut(fileName, timeOut):
    tdfto(fileName, timeOut)

class TrainData_Recoil(TrainData):
    '''
    Class for Recoil training
    '''
    
    def __init__(self,args):
        import numpy
        TrainData.__init__(self)

        #parse specific arguments to this class
        parser=ArgumentParser('TrainData_Recoil parser')
        parser.add_argument('--target',
                            help='Regression target',
                            default='lne1')
        parser.add_argument('--modelMethod',
                            help='model method',
                            default=None,
                            type=int)
        parser.add_argument('--varList',
                            help='Variables to use in the regression', 
                            default='htkmet_pt,htkmet_phi,htkmet_n,htkmet_scalar_ht,htkmet_scalar_sphericity,htkmet_rho')
        parser.add_argument('--sel',
                            help='Apply branch flag [%default]', 
                            default=None)
        parser.add_argument('--addFunctorOutput',
                            help='add functor output [%default]', 
                            action="store_true",
                            default=False)
        args=parser.parse_args(args.split())

        #event selection
        self.selection=args.sel
        
        #setting DeepJet specific defaults
        self.treename="tree"
        self.undefTruth=[]
        
        self.referenceclass='flatten'
        self.truthclasses=['isGood']
        
        # register ALL branches you intend to use in the numpy recArray style
        # no need to register those that use the DeepJetCore conversion funtions
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)

        #regression target
        self.regressiontargets=args.target.split(',')

        regressionBranches=args.varList.split(',')
        otherBranches=[t for t in self.regressiontargets]
        if self.selection: otherBranches.append(self.selection)
        self.registerBranches(regressionBranches+otherBranches)
        self.addBranches(regressionBranches)

        #make these distributions uniform in the training (disabled for the moment)
        self.weightbranchX='nVertF'
        self.weightbranchY='htkmet_scalar_sphericity'
        self.weight_binX = numpy.array([0,100],dtype=float)
        self.weight_binY = numpy.array([0,10000],dtype=float)

        #define the regression classes
        self.modelMethod=args.modelMethod
        self.predLabelsPerTruth=MODELLIST[self.modelMethod]['predLabels']
        for i in xrange(0,len( self.regressiontargets )):
            for j in xrange(0,len( self.predLabelsPerTruth )):
               self.regressiontargetclasses.append( self.predLabelsPerTruth[j] +'_'+ self.regressiontargets[i] ) 

        print "self.regressiontargetclasses", self.regressiontargetclasses
        setattr(self,'customlabels',self.regressiontargetclasses) # will be used by DeepJetCore/evaluation.py

        #add a function of the regressed values to compute peak and random correction on the fly
        if args.addFunctorOutput and len(self.regressiontargets)==2 and self.modelMethod>=300:
            setattr(self, 'predictionFunctor', evalSemiParametricCorrections)
            predictionOutputs=[]
            for t in self.regressiontargets: predictionOutputs += ['%s_reg'%t,'%s_peak'%t]
            setattr(self, 'predictionFunctorClasses', predictionOutputs)
            print '[TrainData_Recoil] Added hook for evalSemiParametricCorrections'
        else:
            print '[TrainData_Recoil] No hook for post-prediction evaluation is set'

        #print some information
        print '-'*50
        print 'Configured TrainData_Recoil'
        print 'Target classes to regress are',self.regressiontargetclasses
        print 'Truth will be set to',self.regressiontargets
        print self.selection,'events will be removed'
        print 'Variables used in the training are',regressionBranches
        print '-'*50
             
        #Call this and the end!
        self.reduceTruth(None)
        
    # this funtion defines the conversion
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        #for reacArray operations
        Tuple = self.readTreeFromRootToTuple(filename)

        #create weights and remove indices
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            if self.selection:
                print 'Removing events selected with',self.selection
                notremoves -= Tuple[self.selection].view(numpy.ndarray)
            weights=notremoves
            #print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            #print('creating weights')
            weights= weighter.getJetWeights(Tuple)
        else:
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)

        truthtuple =  Tuple[self.truthclasses]

        
        #print(self.truthclasses)
        #that would be for labels
        alltruth=self.reduceTruth(truthtuple)
        
        reg_truth=[Tuple[x].view(numpy.ndarray) for x in self.regressiontargets]
        
        #stuff all in one long vector
        x_all=MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        #print(alltruth.shape)
        if self.remove:
            #notremoves=notremoves - (1-truthtuple)
            #print('remove')
            weights=weights[notremoves > 0]
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            reg_truth=[x[notremoves > 0] for x in reg_truth]
            print len(Tuple),'->',len(x_all),'after remove'

        newnsamp=x_all.shape[0]
        #print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        self.w=[weights]
        self.x=[x_all]
    
        self.y=[]
        if self.modelMethod>=200:
            #for semi-parametric models the N parameters are adjusted to 1 truth
            self.y=reg_truth
            print 'Semi-parametric model trained with %d size truth array'%len(self.y)
        else:
            #for other models the truth needs to be replicated for each parameter
            for i in xrange(0,len(reg_truth)):
                for j in xrange(0,len(self.predLabelsPerTruth)):
                    self.y.append( reg_truth[i] )
            print 'Simple model trained with %d size truth array'%len(self.y)
        
    
