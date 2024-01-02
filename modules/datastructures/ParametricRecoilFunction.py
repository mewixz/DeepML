import ROOT
import numpy as np
import copy
from scipy.interpolate import UnivariateSpline

class ParametricRecoilFunction:

    def __init__(self,model,paramList={},valRange=None,verbose=False):

        """constructor"""

        #configure
        self.model=model
        self.paramList=copy.deepcopy(paramList)
        self.valRange=valRange
        self.verbose=verbose
        
        #compute normalization
        self.norm=self.normalization()

        #inverse of the CDF function
        self.cdf_inv=None


    def normalization(self):

        """computes the normalization"""

        norm=0

        #gaussian double expo
        if self.model=='gd':
            mu,sigma,aL,aR,offset=[self.paramList[x] for x in ['mu','sigma','aL','aR','offset']]
            if self.valRange:                
                tval=[-(ROOT.TMath.Pi()+mu)/sigma,(ROOT.TMath.Pi()-mu)/sigma]
                norm =(ROOT.TMath.Exp(0.5*aL**2)/aL)*(ROOT.TMath.Exp(-aL**2)-ROOT.TMath.Exp(aL*tval[0]))
                norm+=(ROOT.TMath.Exp(0.5*aR**2)/aL)*(ROOT.TMath.Exp(-aR**2)-ROOT.TMath.Exp(-aR*tval[1]))
                norm+=ROOT.TMath.Sqrt(ROOT.TMath.Pi()/2)*(ROOT.TMath.Erf(aL/ROOT.TMath.Sqrt(2))+ROOT.TMath.Erf(aR/ROOT.TMath.Sqrt(2)))
                norm+=(tval[1]-tval[0])*offset
            else:
                norm =1./(aL*ROOT.TMath.Exp(0.5*(aL**2)))
                norm+=1./(aR*ROOT.TMath.Exp(0.5*(aR**2)))
                norm+=ROOT.TMath.Sqrt(ROOT.TMath.Pi()/2)*(ROOT.TMath.Erf(aL/ROOT.TMath.Sqrt(2))+ROOT.TMath.Erf(aR/ROOT.TMath.Sqrt(2)))
            norm *= sigma

        #bifurcated gauss+gauss
        if self.model=='bgsum':
            mu,sigma,sigmaL,fL,sigmaR,fR=[self.paramList[x] for x in ['mu','sigma','sigmaL','fL','sigmaR','fR']]
            if self.valRange:
                norm  = sigma*ROOT.TMath.Erf((ROOT.TMath.Pi()+mu)/(ROOT.TMath.Sqrt(2)*sigma))
                norm += sigma*ROOT.TMath.Erf((ROOT.TMath.Pi()-mu)/(ROOT.TMath.Sqrt(2)*sigma))*(1+fL)/(1+fR)
                norm += fL*sigmaL*ROOT.TMath.Erf((ROOT.TMath.Pi()+mu)/(ROOT.TMath.Sqrt(2)*sigmaL))
                norm += fR*sigmaR*ROOT.TMath.Erf((ROOT.TMath.Pi()-mu)/(ROOT.TMath.Sqrt(2)*sigmaR))*(1+fL)/(1+fR)
            else:
                norm  = sigma*(1+(1+fL)/(1+fR))
                norm += fL*sigmaL
                norm += fR*sigmaR*((1+fL)/(1+fR))
            norm *= ROOT.TMath.Sqrt(ROOT.TMath.Pi()/2)

        return 1./norm

    def eval(self,x):
        """evaluate PDF at a value of x"""

        #check if value with range
        if self.valRange:
            if x>self.valRange[1] or x<self.valRange[0]:
                return 0.

        #gaussian double expo
        if self.model=='gd':
            mu,sigma,aL,aR,offset=[paramList[p] for p in ['mu','sigma','aL','aR','offset']]
            t=(x-mu)/sigma
            ft=offset
            if t<-aL:
                ft+=ROOT.TMath.Exp(aL*(t+0.5*aL))
            elif t>aR:
                ft+=ROOT.TMath.Exp(-aR*(t-0.5*aR))
            else:
                ft+=ROOT.TMath.Exp(-0.5*t*t)

        #bifurcated gauss+gauss
        if self.model=='bgsum':
            mu,sigma,sigmaL,fL,sigmaR,fR=[self.paramList[p] for p in ['mu','sigma','sigmaL','fL','sigmaR','fR']]
            if x<mu:
                ft  = ROOT.TMath.Exp(-0.5*pow((x-mu)/sigma,2))
                ft += fL*ROOT.TMath.Exp(-0.5*pow((x-mu)/sigmaL,2))
            else:
                ft  = ROOT.TMath.Exp(-0.5*pow((x-mu)/sigma,2))
                ft += fR*ROOT.TMath.Exp(-0.5*pow((x-mu)/sigmaR,2))
                ft *= (1+fL)/(1+fR)
        
        #normalize
        ft *= self.norm
        return ft

    def getEventPDF(self,bins):
        
        """returns the event PDF"""

        return [ self.eval(x) for x in bins ]

    def generateCDFInv(self,xmin,xmax,nintpts=100):
        
        """generates the CDF from integrating the PDF in a grid
        then inverts it with a spline function"""

        dx=(xmax-xmin)/float(nintpts)
        x=np.arange(xmin,xmax,dx)
        cdf=np.cumsum(self.getEventPDF(x))
        cdf*=dx

        cdf_inv=[]
        for i in xrange(0,len(cdf)):
            if cdf[i]<1e-10 : continue
            if abs(1-cdf[i])<1e-10: continue
            if len(cdf_inv)>0 and abs(cdf[i]-cdf_inv[-1][1])<1e-10 : continue
            cdf_inv.append( (x[i],cdf[i]) )        
        cdf_inv=np.array(cdf_inv)
        self.cdf_inv = UnivariateSpline(cdf_inv[:,1],cdf_inv[:,0])
        self.cdf_inv.set_smoothing_factor(0.5)


    def getRandom(self,xmin,xmax,nintpts=100):
    
        """ generate a random number based on this PDF """

        #determine the cdf if needed
        if not self.cdf_inv:
            while True:
                try:
                    self.generateCDFInv(xmin,xmax,nintpts)
                    break
                except Exception as e:
                    print e
                    nintpts *= 2
                    if nintpts>1e4:
                        raise ValueError('Unable to find enough points to interpolate inverse CDF')
                    pass
        
        #generate random
        r=np.random.uniform()
        return self.cdf_inv(r)


def evalSemiParametricCorrections(y_pred):

    """wrapper to get a random value and the peak out of the regressed semi-parametric PDF"""

    def rangeTransform(x,a,b):
        return a+0.5*(b-a)*(ROOT.TMath.Sin(x)+1.0)

    #DeepJetCore will call this sequentially for the first and second set of parameters
    #instead of calling only once with all parameters at once
    nevts=len(y_pred[0])
    print '@evalSemiParametricCorrections'
    print 'Evaluation may take some time...{0} events available'.format(nevts)

    result=[]
    paramList = {}
    for i in xrange(0,nevts):

        p_pred=y_pred[0][i]
        paramList['mu']     = rangeTransform(p_pred[0],-3,3)
        paramList['sigma']  = rangeTransform(p_pred[1],1e-3,5)
        paramList['sigmaL'] = rangeTransform(p_pred[2],1e-3,10)
        paramList['fL']     = rangeTransform(p_pred[3],1e-6,1)
        paramList['sigmaR'] = rangeTransform(p_pred[4],1e-3,10)
        paramList['fR']     = rangeTransform(p_pred[5],1e-6,1)
        prf_lne1            = ParametricRecoilFunction('bgsum',paramList)
        
        p_pred=y_pred[1][i]
        paramList['mu']     = rangeTransform(p_pred[0],-0.5*ROOT.TMath.Pi(),0.5*ROOT.TMath.Pi())
        paramList['sigma']  = rangeTransform(p_pred[1],1e-3,ROOT.TMath.Pi())
        paramList['sigmaL'] = rangeTransform(p_pred[2],1e-3,1e4)
        paramList['fL']     = rangeTransform(p_pred[3],1e-6,1.0)
        paramList['sigmaR'] = rangeTransform(p_pred[4],1e-3,1e4)
        paramList['fR']     = rangeTransform(p_pred[5],1e-6,1.0)
        prf_e2              = ParametricRecoilFunction('bgsum',paramList,valRange=(-ROOT.TMath.Pi(),ROOT.TMath.Pi()))
        
        result.append( [prf_lne1.getRandom(-6,6), 
                        prf_lne1.paramList['mu'],
                        prf_e2.getRandom(-ROOT.TMath.Pi(),ROOT.TMath.Pi()), 
                        prf_e2.paramList['mu']] )

    print 'Result has',len(result)
    return np.array(result)

            
