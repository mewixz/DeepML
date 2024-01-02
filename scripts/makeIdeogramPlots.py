import os
import sys
import ROOT
from ParametricRecoilFunction import *
import optparse

def rangeTransform(x,a,b):
    """transform to a limited range as Minuit does (cf. https://root.cern.ch/download/minuit.pdf Sec. 1.2.1)"""
    return a+0.5*(b-a)*(ROOT.TMath.Sin(x)+1.0)


def makeIdeogramPlots(data,
                      maxEvts,
                      nbins={'lne1':100,'e2':100},
                      xmin={'lne1':0,'e2':-ROOT.TMath.Pi()},
                      xmax={'lne1':100,'e2':ROOT.TMath.Pi()}):

    """Uses the regressed PDF to re-destribute the contribution of a single event"""

    dogPDF=hasattr(data,'sigma') or hasattr(data,'qp')

    #prepare histograms
    ideograms={}
    dx,xcen,xof={},{},{}
    for c in nbins:
        dx[c]=float((xmax[c]-xmin[c])/nbins[c])
        xcen[c]=[xmin[c]+(xbin+0.5)*dx[c] for xbin in xrange(0,nbins[c])]
        xof[c] = xcen[c][-1]+0.5*dx[c]

    def fillIdeogram(key,c,val,truth,wgt=1.0):
        """creates an ideogram if not existing and fills it up"""
        name=key+'_'+c
        if not name in ideograms:
            xtit='Recoil scale [GeV]' if c=='lne1' else 'Recoil direction [rad]'
            ideograms[name]=ROOT.TH1F('h'+name,'%s;%s;Events'%(name,xtit),nbins[c],xmin[c],xmax[c])
            ideograms[name].Sumw2()
            ideograms[name+'corr']=ROOT.TH2F('hcorr'+name,'%s;True %s; %s;Events'%(name,xtit,xtit),nbins[c],xmin[c],xmax[c],nbins[c],xmin[c],xmax[c])
            ideograms[name+'corr'].Sumw2()
        ideograms[name].Fill(val,wgt)
        ideograms[name+'corr'].Fill(truth,val,wgt)

    def fillIdeogramResol(key,c,reg,truth,xtruth,wgt=1.0):
        """creates an histogram for the resolution if not available and fills it up"""
        
        name=key+'_'+c+'_diff'
        if not name in ideograms:
            if c=='lne1':
                ytit='h-h_{true} [GeV]'
                yran=(-100,100)
            else:
                ytit='#phi-#phi_{true} [rad]'
                yran=(-ROOT.TMath.Pi(),ROOT.TMath.Pi())

            ideograms[name]=ROOT.TH2F('h'+name,
                                      ';Recoil scale [GeV];%s;Events'%ytit,
                                      nbins['lne1'],xmin['lne1'],xmax['lne1'],
                                      100,yran[0],yran[1])
            ideograms[name].Sumw2()
        if c=='lne1':
            diff=reg-truth
        if c=='e2':
            diff=ROOT.TVector2.Phi_mpi_pi(reg-truth)
        ideograms[name].Fill(xtruth,diff,wgt)

    #determine which parameterizations have been trained
    pdfParams={}
    data.GetEntry(0)
    semiParamTrain=None
    for c in ['lne1','e2']:
         if hasattr(data,'mu_'+c):
             pdfParams[('peak',c)]=['mu_'+c]
             if hasattr(data,'offset_'+c):
                 semiParamTrain='gd'
                 pdfParams[('gauss',c)]=pdfParams[('peak',c)]+['sigma_'+c]
                 pdfParams[(semiParamTrain,c)]=pdfParams[('gauss',c)]+['aL_'+c,'aR_'+c,'offset_'+c]
             if hasattr(data,'fL_'+c):
                 semiParamTrain='bgsum'
                 pdfParams[('gauss',c)]=pdfParams[('peak',c)]+['sigma_'+c]
                 pdfParams[(semiParamTrain,c)]=pdfParams[('gauss',c)]+['sigmaL_'+c,'fL_'+c,'sigmaR_'+c,'fR_'+c]
             elif hasattr(data,'qm_'+c):
                 pdfParams[('gauss',c)]=pdfParams[('peak',c)]+['qm_'+c,'qp_'+c]
    print 'The following PDF parameters are available in this training'
    print pdfParams

    def fillPDFParamValues():

        """wraps up the procedure used to fill the parameters of the PDFs for each event"""

        pdfParamVals={}
        #loop over the required PDFs and fill with the values
        for key in pdfParams:
            pdfParamVals[key]=[]

            for p in pdfParams[key]:
                val=getattr(data,p)

                #gaussian double expo
                if semiParamTrain=='gd':
                    if key[1]=='lne1':
                        if p=='mu_lne1':        val=rangeTransform(val,-3,3) 
                        elif p=='offset_lne1':  val=0
                        else:                   val=rangeTransform(val,1e-3,5)
                    if key[1]=='e2':
                        if p=='mu_e2':       val=rangeTransform(val,-ROOT.TMath.Pi(),ROOT.TMath.Pi()) 
                        elif p=='offset_e2': val=rangeTransform(val,0,1)
                        elif p=='sigma_e2':  val=rangeTransform(val,1e-3,2*ROOT.TMath.Pi())
                        else:                val=rangeTransform(val,1e-3,ROOT.TMath.Pi())                        

                #bifur gauss+gaussian
                if semiParamTrain=='bgsum':
                    if key[1]=='lne1':
                        if p=='mu_lne1':        val=rangeTransform(val,-3,3) 
                        elif p=='sigma_lne1':   val=rangeTransform(val,1e-3,5) 
                        elif p=='sigmaL_lne1':  val=rangeTransform(val,1e-3,10) 
                        elif p=='fL_lne1':      val=rangeTransform(val,1e-6,1) 
                        elif p=='sigmaR_lne1':  val=rangeTransform(val,1e-3,10)
                        elif p=='fR_lne1':      val=rangeTransform(val,1e-6,1)  
                    if key[1]=='e2':
                        if p=='mu_e2':          val=rangeTransform(val,-0.5*ROOT.TMath.Pi(),0.5*ROOT.TMath.Pi()) 
                        elif p=='sigma_e2':     val=rangeTransform(val,1e-3,ROOT.TMath.Pi()) 
                        elif p=='sigmaL_e2':    val=rangeTransform(val,1e-3,1e4) 
                        elif p=='fL_e2':        val=rangeTransform(val,1e-6,1.0) 
                        elif p=='sigmaR_e2':    val=rangeTransform(val,1e-3,1e4)
                        elif p=='fR_e2':        val=rangeTransform(val,1e-6,1.0)  

                pdfParamVals[key].append( val )
            
            #for the non-parametric quantile-regression transform to a gaussian-like
            if not semiParamTrain and key[0]=='gauss':
                valList=pdfParamVals[key]
                if key[1]=='lne1':
                    sL=abs(valList[2]-valList[0])
                    sR=abs(valList[0]-valList[1])
                else:
                    sL=abs(valList[2]-valList[0])
                    sR=abs(valList[0]-valList[1])
                pdfParamVals[key]=[valList[0],0.5*(sL+sR)]

        #all done
        return pdfParamVals

    #loop over events
    nentries=data.GetEntries() 
    if maxEvts>0: nentries=min(nentries,maxEvts)
    for i in xrange(0,nentries):

        data.GetEntry(i)
        pdfParamVals=fillPDFParamValues()
        
        for c in ['lne1','e2']:
            
            truth=data.trueh if c=='lne1' else data.truephi
            fillIdeogram('true',c,truth,truth)

            if c=='lne1':
                rawVal=data.tkmet_pt
                rawCorr=(2.075-1)*rawVal
            if c=='e2':
                rawVal=data.tkmet_phi
                rawCorr=0
            fillIdeogram('raw',c,rawVal+rawCorr,truth)
            fillIdeogramResol('raw',c,rawVal+rawCorr,truth,data.trueh)

            for pdf in ['peak','gauss',semiParamTrain]:

                key=(pdf,c)
                if not key in pdfParamVals: continue

                #peak-based correction
                if pdf=='peak':
                    if c=='lne1':
                        regVal=rawVal*ROOT.TMath.Exp(pdfParamVals[key][0])
                    if c=='e2':
                        regVal=ROOT.TVector2.Phi_mpi_pi(rawVal+pdfParamVals[key][0])
                    fillIdeogram('peak',c,regVal,truth)
                    fillIdeogramResol('peak',c,regVal,truth,data.trueh)

                #gaussian-based correction
                if pdf=='gauss':

                    mu,sigma=pdfParamVals[key]

                    if c=='lne1':
                        bins   = [ ROOT.TMath.Log(xcen[c][i]/rawVal)                       for i in xrange(0,len(xcen[c])) ]
                        g_pdf  = [ ROOT.TMath.Gaus(bins[i],mu,sigma,True)*dx[c]/xcen[c][i] for i in xrange(0,len(xcen[c])) ]
                        g_norm = sum(g_pdf)
                        fillIdeogram('gauss',c, xof[c], truth, 1-g_norm)
                    if c=='e2':
                        bins   = [ ROOT.TVector2.Phi_mpi_pi(xcen[c][i]-rawVal) for i in xrange(0,len(xcen[c])) ]
                        g_pdf  = [ ROOT.TMath.Gaus(bins[i],mu,sigma,True)*dx[c]      for i in xrange(0,len(xcen[c])) ]
                        g_norm = sum(g_pdf)
                        g_pdf  = [ x/g_norm for x in g_pdf ]

                    for xbin in xrange(0,len(g_pdf)):
                        fillIdeogram('gauss',c, xcen[c][xbin], truth, g_pdf[xbin])
                        fillIdeogramResol('gauss',c,xcen[c][xbin],truth,data.trueh,g_pdf[xbin])

                #semi-parametric training
                if not pdf: continue
                if pdf!=semiParamTrain: continue

                valRange=None if c=='lne1' else [-ROOT.TMath.Pi(),ROOT.TMath.Pi()]

                #gaussian double expo-based correction
                if pdf=='gd':
                    paramList={'mu':pdfParamVals[key][0],'sigma':pdfParamVals[key][1],'aL':pdfParamVals[key][2],'aR':pdfParamVals[key][3],'offset':pdfParamVals[key][4]}
                else:
                    paramList={'mu':pdfParamVals[key][0],'sigma':pdfParamVals[key][1],'sigmaL':pdfParamVals[key][2],'fL':pdfParamVals[key][3],'sigmaR':pdfParamVals[key][4],'fR':pdfParamVals[key][5]}
                prf=ParametricRecoilFunction(semiParamTrain,paramList,valRange)

                if c=='lne1':
                    bins   = [ ROOT.TMath.Log(xcen[c][i]/rawVal) for i in xrange(0,len(xcen[c])) ]
                    g_pdf  = prf.getEventPDF(bins)
                    g_pdf  = [ g_pdf[i]*dx[c]/xcen[c][i] for i in xrange(0,len(xcen[c])) ]
                    g_norm = sum(g_pdf)
                    if g_norm<1:
                        fillIdeogram(semiParamTrain, c, xof[c], truth, 1-g_norm)


                    randVal=rawVal*ROOT.TMath.Exp(prf.getRandom(-6,6))
                    fillIdeogram(semiParamTrain+'random',c, randVal, truth)
                if c=='e2':
                    bins   = [ ROOT.TVector2.Phi_mpi_pi(xcen[c][i]-rawVal) for i in xrange(0,len(xcen[c])) ]
                    g_pdf  = prf.getEventPDF(bins)
                    g_pdf  = [ g_pdf[i]*dx[c]      for i in xrange(0,len(xcen[c])) ]
                    g_norm = sum(g_pdf)                
                    g_pdf  = [ x/g_norm for x in g_pdf ]
                    randVal=ROOT.TVector2.Phi_mpi_pi(prf.getRandom(-ROOT.TMath.Pi(),ROOT.TMath.Pi())-rawVal)
                    fillIdeogram(semiParamTrain+'random',c, randVal, truth)

                for xbin in xrange(0,len(g_pdf)):
                    fillIdeogram(semiParamTrain,c, xcen[c][xbin], truth, g_pdf[xbin])                        
                    fillIdeogramResol(semiParamTrain,c,xcen[c][xbin],truth,data.trueh, g_pdf[xbin])
                
    return ideograms


def main():
    """wrapper to be used from command line"""

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--in' ,          dest='input',       help='Input directory [%default]',            default=None)    
    parser.add_option('-n', '--maxEvts' ,     dest='maxEvts',     help='Max. events to process [%default]',     default=-1, type=int)    
    parser.add_option('-o', '--out' ,         dest='output',      help='output dir [%default]',                 default='plots')
    (opt, args) = parser.parse_args()

    #start the chain of events to plot
    fList=os.path.join(opt.input,'predict/tree_association.txt')
    data=ROOT.TChain('data')
    tree=ROOT.TChain('tree')
    with open(fList,'r') as f:
        for x in f.read().split('\n'):
            try:
                dF,tF=x.split()
                data.AddFile(dF)
                tree.AddFile(tF)
            except:
                pass
    data.AddFriend(tree)

    #make the plots
    ideograms=makeIdeogramPlots(data,opt.maxEvts)

    #save to output
    outdir=os.path.dirname(opt.output)
    if len(outdir)>0: os.system('mkdir -p %s'%outdir)
    fOut=ROOT.TFile.Open('%s'%opt.output,'RECREATE')
    for key in ideograms: 
        ideograms[key].Write()
    fOut.Close()
    

if __name__ == "__main__":
    sys.exit(main())
