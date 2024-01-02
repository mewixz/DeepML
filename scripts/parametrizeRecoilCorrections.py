import ROOT
import sys
import optparse
import numpy as np
from array import array

CONDVAR='htkmet_pt'
PARTITLES={'mu':'#mu', 
           'sigma':'#sigma','sigmaL':'#sigma_{L}','sigmaR':'#sigma_{R}','sigmaT':'#Sigma',
           'aL':'a_{L}','aR':'a_{R}','fL':'f_{L}','fR':'f_{R}','f':'f',    
           'off':'offset',
           'lne1':'log(e_{1})=log(h_{true}/h_{tk})',
           'e2':'e_{2}=#phi_{true}-#phi_{tk}',
           'chi2':'#chi^{2}/dof',
           'pval':'p-val',
           'count':'N',
           'htkmet_scalar_sphericity':'S',
           'htkmet_pt':'h_{tk}'}

def showFitResults(var,data,pdf,title,name='fitresults',nbins=50,rangeX=[-6,6],extraText=[],nfreepar=0,pext=['png','pdf'],logy=False):

    """Plots the data and the PDF fit in a plot, together with a pad with the ratio of the two"""

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    c=ROOT.TCanvas('c','c',500,500)
    c.SetTopMargin(0)
    c.SetLeftMargin(0)
    c.SetRightMargin(0)
    c.SetBottomMargin(0)
    p1 = ROOT.TPad('p1','p1',0.0,0.25,1.0,1.0)
    p1.SetRightMargin(0.05)
    p1.SetLeftMargin(0.15)
    p1.SetTopMargin(0.08)
    p1.SetBottomMargin(0.02)
    p1.SetLogy(logy)
    p1.Draw()
    c.cd()
    p2 = ROOT.TPad('p2','p2',0.0,0.0,1.0,0.25)
    p2.SetBottomMargin(0.45)
    p2.SetRightMargin(0.05)
    p2.SetLeftMargin(0.15)
    p2.SetTopMargin(0.001)
    p2.SetGridy(True)
    p2.Draw()

    p1.cd()
    frame=var.frame(ROOT.RooFit.Range(rangeX[0],rangeX[1]),ROOT.RooFit.Bins(nbins))
    data.plotOn(frame,ROOT.RooFit.Name('datafit'),ROOT.RooFit.DrawOption('p'))

    pdf.plotOn(frame,               
               ROOT.RooFit.Name('pdf'),
               ROOT.RooFit.ProjWData(data),
               ROOT.RooFit.MoveToBack())
    chisq_ndof=frame.chiSquare(0)
    pval=ROOT.TMath.Prob(chisq_ndof*(nbins-nfreepar),nbins-nfreepar)
    frame.Draw()
    if logy:
        frame.GetYaxis().SetRangeUser(1,frame.GetMaximum()*3)
    else:
        frame.GetYaxis().SetRangeUser(0,frame.GetMaximum()*1.2)
    frame.GetYaxis().SetTitle("Events")
    frame.GetYaxis().SetTitleOffset(0.8)
    frame.GetYaxis().SetTitleSize(0.07)
    frame.GetYaxis().SetLabelSize(0.05)
    frame.GetXaxis().SetTitleSize(0.)
    frame.GetXaxis().SetLabelSize(0.)
    frame.GetXaxis().SetTitleOffset(1.5)
    frame.GetXaxis().SetTitle(var.GetTitle())
    
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.045)
    label.DrawLatex(0.18,0.85,'#scale[1.2]{#bf{CMS} #it{simulation Preliminary}}') 
    label.DrawLatex(0.85,0.95,'(13 TeV)')
    label.DrawLatex(0.18,0.8,'%s'%title)
    label.DrawLatex(0.18,0.75,'#chi^{2}/dof = %3.1f'%chisq_ndof)
    label.DrawLatex(0.18,0.70,'p-val = %3.1f'%pval)
    for i in xrange(0,len(extraText)):
        label.DrawLatex(0.7,0.8-0.05*i,extraText[i])
    p1.RedrawAxis()

    #show pull
    p2.cd()
    hpull = frame.pullHist('datafit','pdf')
    pullFrame=var.frame(ROOT.RooFit.Range(rangeX[0],rangeX[1]),ROOT.RooFit.Bins(nbins))
    pullFrame.addPlotable(hpull,"P") ;
    pullFrame.Draw()
    pullFrame.GetYaxis().SetTitle("#frac{Data-Fit}{Unc.}")
    pullFrame.GetYaxis().SetTitleSize(0.20)
    pullFrame.GetYaxis().SetLabelSize(0.16)
    pullFrame.GetYaxis().SetTitleOffset(0.27)
    pullFrame.GetXaxis().SetTitleSize(0.20)
    pullFrame.GetXaxis().SetLabelSize(0.16)
    pullFrame.GetXaxis().SetTitleOffset(0.85)
    pullFrame.GetYaxis().SetNdivisions(4)
    pullFrame.GetYaxis().SetRangeUser(-8.0,8.0)
    p2.RedrawAxis()
        
    c.cd()
    c.Modified()
    c.Update()
    for ext in pext:
        c.SaveAs(name+'.'+ext)


    return pval,chisq_ndof


def getPDFQuantiles(var,pdf,xmin,xmax,step):
    """get PDF quantiles"""

    cdf = pdf.createCdf(ROOT.RooArgSet(var))
    dq=[]
    xq=[]
    for x in np.arange(xmin,xmax,step):
        var.setVal(x)
        cdfVal=cdf.getVal()
        
        if len(dq)==0:
            dq=[abs(cdfVal-0.16),abs(cdfVal-0.5),abs(cdfVal-0.84)]
            xq=[x,x,x]
        else:
            if abs(cdfVal-0.16)<dq[0]:
                dq[0]=abs(cdfVal-0.16)
                xq[0]=x
            if abs(cdfVal-0.5)<dq[1]:
                dq[1]=abs(cdfVal-0.5)
                xq[1]=x
            if abs(cdfVal-0.84)<dq[2]:
                dq[2]=abs(cdfVal-0.84)
                xq[2]=x

    return xq



def getCondVarQuantiles(tree,q=np.arange(10,100,10),maxEntries=50000):
    """compute the quantiles for the condition variables"""
    vals=[]
    for i in xrange(0,min(tree.GetEntriesFast(),maxEntries)):
        tree.GetEntry(i)
        if tree.isGood==0: continue
        if tree.isW==0: continue
        vals.append( [getattr(tree,CONDVAR)] )
    return np.percentile(vals,q,axis=0)

def defineSlices(condVarQ):

    """define the slices"""

    nq=len(condVarQ)    
    cuts=[]
    for i in xrange(0,nq+1):
        if i==0:
            cut_i='{0}<{1}'.format(CONDVAR,condVarQ[i][0])
        elif i==nq:
            cut_i='{0}>={1}'.format(CONDVAR,condVarQ[i-1][0])
        else:
            cut_i='{0}>={1} && {0}<{2}'.format(CONDVAR,condVarQ[i-1][0],condVarQ[i][0])            
        cuts.append( [i+1,cut_i] )

    return cuts

def runParametrizations(opt):
    """slices the simulated events and parametrizes the recoil corrections (scale or direction)"""

    #import data to fit
    tree=ROOT.TChain('tree')
    for f in opt.input.split(','):
        tree.AddFile(f)

    print 'Determining quantiles'
    if opt.incFit:
        condVarQ=getCondVarQuantiles(tree,q=50)
        cuts=[ [condVarQ[0],''] ]
    else:
        condVarQ=getCondVarQuantiles(tree,q=np.arange(10,100,10))
        cuts=defineSlices(condVarQ)

    #graphs to return
    grEvol={}
    meanEvol=ROOT.TGraph()
    meanEvol.SetName('meanevol')
    sigmaEvol=ROOT.TGraph()
    sigmaEvol.SetName('sigmaevol')
    def getEvolGr(p):
        if not p in grEvol:
            x=[0]+[q for q in condVarQ[:,0] if q<50]+[100.0]
            grEvol[p]=ROOT.TH1F('%s_prof'%p,
                                '%s;%s'%(PARTITLES[CONDVAR],PARTITLES[p]),
                                len(x)-1,array('d',x))
            grEvol[p].Sumw2()
            grEvol[p].SetDirectory(0)
        return grEvol[p]


    w=ROOT.RooWorkspace('w')
    w.factory('lne1[-10,10]')
    w.var('lne1').SetTitle(PARTITLES['lne1'])
    w.factory('e2[-3.15,3.15]')
    w.var('e2').SetTitle(PARTITLES['e2'])
    w.factory('isW[0,1]')
    w.factory('%s[0,-1e12,1e12]'%CONDVAR)
    args=ROOT.RooArgSet(w.var('lne1'),w.var('e2'),w.var('isW'))
    presel='isW>0'
    args.add(w.var(CONDVAR))
    print 'Importing dataset to workspace'
    data=ROOT.RooDataSet('data','data',tree,args,presel)
    getattr(w,'import')(data)
    
    print 'Defining the PDFs'   
    mu=data.mean(w.var(opt.regTarget))
    sigma=data.sigma(w.var(opt.regTarget))    
    if opt.regTarget=='lne1':
        w.factory('mu[%f,%f,%f]'%(mu,mu-2*sigma,mu+2*sigma))
        w.factory('sigma[%f,%f]'%(sigma*0.1,sigma*2))
    else:
        w.factory('mu[%f,%f,%f]'%(mu,-ROOT.TMath.Pi(),ROOT.TMath.Pi()))        
        w.factory('sigma[0.01,%f]'%(ROOT.TMath.Pi()*0.5))
    if opt.pdf=='gd':
        w.factory("expr::t('(@0-@1)/@2',%s,mu,sigma)"%opt.regTarget)
        if opt.regTarget=='lne1':
            w.factory('aL[1.0,1,10]')
            w.factory('aR[1.0,1,10]')
        else:
            w.factory('aL[1.0,0.1,3.1415]')
            w.factory('aR[1.0,0.1,3.1415]')
        w.factory("expr::exp_L('@0<-@1 ? exp(@1*(@0+@1/2.)): 0.',t,aL)")
        w.factory("expr::exp_R('@0>@1 ? exp(-@1*(@0-@1/2.)): 0.',t,aR)")
        w.factory("expr::exp_0('@0>-@1 && @0<@2  ? exp(-@0*@0/2): 0.',t,aL,aR)")
        w.factory("EXPR::pdf('@0+@1+@2',exp_L,exp_0,exp_R)") 
    elif opt.pdf=='gdoffset':
        w.factory('off[0.0,1.0]')
        if opt.regTarget=='lne1':
            w.factory('aL[1.0,1,10]')
            w.factory('aR[1.0,1,10]')
        else:
            w.factory('aL[1.0,0.1,3.1415]')
            w.factory('aR[1.0,0.1,3.1415]')
        w.factory("expr::t('(@0-@1)/@2',%s,mu,sigma)"%opt.regTarget)
        w.factory("expr::exp_L('@0<-@1 ? exp(@1*(@0+@1/2.))+@2: 0.',t,aL,off)")
        w.factory("expr::exp_R('@0>@1 ? exp(-@1*(@0-@1/2.))+@2: 0.',t,aR,off)")
        w.factory("expr::exp_0('@0>-@1 && @0<@2  ? exp(-@0*@0/2)+@3 : 0.',t,aL,aR,off)")
        w.factory("EXPR::pdf('@0+@1+@2',exp_L,exp_0,exp_R)") 
    elif opt.pdf=='gsum':
        if opt.regTarget=='lne1':
            w.factory('sigmaT[%f,%f,%f]'%(2*sigma,sigma,sigma*10))
        else:
            w.factory('sigmaT[%f,%f]'%(0.25*ROOT.TMath.Pi(),1e5))
        w.factory('f[0,1]')
        w.factory("SUM::pdf( f*Gaussian::g1({0},mu,sigmaT),Gaussian::g0({0},mu,sigma) )".format(opt.regTarget))
    elif opt.pdf=='bgsum':
        if opt.regTarget=='lne1':
            w.factory('sigmaL[%f,%f,%f]'%(2*sigma,sigma,sigma*10))
            w.factory('sigmaR[%f,%f,%f]'%(2*sigma,sigma,sigma*10))        
        else:
            w.factory('sigmaL[%f,%f]'%(0.25*ROOT.TMath.Pi(),1e5))
            w.factory('sigmaR[%f,%f]'%(0.25*ROOT.TMath.Pi(),1e5))
        w.factory('fL[0,1]')
        w.factory('fR[0,1]')
        w.factory("expr::exp_L('@0<@1  ? exp(-0.5*pow((@0-@1)/@2,2))+@4*exp(-0.5*pow((@0-@1)/@3,2))                   : 0.',%s,mu,sigma,sigmaL,fL)"%opt.regTarget)
        w.factory("expr::exp_R('@0>=@1 ? (exp(-0.5*pow((@0-@1)/@2,2))+@4*exp(-0.5*pow((@0-@1)/@3,2)))*((1+@5)/(1+@4)) : 0.',%s,mu,sigma,sigmaR,fR,fL)"%opt.regTarget)
        w.factory("EXPR::pdf('@0+@1',exp_L,exp_R)")
  
    regTarget=opt.regTarget
    for i in xrange(0,len(cuts)):

        data=w.data('data')        

        #project the data and do the fit
        ibinx,cut=cuts[i]
        if len(cut)>0:
            data=data.reduce(cut)

        #check number of entries
        nEntries=data.sumEntries()
        if nEntries<100:
            print 'Too few entries for',cut
            continue

        w.var('mu').setVal( data.mean(w.var(opt.regTarget)) )
        w.var('sigma').setVal( data.sigma(w.var(opt.regTarget)) )    
        w.pdf('pdf').fitTo( data )

        rangeX=[-6,6] if opt.regTarget=='lne1' else [-3.16,3.16]
        pdfQ=getPDFQuantiles(w.var(opt.regTarget),
                             w.pdf('pdf'),
                             rangeX[0],rangeX[1],0.01)
        
        #parameters fit
        extraText=[]
        iter = w.pdf('pdf').getParameters(data).createIterator()
        iparam = iter.Next()
        while iparam :
            if not iparam.getAttribute('Constant') and not opt.incFit:
                gr=getEvolGr(iparam.GetName())
                gr.SetBinContent(ibinx,iparam.getVal())
                gr.SetBinError(ibinx,iparam.getError())
                extraText.append('#scale[0.6]{%s=%3.2f#pm%3.2f}'%(PARTITLES[iparam.GetName()],
                                                                  iparam.getVal(),
                                                                  iparam.getError()))
            iparam = iter.Next()        
                

        print pdfQ
        meanEvol.SetPoint(meanEvol.GetN(),gr.GetXaxis().GetBinCenter(ibinx),pdfQ[1])
        sigmaEvol.SetPoint(sigmaEvol.GetN(),gr.GetXaxis().GetBinCenter(ibinx),0.5*(pdfQ[2]-pdfQ[0]))

        #show the plot for control
        if opt.incFit:
            title='inclusive'
        else:
            title='%s=%3.2f'%(PARTITLES[CONDVAR],gr.GetXaxis().GetBinCenter(ibinx))
        name=regTarget if opt.incFit else '%s_fit%d'%(regTarget,i)
        nfreepar=4 if opt.regTarget=='lne1' else 5
        pval,chi2=showFitResults(var=w.var(opt.regTarget),
                                 data=data,
                                 pdf=w.pdf('pdf'),
                                 title=title,
                                 name=name,
                                 nbins=50,
                                 rangeX=rangeX,
                                 extraText=extraText,
                                 nfreepar=nfreepar,
                                 logy=True if opt.regTarget=='lne1' else False)

        if not opt.incFit:
            for val,valName in [(pval,'pval'),(chi2,'chi2'),(nEntries,'count')]:
                gr=getEvolGr(valName)
                gr.SetBinContent(ibinx,val)

    #return the summary graphs with the evolution of the parameters
    return grEvol,meanEvol,sigmaEvol

def plotParameterizations(opt):
    """opens the results of the fits and makes some plots"""
    fIn=ROOT.TFile.Open('recoilparam_%s_%s.root'%(opt.regTarget,opt.pdf))


    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    c=ROOT.TCanvas('c','c',500,500)
    c.SetTopMargin(0.05)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.12)
    c.SetBottomMargin(0.1)
    c.SetLogx()
    chi2H,pvalH,countH=None,None,None
    for x in fIn.GetListOfKeys():
        gr=x.ReadObj()
        gr.Draw('colz')
        xt,yt,t=gr.GetTitle(),gr.GetXaxis().GetTitle(),gr.GetYaxis().GetTitle()
        gr.GetXaxis().SetTitle(xt)
        gr.GetXaxis().SetMoreLogLabels()
        gr.GetYaxis().SetTitle(yt)
        gr.SetTitle(t)
        
        if opt.regTarget=='lne1' and gr.GetName()=='mu_prof':     gr.GetZaxis().SetRangeUser(-2,2)
        if opt.regTarget=='lne1' and gr.GetName()=='sigma_prof':  gr.GetZaxis().SetRangeUser(1e-3,2)
        if opt.regTarget=='lne1' and gr.GetName()=='sigmaL_prof': gr.GetZaxis().SetRangeUser(1e-3,5)
        if opt.regTarget=='lne1' and gr.GetName()=='fL_prof':     gr.GetZaxis().SetRangeUser(0,1)
        if opt.regTarget=='lne1' and gr.GetName()=='sigmaR_prof': gr.GetZaxis().SetRangeUser(1e-3,5)
        if opt.regTarget=='lne1' and gr.GetName()=='fR_prof':     gr.GetZaxis().SetRangeUser(0,1)
        if opt.regTarget=='e2'   and gr.GetName()=='mu_prof':     gr.GetZaxis().SetRangeUser(-0.5*ROOT.TMath.Pi(),0.5*ROOT.TMath.Pi())
        if opt.regTarget=='e2'   and gr.GetName()=='sigma_prof':  gr.GetZaxis().SetRangeUser(1e-3,ROOT.TMath.Pi())
        if opt.regTarget=='e2'   and gr.GetName()=='sigmaL_prof': gr.GetZaxis().SetRangeUser(1e-3,1e4)
        if opt.regTarget=='e2'   and gr.GetName()=='fL_prof':     gr.GetZaxis().SetRangeUser(1e-6,1)
        if opt.regTarget=='e2'   and gr.GetName()=='sigmaR_prof': gr.GetZaxis().SetRangeUser(1e-3,1e4)
        if opt.regTarget=='e2'   and gr.GetName()=='fR_prof':     gr.GetZaxis().SetRangeUser(1e-6,1)

        label = ROOT.TLatex()
        label.SetNDC()
        label.SetTextFont(42)
        label.SetTextSize(0.04)
        label.DrawLatex(0.12,0.96,'#bf{CMS} #it{simulation Preliminary}') 
        label.DrawLatex(0.15,0.85,'#bf{%s}'%gr.GetTitle())
        label.DrawLatex(0.75,0.96,'(13 TeV)')
        c.Modified()
        c.Update()
        for ext in ['pdf','png']:
            c.SaveAs('{0}_{1}.{2}'.format(opt.regTarget,gr.GetName(),ext))

        if gr.GetName()=='count_prof': countH=gr.Clone()
        if gr.GetName()=='chi2_prof':  chi2H=gr.Clone()
        if gr.GetName()=='pval_prof':  pvalH=gr.Clone()


    if countH:
        count,P,Xi2=0,0,0
        for xbin in xrange(1,countH.GetNbinsX()+1):
            for ybin in xrange(1,countH.GetNbinsY()+1):
                count += countH.GetBinContent(xbin,ybin)
                if chi2H : Xi2 += chi2H.GetBinContent(xbin,ybin)*countH.GetBinContent(xbin,ybin)
                if pvalH : P   += pvalH.GetBinContent(xbin,ybin)*countH.GetBinContent(xbin,ybin)
        print '-'*20
        print '<chi2>=%3.1f'%(Xi2/count)
        print '<p-val>=%3.2f'%(P/count)
        print '-'*20

    fIn.Close()

def main():

    #readout the configuration
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--in',    
                      dest='input', 
                      help='input file [%default]',
                      default='/eos/cms/store/cmst3/user/psilva/Wmass/Recoil_regress-data/WJetsToLNu_part1.root',
                      type='string')
    parser.add_option('-r', '--regTarget',    
                      dest='regTarget',
                      help='regression target [%default]',
                      default='lne1',
                      type='string')
    parser.add_option('-p', '--pdf',
                      dest='pdf',
                      help='pdf name [%default]',
                      default='bgsum',
                      type='string')
    parser.add_option('--incFit',
                      dest='incFit',
                      help='inclusive fit only [%default]',
                      default=False,
                      action='store_true')
    parser.add_option('--show',
                      dest='show',
                      help='show results of the fits [%default]',
                      default=False,
                      action='store_true')
    (opt, args) = parser.parse_args()

    ROOT.gROOT.SetBatch(True)

    if not opt.show:
        grEvol,meanEvol,sigmaEvol=runParametrizations(opt)
        fOut=ROOT.TFile.Open('recoilparam_%s_%s.root'%(opt.regTarget,opt.pdf),'RECREATE')
        meanEvol.Write()
        sigmaEvol.Write()
        for key in grEvol:
            grEvol[key].SetDirectory(fOut)
            grEvol[key].Write()
        fOut.Close()
    else :
        plotParameterizations(opt)

if __name__ == "__main__":
    sys.exit(main())
