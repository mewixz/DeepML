import ROOT
import os,sys
import optparse
from array import array

COLORS       = ['#000000','#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f']
MARKERS      = [1,        20,       21,       22,       23,       24,       25       ]

BASE_IDEOGRAMLIST = {'lne1':[('peak','ideograms0_lne1_e2.root'),
                             ('peak+quantiles','ideograms100_lne1_e2.root'),
                             ('semi-param','ideograms300_lne1_e2.root'),
                             ]
                     }
BASE_IDEOGRAMLIST['e2']=BASE_IDEOGRAMLIST['lne1']

SP_IDEOGRAMLIST = {'lne1':[('%d'%i,'ideograms%d_lne1_e2.root'%i) for i in range(300,321)]}
SP_IDEOGRAMLIST['e2']=SP_IDEOGRAMLIST['lne1']

def getCMSHeader(rm=0.15):
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.036)
    label.DrawLatex(rm,0.96,'#bf{CMS} #it{Simulation Preliminary}')
    label.DrawLatex(0.75,0.96,'W#rightarrowl#nu (13 TeV)')

def fixExtremities(h,addOverflow=True,addUnderflow=True):
    if h.InheritsFrom('TH2') : return
    if addUnderflow :
        fbin  = h.GetBinContent(0) + h.GetBinContent(1)
        fbine = ROOT.TMath.Sqrt(h.GetBinError(0)*h.GetBinError(0) + h.GetBinError(1)*h.GetBinError(1))
        h.SetBinContent(1,fbin)
        h.SetBinError(1,fbine)
        h.SetBinContent(0,0)
        h.SetBinError(0,0)
    if addOverflow:
        nbins = h.GetNbinsX();
        fbin  = h.GetBinContent(nbins) + h.GetBinContent(nbins+1)
        fbine = ROOT.TMath.Sqrt(h.GetBinError(nbins)*h.GetBinError(nbins)  + h.GetBinError(nbins+1)*h.GetBinError(nbins+1))
        h.SetBinContent(nbins,fbin)
        h.SetBinError(nbins,fbine)
        h.SetBinContent(nbins+1,0)
        h.SetBinError(nbins+1,0)

def getHists(fname,var):
    fIn=ROOT.TFile.Open(fname)
    hists={}
    for key,h in [('truth',     'htrue_{0}'),
                  ('raw',      'hraw_{0}'),
                  ('peak',      'hpeak_{0}'),
                  ('gauss',     'hgauss_{0}'),
                  ('bgsum',     'hbgsum_{0}')]:
        
        obj=fIn.FindKey(h.format(var))
        try:
            hists[key]=obj.ReadObj().Clone(key)            
            hists[key].SetTitle(key if key !='raw' else '2.08xhtk')
            hists[key].SetDirectory(0)
            #fixExtremities(hists[key])
        except:
            continue

        if key=='truth' : continue
        hists[key+'_diff']=fIn.Get(h.format(var)+'_diff').Clone(key+'_diff')
        hists[key+'_diff'].SetTitle(key if key !='raw' else '2.08xhtk')
        hists[key+'_diff'].SetDirectory(0)

    for key in hists:
        hists[key].SetDirectory(0)
    fIn.Close()
    return hists


def showComparison(hists,fOut):

    c=ROOT.TCanvas('c','c',500,500)
    c.SetTopMargin(0.0)
    c.SetBottomMargin(0.0)
    c.SetLeftMargin(0.0)
    c.SetRightMargin(0.0)
    c.cd()
    p1=ROOT.TPad('p1','p1',0.0,0.2,1.0,1.0)
    p1.SetRightMargin(0.05)
    p1.SetLeftMargin(0.12)
    p1.SetTopMargin(0.06)
    p1.SetBottomMargin(0.01)
    p1.Draw()
    p1.cd()
    leg=ROOT.TLegend(0.6,0.9,0.9,0.9-0.04*len(hists))
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    for i in xrange(0,len(hists)):
        hists[i].Draw('hist' if i==0 else 'histsame')
        ci=ROOT.TColor.GetColor(COLORS[i])
        hists[i].GetYaxis().SetTitle('Events (a.u.)')
        hists[i].SetLineColor(ci)
        hists[i].SetLineWidth(2)        
        hists[i].SetMarkerStyle(MARKERS[i])
        hists[i].SetMarkerColor(ci)
        
        leg.AddEntry(hists[i],hists[i].GetTitle(),'lp')
    leg.Draw()
    getCMSHeader()

    c.cd()
    p2 = ROOT.TPad('p2','p2',0.0,0.0,1.0,0.2)
    p2.Draw()
    p2.SetBottomMargin(0.4)
    p2.SetRightMargin(0.05)
    p2.SetLeftMargin(0.12)
    p2.SetTopMargin(0.01)
    p2.SetGridx(False)
    p2.SetGridy(True)
    p2.Draw()
    p2.cd()

    ratioHists=[]
    ratioHists.append( hists[0].Clone('frame') )
    ratioHists[0].Reset('ICE')
    ratioHists[0].GetYaxis().SetNdivisions(5)
    ratioHists[0].GetYaxis().SetTitle('Ratio')
    ratioHists[0].GetYaxis().SetTitleOffset(0.3)
    ratioHists[0].GetYaxis().SetRangeUser(0.7,1.3)
    ratioHists[0].Draw()
    ratioHists[0].GetXaxis().SetTitleSize(0.15)
    ratioHists[0].GetYaxis().SetTitleSize(0.15)
    ratioHists[0].GetXaxis().SetLabelSize(0.15)
    ratioHists[0].GetYaxis().SetLabelSize(0.15)
    
    for i in xrange(1,len(hists)):
        ratioHists.append( hists[i].Clone('{0}_ratio_{1}'.format(hists[i].GetName(),hists[0].GetName())) )
        ratioHists[-1].Divide(hists[0])
        ratioHists[-1].Draw('e1same')

    c.cd()
    c.Modified()
    c.Update()
    for ext in ['png','pdf']:
        c.SaveAs('%s.%s'%(fOut,ext))

    c.Delete()


def getProfiles(h,step=1):
    """build the median and 68%CI profiles"""

    try:
        if h.GetEntries()==0 : return None,None
    except:
        return None,None    

    medianGr = ROOT.TGraphErrors()
    widthGr  = ROOT.TGraphErrors()

    totalPts,avgMedian,avgWidth=0,0,0
    xq = array('d', [0.16,0.5,0.84])
    yq = array('d', [0.0 ,0.0,0.0 ])

    totalPts=0
    for xbin in xrange(1,h.GetNbinsX()+1,step):
        tmp=h.ProjectionY('tmp',xbin,xbin+(step-1))
        tmp.GetQuantiles(3,yq,xq)

        npts = tmp.Integral()
        totalPts  += npts
        avgMedian += abs(yq[1])*npts
        avgWidth  += abs(yq[2]-yq[1])*npts

        xcen=h.GetXaxis().GetBinCenter(xbin)
        npts=medianGr.GetN()        
        medianGr.SetPoint(npts,xcen,yq[1])
        medianGr.SetPointError(npts,0,1.2533*tmp.GetMeanError())
        widthGr.SetPoint(npts,xcen,yq[2]-yq[1])
        widthGr.SetPointError(npts,0,tmp.GetRMSError())

    return medianGr,widthGr,avgMedian/totalPts,avgWidth/totalPts

def showGraph(gr,xtit,ytit,gridx,gridy,fout):
    c=ROOT.TCanvas('c','c',500,500)
    c.SetTopMargin(0.05)
    c.SetBottomMargin(0.1)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.05)
    c.SetGridx(gridx)
    c.SetGridy(gridy)
    gr.Draw('a')
    gr.GetXaxis().SetTitle(xtit) 
    gr.GetYaxis().SetTitle(ytit)
    ngr=gr.GetListOfGraphs().LastIndex()+1
    if ngr<5:
        leg=c.BuildLegend(0.6,0.9,0.9,0.9-0.04*ngr)
    else:
        ncol=2
        leg=c.BuildLegend(0.6,0.9,0.9,0.9-0.04*ngr/ncol)
        leg.SetNColumns(ncol)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    getCMSHeader()
    c.cd()
    c.Modified()
    c.Update()
    for ext in ['png','pdf']:
        c.SaveAs('%s.%s'%(fout,ext))
    c.Delete()

def analyzeBiasAndResol(hists):

    perfGr={}
    profiles=[ROOT.TMultiGraph(),ROOT.TMultiGraph(),ROOT.TMultiGraph()]    
    for k in xrange(0,len(hists)):
        h,color,marker=hists[k],COLORS[k+1],MARKERS[k+1]
        iprof=getProfiles(h)
        ci=ROOT.TColor.GetColor(color)
        for i in xrange(0,3):
            if i<2:
                iprof[i].SetTitle(h.GetTitle())
                iprof[i].SetLineColor(ci)
                iprof[i].SetMarkerColor(ci)
                iprof[i].SetMarkerStyle( marker )
                profiles[i].Add(iprof[i],'p')
            else:
                key=h.GetTitle()
                perfGr[key]=ROOT.TGraph()
                perfGr[key].SetTitle(h.GetTitle())
                perfGr[key].SetLineColor(ci)
                perfGr[key].SetMarkerColor(ci)
                perfGr[key].SetMarkerStyle( marker )
                perfGr[key].SetMarkerSize(1.5)
                perfGr[key].SetPoint(0,iprof[2],iprof[3])
                profiles[i].Add(perfGr[key].Clone(),'p')

    return profiles,perfGr

    
def main():
    """wrapper to be used from command line"""

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--in' ,   dest='input',  help='input dir [%default]',   default='./')
    parser.add_option('-l', '--list' ,   dest='list',  help='list of comparisons [%default]',   default='base')
    parser.add_option('-o', '--out' ,  dest='output', help='output dir [%default]',  default='plots')
    (opt, args) = parser.parse_args()

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gROOT.SetBatch(True)

    ideogramList=BASE_IDEOGRAMLIST
    if opt.list=='sparam' : ideogramList=SP_IDEOGRAMLIST
    

    garbageList=[]
    for var in ideogramList:

        unit ='[GeV]' if var=='lne1' else '[rad]'

        plist,pdflist=[],[]
        for m,f in ideogramList[var]:
            f = os.path.join(opt.input,f)
            if not os.path.isfile(f) : continue
            print 'Analysing',f,'for',var

            hists=getHists(f,var)
            profiles,perfGr=analyzeBiasAndResol([hists[x+'_diff'] for x in ['raw','peak','gauss','bgsum'] if x+'_diff' in hists])

            if len(plist)==0:
                plist.append(perfGr['2.08xhtk'].Clone('htk'))
                plist[-1].SetTitle('2.08xhtk')
            plist.append(perfGr['peak'].Clone(m))
            plist[-1].SetTitle(m)

            if 'bgsum' in perfGr:
                pdflist.append(perfGr['bgsum'].Clone(m+'pdf'))
                pdflist[-1].SetTitle(m)
            elif 'gauss' in perfGr:
                pdflist.append(perfGr['gauss'].Clone(m+'pdf'))
                pdflist[-1].SetTitle(m)

            #show plots
            tag=m.replace(' ','_')
            xtit=hists['truth'].GetXaxis().GetTitle()
            showComparison([hists[x] for x in ['truth','raw','peak','gauss','bgsum'] if x in hists],'%s/comp_%s_%s'%(opt.output,var,tag))
            showGraph(profiles[0],xtit,'Bias',False,True,'%s/comp_%s_%s_bias'%(opt.output,var,tag))
            showGraph(profiles[1],xtit,'Resolution',False,True,'%s/comp_%s_%s_resol'%(opt.output,var,tag))
            showGraph(profiles[2], 'Average |bias| ' + unit,'Average resolution ' + unit,True,True,'%s/comp_%s_%s_benchmark'%(opt.output,var,tag))

            garbageList += profiles
            for p in perfGr: garbageList.append(perfGr[p])

        garbageList += plist
        garbageList += pdflist



        #benchmark (peak-based)
        pbenchmark=ROOT.TMultiGraph()
        for i in xrange(0,len(plist)):
            cidx=i/(len(COLORS)-1)+1
            if i==0 : cidx=0
            ci=ROOT.TColor.GetColor( COLORS[cidx] )
            midx=i%(len(COLORS)-1)+1
            marker=MARKERS[midx]
            plist[i].SetLineColor(ci)
            plist[i].SetMarkerColor(ci)
            plist[i].SetMarkerStyle( marker )
            pbenchmark.Add(plist[i],'p')
        showGraph(pbenchmark,'Average |bias| ' + unit,'Average resolution ' + unit,True,True,'%s/peak_benchmark_%s_%s'%(opt.output,var,opt.list))

        #benchmark (peak-based)
        benchmark=ROOT.TMultiGraph()
        for i in xrange(0,len(pdflist)):
            cidx=i/(len(COLORS)-1)+1
            ci=ROOT.TColor.GetColor( COLORS[cidx] )
            midx=i%(len(COLORS)-1)+1
            marker=MARKERS[midx]
            pdflist[i].SetLineColor(ci)
            pdflist[i].SetMarkerColor(ci)
            pdflist[i].SetMarkerStyle( marker )
            benchmark.Add(pdflist[i],'p')
        showGraph(benchmark,'Average |bias| ' + unit,'Average resolution ' + unit,True,True,'%s/pdf_benchmark_%s_%s'%(opt.output,var,opt.list))

    #for p in garbageList: p.Delete()

        
if __name__ == "__main__":
    sys.exit(main())
