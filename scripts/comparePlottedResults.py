import ROOT

def getCMS(rm=0.12):
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.036)
    label.DrawLatex(rm,0.96,'#bf{CMS} #it{Simulation Preliminary}')
    label.DrawLatex(0.75,0.96,'W#rightarrowl#nu (13 TeV)')

mcTruth=None
models=[]
modelRatios=[]
modelmse=[]
colors=['#018571','#a6611a','#2b83ba','#fdae61','#c51b8a']

peakEstimates=[('plots/hpt_results_flatlne1_model_0.root','flat log(e_{1}), Huber','peak_3'),
               ('plots/hpt_results_flatlne1_model_1.root','flat log(e_{1}), asymm. Huber','peak_3'),
               ('plots/hpt_results_flatlne1_model_2.root','flat log(e_{1}), asymm. Huber + q','peak_3'),
               ('plots/hpt_results_flatlne1_model_3.root','flat log(e_{1}), Huber (wide DNN)','peak_3'),
               ('plots/hpt_results_flatlne1_model_50.root','flat log(e_{1}), semi-param','peak_3'),
               ('plots/hpt_results_isW_model_0.root','inc, Huber','peak_3'),
               ('plots/hpt_results_isW_model_1.root','inc, asymm. Huber','peak_3'),
               ('plots/hpt_results_isW_model_2.root','inc, asymm. Huber + q','peak_3'),
               ('plots/hpt_results_isW_model_3.root','inc, Huber (wide DNN)','peak_3'),
               ('plots/hpt_results_isW_model_50.root','inc, semi-param','peak_3')]

meanEstimates=[('plots/hpt_results_flatlne1_model_50.root','flat log(e_{1}), semi-param (mean)','mean_2'),
               ('plots/hpt_results_flatlne1_model_50.root','flat log(e_{1}), semi-param (#mu)','peak_3'),
               ('plots/hpt_results_isW_model_50.root','inc, semi-param (mean)','mean_2'),
               ('plots/hpt_results_isW_model_50.root','inc, semi-param (#mu)','peak_3')]
compSet=peakEstimates
compSet=meanEstimates
ncomp=len(compSet)/2
for url,title,hname in compSet:
    f=ROOT.TFile.Open(url)
    if not mcTruth:
        mcTruth=f.Get('MC truth_0').Clone('mctruth')
        mcTruth.SetTitle('Gen. level')
        mcTruth.SetDirectory(0)
        mcTruth.SetLineColor(1)
        mcTruth.SetLineWidth(3)

    n=len(models)
    models.append( f.Get(hname).Clone(title) )
    models[-1].SetTitle(title)
    models[-1].SetDirectory(0)
    models[-1].SetLineStyle(1 if n<ncomp else 7)
    models[-1].SetLineWidth(2)
    models[-1].SetLineColor(ROOT.TColor.GetColor(colors[n%ncomp]))
    modelRatios.append( models[-1].Clone(models[-1].GetName()+'_ratio') )
    modelRatios[-1].Divide(mcTruth)
    modelRatios[-1].SetDirectory(0)
    mse=0
    for xbin in xrange(1,models[-1].GetNbinsX()+1):
        mse += 0.5*((models[-1].GetBinContent(xbin)-mcTruth.GetBinContent(xbin))**2)
    mse /= models[-1].GetNbinsX()
    modelmse.append((title,mse))
    

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
c=ROOT.TCanvas('c','c',500,500)
c.SetTopMargin(0.05)
c.SetRightMargin(0.05)
c.SetLeftMargin(0.12)
c.SetBottomMargin(0.1)
c.SetLogy()
mcTruth.Draw()
mcTruth.GetXaxis().SetTitle('W p_{T} [GeV]')
mcTruth.GetYaxis().SetTitle('Events (a.u.)')
mcTruth.GetYaxis().SetRangeUser(1e3,1e6)
for h in models: h.Draw('histsame')
leg=c.BuildLegend(0.5,0.9,0.9,0.5)
leg.SetBorderSize(0)
leg.SetTextFont(42)
getCMS()
c.Modified()
c.Update()
c.SaveAs('wpt.png')

c.Clear()
c.SetLogy(False)
drawOpt='hist'
for h in modelRatios:
    h.Draw(drawOpt)
    h.GetYaxis().SetRangeUser(0,2)
    h.GetYaxis().SetTitle('Ratio to gen. level')
    h.GetXaxis().SetTitle('W p_{T} [GeV]')
    drawOpt='histsame'
leg=c.BuildLegend(0.3,0.94,0.7,0.65)
leg.SetBorderSize(0)
leg.SetTextFont(42)
c.SetGridy()
getCMS()
c.Modified()
c.Update()
c.SaveAs('wpt_ratio.png')

mse=ROOT.TH1F('mse',';;Binned MSE',len(modelmse),0,len(modelmse))
for xbin in xrange(0,len(modelmse)):
    mse.GetXaxis().SetBinLabel(xbin+1,modelmse[xbin][0])
    mse.SetBinContent(xbin+1,modelmse[xbin][1])
mse.Draw('hbar')
mse.SetFillColor(ROOT.TColor.GetColor(colors[0]))
c.SetLeftMargin(0.3)
getCMS(0.3)
c.Modified()
c.Update()
c.SaveAs('wpt_mse.png')
