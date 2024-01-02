import ROOT
import sys

tag=sys.argv[1]
var=sys.argv[2]

tree=ROOT.TChain('tree')
vtree=ROOT.TChain('tree')
baseDir='/eos/cms/store/cmst3/user/psilva/Wmass/RecoilTest_regress-models/'

with open('%s/%s/predict/predict/tree_association.txt'%(baseDir,tag)) as fileList:
    for line in fileList:
        orig,pred=line.strip().split(' ')
        tree.AddFile(orig)
        vtree.AddFile(pred)
        break
tree.AddFriend(vtree)

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
c=ROOT.TCanvas('c','c',500,500)
c.SetTopMargin(0.05)
c.SetLeftMargin(0.12)
c.SetRightMargin(0.12)
c.SetBottomMargin(0.1)
c.SetLogx()

hdef='(50,0,100,50,0,1)'

tree.Draw('htkmet_scalar_sphericity:htkmet_pt >> h%s'%hdef,'(isW!=0)','colz')
hraw=c.GetPrimitive('h').Clone('hraw')
hraw.SetDirectory(0)

rt='({1}+0.5*({2}-({1}))*(TMath::Sin({0})+1.0))'
if var=='lne1':
    spParams=[('mu_%s'%var,-3,3,'#mu'),
              ('sigma_%s'%var,1e-3,5,'#sigma'),         
              ('sigmaL_%s'%var,1e-3,10,'#sigma_{L}'),
              ('sigmaR_%s'%var,1e-3,10,'#sigma_{R}'),
              ('fL_%s'%var,1e-6,1,'f_{L}'),
              ('fR_%s'%var,1e-6,1,'f_{R}')
              ]
else:
    spParams=[('mu_%s'%var,-0.5*ROOT.TMath.Pi(),0.5*ROOT.TMath.Pi(),'#mu'),
              ('sigma_%s'%var,1e-3,ROOT.TMath.Pi(),'#sigma'),         
              ('sigmaL_%s'%var,1e-3,1e4,'#sigma_{L}'),
              ('fL_%s'%var,1e-6,1.0,'f_{L}'),
              ('sigmaR_%s'%var,1e-3,1e4,'#sigma_{R}'),
              ('fR_%s'%var,1e-6,1.0,'f_{R}')
              ]

for p,pmin,pmax,title in spParams:
    if not hasattr(tree,p) : continue
    w=rt.format(p,pmin,pmax)
    tree.Draw('htkmet_scalar_sphericity:htkmet_pt >> h%s'%hdef,'(isW!=0)*%s'%w,'colz')
    h=c.GetPrimitive('h')
    h.Divide(hraw)
    h.Draw('colz')
    h.GetYaxis().SetTitle('S')
    h.GetXaxis().SetTitle('h_{tk} [GeV]')    
    h.GetXaxis().SetMoreLogLabels()
    h.GetZaxis().SetTitleOffset(1.0)
    h.GetYaxis().SetTitleOffset(1.1)    
    h.GetZaxis().SetRangeUser(pmin,pmax)
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.04)
    label.DrawLatex(0.12,0.96,'#bf{CMS} #it{simulation Preliminary}') 
    label.DrawLatex(0.15,0.85,'#bf{#LT %s #GT}'%title)
    label.DrawLatex(0.75,0.96,'(13 TeV)')
    c.Modified()
    c.Update()    
    for ext in ['pdf','png']:
        c.SaveAs('sparam_%s_%s_prof.%s'%(var,p,ext))
