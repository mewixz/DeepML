import ROOT
import optparse
import os
import sys
import collections
from array import array


DEFAULT_NTHREADS=8
DEFAULT_VARLIST='nVertF,nJetF,rho,mindz,vx,vy,vz,'
for h in ['tkmet','npv_tkmet','closest_tkmet','puppimet','invpuppimet','gen']:
    DEFAULT_VARLIST += 'h{0}_pt,h{0}_phi,h{0}_n,'.format(h)
    DEFAULT_VARLIST += 'h{0}_scalar_ht,h{0}_scalar_sphericity,'.format(h)
    DEFAULT_VARLIST += 'h{0}_leadpt,h{0}_leadphi,'.format(h)
    DEFAULT_VARLIST += 'h{0}_thrust,h{0}_oblateness,h{0}_thrustMajor,h{0}_thrustMinor,h{0}_thrustTransverse,h{0}_thrustTransverseMinor,'.format(h)
    DEFAULT_VARLIST += 'h{0}_sphericity,h{0}_aplanarity,h{0}_C,h{0}_D,h{0}_detST,'.format(h)
    DEFAULT_VARLIST += 'h{0}_tau1,h{0}_tau2,h{0}_tau3,h{0}_tau4,h{0}_rho,'.format(h)
DEFAULT_VARLIST += 'visgenpt,visgenphi,vispt,visphi,'
DEFAULT_VARLIST += 'isGood,isW,isZ,lne1,e2'


def pruneTreeAndSaveTo(fileList,branchList,outFile):

    """prunes trees for selected branch list and saves to an outputfile"""

    #open and make friends of all trees needed
    tree=None
    fopenList=[]
    for fname,tname in fileList:
        fopenList.append( ROOT.TFile.Open(fname) )
        if tree:
            tree.AddFriend( fopenList[-1].Get(tname) )
        else:
            tree = fopenList[-1].Get(tname) 

    N=tree.GetEntries()
    print 'Saving',N,'events to',outFile

    #de-activate all branches except those of interest
    tree.SetBranchStatus("*",0)
    branchesToConvert=[]
    for b in branchList:
        tree.SetBranchStatus(b,1)
        
    #clone tree
    fopenList.append( ROOT.TFile(outFile,'RECREATE') )
    tree.CloneTree(N,'fast').Write()
        
    #close 
    for f in fopenList:
        f.Close()

def packedPruneTreeAndSaveTo(args):
    """wrapper for execution in a thread"""
    try:
        pruneTreeAndSaveTo(*args)
    except Exception as e:
        print 50*'<'
        print 'Problem with',args
        print e
    return False


def main():

    usage = 'usage: %prog main_dir:main_treename:main_pfix second_dir:second_treename:second_pfix ... [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-o',         dest='output',       help='output directory [%default]', type='string', default=None)
    parser.add_option('--tags',     dest='tags',         help='process only these tags [%default]', type='string', default=None)
    parser.add_option('--branches', dest='branchList',   help='do these branches [default is to long to print...]',  type='string', default=DEFAULT_VARLIST)
    (opt, args) = parser.parse_args()

    branchList=opt.branchList.split(',')

    if len(args)==0 or not opt.output or len(branchList)==0:
        parser.print_help()
        return -1

    #prepare output directory
    os.system('mkdir -p %s'%opt.output)

    #parse tags to process
    tags=[]
    if opt.tags:
        tags=opt.tags.split(',')

    #build list of files
    file_dict=collections.OrderedDict()
    for i in xrange(0,len(args)):
        indir,tname,pfix=args[i].split(':')

        #the first set is the main one
        if i==0:
            for f in os.listdir(indir):

                #filter this tag
                if len(tags):
                    found=False
                    for t in tags:
                        if not t in f: continue
                        found=True
                        break
                    if not found: continue
                    
                file_dict[f]=[(os.path.join(indir,f,pfix),tname)]
        else:
            for f in file_dict:
                friendFile=os.path.join(indir,pfix.format(f))
                if not os.path.isfile(friendFile): continue
                file_dict[f].append( (friendFile,tname) )

    #create tasks
    task_list=[]
    for f in file_dict:
        outFile=os.path.join(opt.output,f+'.root')
        if len(file_dict[f])!= len(args):
            print 'Not all trees were collected for',f,'leaving it aside'
        else:
            task_list.append( (file_dict[f],branchList,outFile) )

    print len(task_list),'tasks to process... this may take some time'

    #run in parallel
    import multiprocessing as MP
    pool = MP.Pool(DEFAULT_NTHREADS)
    pool.map(packedPruneTreeAndSaveTo,task_list)
    

if __name__ == "__main__":
    sys.exit(main())
