import optparse
import os
import sys
from modules.QuantileRegressor import *
from modules.QuantileCorrector import *

def main():

   usage = 'usage: %prog [options]'
   parser = optparse.OptionParser(usage)
   parser.add_option('-r', '--run',  dest='run',             help='train/predict [%d]',     default='train')
   parser.add_option('-t',           dest='trainedInputs',   help='trained inputs [%d]',    default=None)
   parser.add_option('-i',           dest='input',           help='input file [%d]',        default=None)
   parser.add_option('-l',           dest='label',           help='label [%d]',             default='recoilqr')
   (opt, args) = parser.parse_args()

   #prepare output
   os.system('mkdir -p %s'%opt.label)

   qr=QuantileRegressor(opt.label)

   #train regressor
   if opt.run=='train':
      qr.loadDF()
      qr.runTrainQuantilesLoop(qList=np.arange(0.1,1.0,0.2),
                               usePrevVarAsFeature=True)
      
   #run prediction
   if opt.run=='predict':

      #read the regressors
      if not opt.trainedInputs:
         print 'Need to provide directory with trained regressor (-t dir)'
         sys.exit(-1)
      regList,featureList=qr.readQuantileRegressors(opt.trainedInputs)
      
      #apply the corrections
      if not opt.input:
         print 'Need to provide input file (-i file)'
         sys.exit(-1)

      #convert the ROOT tree to a pandas DataFrame
      df = rpd.read_root(opt.input,qr.tree,columns=qr.branches)
      print '%d events available in %s'%(len(df.index),opt.input)

      #run corrections
      qc=QuantileCorrector(opt.label,df,regList,featureList)
      qc.saveCorrections(os.path.basename(opt.input))


if __name__ == "__main__":
    sys.exit(main())
