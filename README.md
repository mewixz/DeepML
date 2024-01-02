# DeepML

Machine learning based on DeepJetCore 

## Installation and environment

We are running within CMSSW (>10_2_X should have the keras, tensorflow, etc. libraries needed).
To install the software you can use the following
```
cmsrel CMSSW_10_2_0_pre5
cd CMSSW_10_2_0_pre5/src
cmsenv
git clone https://github.com/DL4Jets/DeepJetCore.git
cd DeepJetCore/compiled/
make -j 4
cd -
git clone https://gitlab.cern.ch/psilva/DeepML.git
cd DeepML
```
Once all is compiled see below some examples on how to run the regression.

## Preparing a flat tree as input

In case the variables are scattered accross different TTrees which are typically bound as friends in the analysis,
DeepJetCore (in particular root_numpy) prefers that a single tree is given as input to the training. 
In scripts/prepareTrainingTreeFromHeppy.py you can find a script which will gather the data in branches from different trees
and dump a summary tree which can be used as input for training. It may take some time to run locally if there are many events.
An example of how to run it is the following

```
python scripts/prepareTrainingTreeFromHeppy.py --tags WJets -o /eos/cms/store/cmst3/user/psilva/Wmass/Recoil_regress-data \
        /eos/cms/store/cmst3/group/wmass/w-helicity-13TeV/ntuplesRecoil/TREES_SIGNAL_1l_recoil_fullTrees/:tree:treeProducerWMass/tree.root 
        /eos/cms/store/cmst3/user/psilva/Wmass/FriendTREES_SIGNAL_1l_recoil_fullTrees/:Friends:tree_Friend_{0}.root
```

The arguments are the input directories, the name of the tree and a postfix to access the ROOT file. 
The first tree is assumed to be the main one, the subsequent are helper friend trees. The {0} can be used to substitute
the original name of the tree found in the first directory.

## Recoil regression

Use the runRecoilScaleRegression.sh found under scripts. 
Running the script with -h will show all the available options.
A simple example is given below (notice it uses only part of the total number of simulated events)
```
sh scripts/runRecoilRegression.sh -r prepare,train,predict -t WJetsToLNu_part -c isW==0 -m 0:lne1 \
   -i /eos/cms/store/cmst3/user/psilva/Wmass/Recoil_regress-data 
```
If a model exists and you want to extend the predicition to other files than the ones used in the training
you can use the same script but running in predict mode
```
sh scripts/runRecoilRegression.sh -r predict -m 0 -i /eos/cms/store/cmst3/user/psilva/Wmass/RecoilTest_regress-v2/Chunks -p regress-results/train
```
Under the test directory you can also find a condor submission script to send out with HTCondor several trainings in parallel.
The usual `condor_sub condor_RecoilRegression.sub` can be used for the purpose.

## Diagnostics

Some scripts to do basic plotting are available.
The inputs can be a directory with results or a CSV list of directories with results.
To plot the loss, mse, and all the metrics which were registered use
```
python scripts/makeTrainValidationPlots.py title:regression_results,title:regression_results,...
```
To make ideogram plots (using event-by-event PDF from the regression).
As it usually takes time -n tells the number of events to use (-1=all)
```
python scripts/makeIdeogramPlots.py -i regression_results/train -o ideograms.root  -n -1
```