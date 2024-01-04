#!/usr/bin/env python
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import json
import os
seed = 412
import scipy.stats as stats
import numpy as np
np.random.seed(seed)
from numpy.random import seed as random_seed
random_seed(seed)
import matplotlib.pyplot as plt  
import math
from pickle import load
def getSamples(args):

	dic = {410471: 'ttbar', 345058: 'ggZH(vv)', 361108: 'Ztautau', 361020: 'JZ0W', 0:'TEST', 1: 'TEST2', 999999: 'Combined', 361021: 'JZ1W'}
	return dic[args]
	
def get_df(path, model):
	path = path+model+'/results/'
	files  = os.listdir(path)
	print('--following result found :',files)
	dic = {}
	for file in files: 
		DSID = file.split('.')[1]
		dic.update({int(DSID): pd.read_hdf(path+file)})

	return dic

_gpath = '/home/mbelfkir/TriggerML/DGX1/Stg5/'
_model = 'Train_Stg5_Cell_Rwg/'
_reweighter = load( open(_gpath+_model+'/model/reweighter.pkl', 'rb'))
dfs = get_df(_gpath, _model)


dfs_rwg   = get_df(_gpath, _model)
dfs_norwg = get_df(_gpath, 'Train_Stg5_Cell_NoRwg/')

'''
 Plot truth MET before and after the reweighting
'''


def plot_MET(dsid, df_rwg, df_norwg):
	
	plt.figure()
	hist_settings = {'bins': 150, 'range':[0, 500], 'density': True, 'histtype': 'step', 'linewidth':1.2}
	plt.hist(df_rwg[dsid].truth_met/1000, **hist_settings)
	plt.hist(df_rwg[dsid].predicted_met/1000, **hist_settings)
	plt.hist(df_norwg[dsid].predicted_met/1000, **hist_settings)
	legend = plt.legend(['Truth', 'w/ Reweighting', 'w/o Reweighting'], loc='best')
	plt.xlabel('Truth MET [GeV]')
	plt.ylabel('Fraction of Events')
	legend.set_title("%s"%(getSamples(dsid)))
	plt.savefig(_gpath+'/Compare/met_Sample_'+getSamples(dsid)+'.pdf')
	plt.close()

def plot_MET_reweighted(rwgt, dsid, df):
	
	wgt = rwgt.predict_weights(df.truth_met)
	
	plt.figure()
	hist_settings = {'bins': 150, 'range':[0, 500], 'density': True, 'histtype': 'step', 'linewidth':1.2}
	plt.hist(df.truth_met/1000, **hist_settings)
	plt.hist(df.truth_met/1000, weights=wgt, **hist_settings)
	#plt.hist(df.predicted_met/1000, **hist_settings)
	legend = plt.legend(['Before Reweighting', 'After Reweighting'], loc='best')
	#legend = plt.legend(['MET'], loc='best')
	plt.xlabel('Truth MET [GeV]')
	plt.ylabel('Fraction of Events')
	legend.set_title("%s"%(getSamples(dsid)))
	plt.savefig(_gpath+_model+'/plots/met_Sample_'+getSamples(dsid)+'.pdf')
	plt.close()
	
	plt.figure()
	hist_settings = {'bins': 100, 'range':[-500, 500], 'density': True, 'histtype': 'step', 'linewidth':1.2}
	plt.hist(df.truth_x/1000, **hist_settings)
	plt.hist(df.truth_x/1000, weights=wgt, **hist_settings)
	legend = plt.legend(['Before Reweighting', 'After Reweighting'], loc='best')
	plt.xlabel('Truth Px [GeV]')
	plt.ylabel('Fraction of Events')
	legend.set_title("%s"%(getSamples(dsid)))
	plt.savefig(_gpath+_model+'/plots/px_Sample_'+getSamples(dsid)+'.pdf')
	plt.close()
	
	
	plt.figure()
	hist_settings = {'bins': 100, 'range':[-500, 500], 'density': True, 'histtype': 'step', 'linewidth':1.2}
	plt.hist(df.truth_y/1000, **hist_settings)
	plt.hist(df.truth_y/1000, weights=wgt, **hist_settings)
	legend = plt.legend(['Before Reweighting', 'After Reweighting'], loc='best')
	plt.xlabel('Truth Py [GeV]')
	plt.ylabel('Fraction of Events')
	legend.set_title("%s"%(getSamples(dsid)))
	plt.savefig(_gpath+_model+'/plots/py_Sample_'+getSamples(dsid)+'.pdf')
	plt.close()
	
	
	plt.figure()
	hist_settings = {'bins': 100, 'range':[-4, 4], 'density': True, 'histtype': 'step', 'linewidth':1.2}
	plt.hist( np.arctan2(df.truth_y, df.truth_x), **hist_settings)
	plt.hist( np.arctan2(df.truth_y, df.truth_x), weights=wgt, **hist_settings)
	#plt.hist( np.arctan2(df.predicted_y, df.predicted_x), **hist_settings)
	legend = plt.legend(['Before Reweighting', 'After Reweighting'], loc='best')
	plt.xlabel('Phi')
	plt.ylabel('Fraction of Events')
	legend.set_title("%s"%(getSamples(dsid)))
	plt.savefig(_gpath+_model+'/plots/phi_Sample_'+getSamples(dsid)+'.pdf')
	plt.close()
	
def plot_target_reweighted(rwgt, dsid, df):
	
	wgt = rwgt.predict_weights(df.truth_met)
	plt.figure()
	hist_settings = {'bins': 100, 'range':[0, 2], 'density': True, 'histtype': 'step', 'linewidth':1.2}
	plt.hist(df.truth_met/df.cell_et, **hist_settings)
	plt.hist(df.truth_met/df.cell_et, weights=wgt, **hist_settings)
	plt.hist(df.predicted_met/df.cell_et, **hist_settings)
	legend = plt.legend(['Before Reweighting', 'After Reweighting', 'Predicted'], loc='best')
	plt.xlabel('Target Truth/Cell')
	plt.ylabel('Fraction of Events')
	legend.set_title("%s"%(getSamples(dsid)))
	plt.savefig(_gpath+_model+'/plots/target_before_after_rwgt_Sample_'+getSamples(dsid)+'.pdf')	


for dsid, df in dfs.items():
	
	#plot_MET_reweighted(_reweighter, dsid, df)
	#plot_target_reweighted(_reweighter, dsid, df)
	if dsid == 361021:
		continue
	plot_MET(dsid, dfs_rwg, dfs_norwg)	
	
	


