#!/usr/bin/env python
import numpy as np
import glob 
import os
import tensorflow as tf 
from sklearn.preprocessing import normalize
import numpy as np
import random
seed = 412
import numpy as np
np.random.seed(seed)
from numpy.random import seed as random_seed
random_seed(seed)
tf.random.set_seed(seed)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from hep_ml import reweight as rw

class SuperCells:
	"""
	Data class for data pre-processing
	Training type CNN
	"""

	def __init__(self, args):
		self.args = args
		self.inputDir = args.input_dir;
		self.findSamples()
		self.DSIDfiles = {}
		self.DicsData  = {}
		self.InputsVars = args.InputVars # ['Z', 'Y', 'Lr0', 'Lr1', 'Lr2', 'Lr3', 'Lr4', 'Lr5', 'Lr6_R0', 'Lr6_R1', 'Lr7_R0', 'Lr7_R1', 'Lr8_R0', 'Lr8_R1', 'Lr21', 'Lr22', 'Lr23']
		#self.InputsVars = ['Z', 'Y', 'Lr0']
		self.strategy = int(self.args.strategy)
		self.setL1XE(self.args.L1_Trigger)
		self.setTarget(self.args.learn_ratio_to)
		self.transformer = StandardScaler()
		self.phiTransformer = QuantileTransformer(output_distribution='normal', random_state=seed)
		self.reweighter  = rw.BinsReweighter(n_bins = 400, n_neighs=0.05)
		
		#self.reweighter  = rw.GBReweighter(n_estimators=80, learning_rate=0.3)
		
	def findSamples(self):
	
		self.DSIDDics = {}
		self.Samples  = os.listdir(self.inputDir)
		self.NSamples = len(self.Samples)
		print('--> found ' + str(self.NSamples) +' samples')
		print(self.Samples)
		for sample in self.Samples:
			DSID = int(sample.split('.')[2])
			dic = {}			
			for name in os.listdir(self.inputDir+sample): 
				
				dic.update({name: self.inputDir+sample+'/'+name+'/'})
			self.DSIDDics.update({DSID: dic})	
		
	def getSample(self, DSID):
	
		if DSID not in self.DSIDDics.keys():
			print(str(DSID) + ' not found')
			print('Only : ' + str(self.DSIDDics.keys()))
			exit()
		return self.DSIDDics[DSID]
				
	def fitTransformerAndReweighter(self):
		
		DSID = self.args.train_on
		file_list = self.DSIDfiles[DSID]['Train'] #use train directory
		print(" Start Scaler() fitting on %i "%(DSID))
		Y = np.load(file_list[0])['Y']
		Z = np.load(file_list[0], allow_pickle=True)['Z']
		for i in range(1, len(file_list)):
			Y = np.append(Y, np.load(file_list[i])['Y'], axis=0)
			Z = np.append(Z, np.load(file_list[i], allow_pickle=True)['Z'], axis=0)
		
		trg = np.random.uniform(Y[:, 0].min(), Y[:, 0].max(), Y[:, 0].shape)
				
		org = np.array(Y[:, 0])	
		self.reweighter.fit(org, trg)
		self.Weights = self.reweighter.predict_weights(org)
		self.SumOfWeights = np.sum(self.Weights)
		phi = np.arctan2(Y[:, 2], Y[:, 1]).reshape(-1, 1)
		
		self.phiTransformer.fit(phi)
		self.transformer.fit(self.getY(Y, Z))
		
					
	def load(self, DSID):
		
		self.files = {}
		for directory, item in self.getSample(DSID).items():
			self.files.update( {directory: [item+f for f in os.listdir(item)] })
		self.DSIDfiles.update({DSID: self.files})
	
	def merge(self, DSID, DSIDs):
	
		tmp = {'Train': [], 'Val': [], 'Test': []}	
		for d in DSIDs:
			for dir in self.DSIDfiles[d].keys():
				
				l = []
				for item in self.DSIDfiles[d][dir]:
					l.append(item)
				tmp[dir] += l
				
		self.DSIDfiles.update({DSID: tmp})
		
	def tf_data_generator(self, DSID, dir, train_test='Train'):
		if train_test == 'Test': self.isTest = True
		if train_test != 'Test': self.isTest = False
		file_list = self.DSIDfiles[DSID][dir]
		i = 0
		generate = True
		while generate:
			if i >= len(file_list):
				np.random.shuffle(file_list)
				generate = False
			else:
				file_chunk = file_list[i:i+1]
				for file in file_chunk:
					temp = np.load(file, allow_pickle=True)
					yield temp, len(file_list)
					del temp
				i = i + 1
				
	def getTargetMET(self, z):
	
		return z[:,self.Targetflag]
		
	def applyL1Trigger(self, x, z):
		z = np.array(z, dtype=np.float32)
		x = np.array(x, dtype=np.float32)
		if self.L1XEflag == -999 or self.L1XEflag == -2022:
			return x, z
		return x[z[:,self.L1XEflag] == 1], z[z[:,self.L1XEflag] == 1]
		
	def applyTruthCut(self, x, y):
		
		if self.L1XEflag != -2022:
			return x
		
		return x[np.array(y, dtype=np.float32)[:,0] < 60*1000]	
	
	def processImage(self, x, z, y):
			# this should be modified
			x = x / x.max(axis= (1,2))[:, np.newaxis, np.newaxis] # normalise the image to the hottest cell
			if not self.isTest:  
				x, z = self.applyL1Trigger(x, z)
			return x
	
	def getWeights(self, y):
		org = np.array(y).reshape(-1, 1)
		org[org/1000 > 300] = 300*1000
		org[org/1000 < 40] = 40*1000
		w  = self.reweighter.predict_weights(org)
		w /= np.median(self.Weights)
		return w.reshape(-1, 1)
		
	
	def getY(self, y, z):
	
		if self.strategy == 1 and self.args.n_outputs == 1:
			return y[:,0].reshape(-1, 1)
			
		elif self.strategy == 2 and self.args.n_outputs == 2:
		     return y[:, 1:3].reshape(-1, 2)
		     
		     '''
		     if self.args.UsePhiTransformer:
			     phi = np.arctan2(y[:, 2], y[:, 1]).reshape(-1, 1)
			     phi = self.phiTransformer.transform(phi).reshape(-1)
		        
			     y = y[:,0]*np.sin(phi)
			     x = y[:,0]*np.cos(phi)
		        
			     return np.concatenate(y.reshape(-1, 1), x.reshape(-1, 1), axis=1).reshape(-1,2)
		     '''
		
		elif self.strategy == 3 and self.args.n_outputs == 2:
			tmp = y[:, 1]/ y[:, 2]
			
			return np.concatenate((y[:, 0].reshape(-1, 1), tmp.reshape(-1, 1)), axis=1).reshape(-1, 2)
		
		elif self.strategy == 4 and self.args.n_outputs == 2:
			return np.concatenate(( np.arctan2( y[:, 1], y[:, 2]).reshape(-1, 1), y[:, 0].reshape(-1, 1)), axis=1).reshape(-1, 2)

		elif self.strategy == 5 and self.args.n_outputs == 1:
			met = self.getTargetMET(z)
			trg = y[:,0]/met
			return trg.reshape(-1, 1)
			
			
		else:
				print("Strategy not defined --> strategy! exit()")
				exit()	
						
	def processfeatures(self, y, z):
		
		if not self.isTest: 
			y, z = self.applyL1Trigger(y, z)
			
		if self.args.UseTransformer:	
			if not self.isTest: return self.transformer.transform(self.getY(y, z)), self.getWeights(y[:, 0])
			return self.transformer.transform(self.getY(y, z)), np.array(z, dtype=np.float64).reshape(-1, z.shape[1])
		else:	
			if not self.isTest: return self.getY(y, z), self.getWeights(y[:, 0])
			return self.getY(y, z), np.array(z, dtype=np.float32).reshape(-1, z.shape[1])
				
	def getprocessfunction(self, name):
		
		if name == "processSuperCells":
			return self.processSuperCells
		else:
			print(name+" not found! exit()")
			exit()
			
	def setL1XE(self, flag):
		
		if flag == 'L1XE30':
			self.L1XEflag = self.args.l1_xe30
		elif flag == 'L1XE50':
			self.L1XEflag = self.args.l1_xe50
		elif flag == 'L1XE60':
			self.L1XEflag = self.args.l1_xe60
		elif flag == 'Truth':
			self.L1XEflag = -2022	
		elif flag == 'None':
			self.L1XEflag = -999
		else:
			print('setL1X1: flag .. '+ flag + ' .. not found')
			exit()		
	
	def setTarget(self, flag):
		
		if flag == 'cell_met':
			self.Targetflag = self.args.cell_met
		elif flag == 'mht_met':
			self.Targetflag = self.args.mht_met
		elif flag == 'topo_puc_met':
			self.Targetflag = self.args.topo_puc_met
		elif flag == 'cell_puc_met':
			self.Targetflag = self.args.cell_puc_met
		elif flag == 'scell_met':
			self.Targetflag = self.args.scell_met	
		elif flag == 'cell_xy':
			self.Targetflag = self.args.cell_xy
		elif flag == 'mht_xy':
			self.Targetflag = self.args.mht_xy
		elif flag == 'topo_puc_xy':
			self.Targetflag = self.args.topo_puc_xy
		elif flag == 'cell_puc_xy':
			self.Targetflag = self.args.cell_puc_xy
		elif flag == 'scell_xy':
			self.Targetflag = self.args.scell_xy				
		elif flag == 'None':
			self.Targetflag = -999
		else:	
			print('setTarget: falg .. '+flag+' .. not found')
			exit()
			
	def processMergedSuperCells(self, x, batch_size):
	
		dic = []
		for item in self.InputsVars:
			if item in ['Z', 'Y']: continue
			dic.append(item)												
							
	def processSuperCells(self, x, batch_size): 
		dic = {}
		for item in self.args.layers_name:
			if item in ['Z', 'Y']: continue
			dic.update({item: self.processImage(x[item], x['Z'], x['Y'])})
			
		with tf.device(self.args.ProcessingDevice):		
			return tf.data.Dataset.from_tensor_slices((dic, self.processfeatures(x['Y'], x['Z']))).shuffle(buffer_size=batch_size, seed=seed).batch(batch_size)	  				
