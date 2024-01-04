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

_gpath = '/media/mbelfkir/diskD/Array/'
_sample= '/user.mobelfki.410471.e6337_s3126_r12406.SCells.V04_OUTPUT/'
_folder= '/Test/'
_file= '/Array_0.npz'

Arr = np.load(_gpath+_sample+_folder+_file, allow_pickle=True)

layers = ['Lr0', 'Lr1', 'Lr2', 'Lr3', 'Lr4', 'Lr5', 'Lr6_R0', 'Lr6_R1', 'Lr7_R0', 'Lr7_R1', 'Lr8_R0', 'Lr8_R1', 'Lr21', 'Lr22', 'Lr23']

for lr in layers:
	x = Arr[lr][0]
	plt.figure()
	plt.imshow( x)
	plt.colorbar()
	plt.savefig('ttbar/NotScaled_'+lr+'.pdf')
	plt.close()
	plt.figure()
	x = x / x.max(axis= (0,1))[:, np.newaxis, np.newaxis]
	plt.imshow( x)
	plt.clim(-1, 1)
	plt.colorbar()
	plt.savefig('ttbar/Scaled_'+lr+'.pdf')
	plt.close()


