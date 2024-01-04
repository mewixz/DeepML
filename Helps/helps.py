import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser
import json
import numpy as np
seed = 412
import numpy as np
np.random.seed(seed)
from numpy.random import seed as random_seed
random_seed(seed)
tf.random.set_seed(seed)
import time

@tf.function
def train_step(x, y, model, optimizer, loss, train_metric, epoch_loss_avg):

	''' No need there's something wrong with this function '''
	with tf.GradientTape() as tape:
		train_y = model(x, training=False) # compute predicted y
		train_loss = loss(y, train_y)  # evaluate the loss function between the predicted y and the truth y
	grads = tape.gradient(train_loss, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	train_metric.update_state(y, train_y)
	epoch_loss_avg.update_state(train_loss)
	return train_loss
	
@tf.function
def val_step(x, y, model, loss, val_metric, epoch_loss_avg):
	''' No need there's something wrong with this function '''
	val_y = model(x, training=False)
	val_loss = loss(y, val_y)
	val_metric.update_state(y, val_y)
	epoch_loss_avg.update_state(val_loss)
	return val_loss   	

def getloss(loss): 
	with tf.device('/device:GPU:0'):
		if loss == "mse":
			return tf.keras.losses.MeanSquaredError()
		if loss == "mae":
			return tf.keras.losses.MeanAbsoluteError()
		if loss == "logcosh":
			return tf.keras.losses.LogCosh()
		if loss == "huber":
			return tf.keras.losses.Huber()
		if loss == "mine":
			return custom_loss		
			
		
def getmetric(metric):
	with tf.device('/device:GPU:0'):
		if metric == "mse":
			return tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanSquaredError()
		if metric == "mae":
			return tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsoluteError()
		if metric == "logcosh":
			return tf.keras.metrics.LogCoshError(), tf.keras.metrics.LogCoshError()		

def getoptimizer(optimizer, learning_rate):
	with tf.device('/device:GPU:0'):
		if optimizer == "adam":
			return tf.keras.optimizers.Adam(learning_rate=learning_rate)
			
def test_on_loop(model, DSID, batch_size, data_generator, processFunction, debug):
	
	pred = []
	start_time = time.time()
	
	if debug: print("Testing ...")
	i = 0
	for data, n in data_generator(DSID, 'Test'):
		if debug: print("	Test on file %i/%i"%(i,n))
		test_batch = processFunction(data, batch_size)
		for step, (x_batch_test, y_batch_test) in enumerate(test_batch, 1):
			if debug: print("		process batch %i/%i"%(step, len(test_batch)))
			with tf.device('/device:GPU:0'):
				pred_y = model(x_batch_test, training=False)
			for y_p, y in zip(pred_y.numpy(), y_batch_test.numpy()):
				pred.append([y_p[0], y[0]])
		del test_batch
	df = pd.DataFrame(pred, columns=['predicted', 'truth'])
	return df	
					

def train_on_loop(model, DSID, batch_size, loss, train_metric, val_metric, optimizer, epochs, data_generator, processFunction, debug, args):
	
	status = []
	for epoch in range(epochs):
		start_time = time.time()
		if debug: print("\nStart of epoch %d" % (epoch,))
		epoch_loss_avg = tf.keras.metrics.Mean()
		epoch_loss = []
		if debug: print("Training...")
		i = 0
		for data, n in data_generator(DSID, 'Train'):
			if debug: print("	Train on file %i/%i"%(i,n))
			train_batch = processFunction(data, batch_size)
			for step, (x_batch_train, y_batch_train) in enumerate(train_batch, 1):
				if debug: print("		process batch %i/%i"%(step, len(train_batch)))
				with tf.device('/device:GPU:0'):
					with tf.GradientTape() as tape:
						train_y = model(x_batch_train, training=True) # compute predicted y
						train_loss = loss(y_batch_train, train_y)  # evaluate the loss function between the predicted y and the truth y
						
						print(y_batch_train)
					grads = tape.gradient(train_loss, model.trainable_variables)
					#print("grads : ", grads)
					optimizer.apply_gradients(zip(grads, model.trainable_variables))
					train_metric.update_state(y_batch_train, train_y)
					epoch_loss_avg.update_state(train_loss)
					
					epoch_loss.append(train_loss)	
				
				if debug: print("		training loss = %.4f"%( float(train_loss)))
			del train_batch
			i = i+1
			
		train_epoch_metric  = float(train_metric.result())
		train_epoch_loss    = float(epoch_loss_avg.result())
		
		train_metric.reset_states()
		epoch_loss_avg.reset_states()
		print(" loss %.4f"%(tf.reduce_mean(epoch_loss)))
		 
		if debug: print("Validation...")
		
		i = 0
		for data, n in data_generator(DSID, 'Val'):
			if debug: print("	Validate on file %i/%i"%(i,n))
			val_batch = processFunction(data, batch_size)
			for step, (x_batch_val, y_batch_val) in enumerate(val_batch, 1):
				if debug: print("		process batch %i/%i"%(step, len(val_batch)))
				
				#val_loss = val_step(x_batch_val, y_batch_val, model, loss, val_metric, epoch_loss_avg)
				with tf.device('/device:GPU:0'):
					val_y = model(x_batch_val, training=False)
					val_loss = loss(y_batch_val, val_y)
					val_metric.update_state(y_batch_val, val_y)
					epoch_loss_avg.update_state(val_loss)
				if debug: print("		validation loss = %.4f"%( float(train_loss)))
			del val_batch
			i = i+1
				
		val_epoch_metric  = float(val_metric.result())
		val_epoch_loss    = float(epoch_loss_avg.result())
		
		val_metric.reset_states()
		epoch_loss_avg.reset_states()
		
		status.append([epoch, train_epoch_loss, train_epoch_metric, val_epoch_loss, val_epoch_metric])
		
		print("Epoch %i: training_loss: %.4f , training_metric: %.4f , validation_loss: %.4f , validation_metric: %.4f "%(epoch, 
		train_epoch_loss, train_epoch_metric,
		val_epoch_loss, val_epoch_metric)) 
		print(" Time : %.2fs" % (time.time() - start_time))
		
		print(" Save model weights on Epoch %i"%(epoch))
		
		model.save_weights(args.output_dir+"/model_weight_epoch%i.h5"%(epoch))
		
	df = pd.DataFrame(status, columns=['epoch', 'train_loss', 'train_metric', 'val_loss', 'val_metric'])
	model.trainable = False
	return model, df
	
def train_test_model(args, model, data): 

	dsid                         = args.train_on	
	batch_size                   = args.batch_size
	loss                         = getloss(args.loss)
	train_metric, val_metric     = getmetric(args.metric)
	optimizer                    = getoptimizer(args.optimizer, args.learning_rate)
	data_generator               = data.tf_data_generator
	processfunction              = data.getprocessfunction(args.processFunction)
	
	model, train_df = train_on_loop(model, dsid, batch_size, loss, train_metric, val_metric, optimizer, args.epochs, data_generator, processfunction, args.debug, args)
	
	test_df = test_on_loop(model, dsid, batch_size, data_generator, processfunction, args.debug)
			
	save_result(args, model, train_df, test_df)	
	
def read_json(args):

		config = json.load(open(str(args.parse_args().config), 'r'))
				
		for key, item in config['data_configuration'].items():
			args.add_argument('--'+key, default=item)
		for key, item in config['model_configuration'].items():
			args.add_argument('--'+key, default=item)
		for key, item in config['training_configuration'].items():
			args.add_argument('--'+key, default=item)
		for key, item in config['array_configuration'].items():
			args.add_argument('--'+key, default=item)			
		
		with open(args.parse_args().output_dir+'/model_config.json', 'w') as outfile:
			json.dump(config, outfile)
				
		return args.parse_args()

def add_plot_configuration(args):
	
		config = json.load(open(str(args.parse_args().plot_config), 'r'))
		
		dic= {'stacks': [], 'resolution': [], 'effsrate': [], 'profile': []}
		for plot in dic.keys():
			for key, item in config[plot].items():		
				args.add_argument('--'+key, default=item)
				dic[plot].append(key)	
		args.add_argument('--plots', default=dic)		
				
		return args			
