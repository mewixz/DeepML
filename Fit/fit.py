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
import Helps.sinkhorn_loss as skhorn

class Fit:

    def __init__(self, model, data, args):
        
        self.data  = data
        self.model = model 
        self.args  = args
        
        device = ''
        
    
    def compile(self):
        
        self.debug                        = self.args.debug
        self.epochs                       = self.args.epochs
        self.dsid                         = self.args.train_on    
        self.batch_size                   = self.args.batch_size
        self.loss                         = self.getloss(self.args.loss)
        self.distance_loss                = tf.keras.losses.KLDivergence()
        self.train_metric, self.val_metric= self.getmetric(self.args.metric)
        self.optimizer                    = self.getoptimizer(self.args.optimizer, self.args.learning_rate)
        self.data_generator               = self.data.tf_data_generator
        self.processfunction              = self.data.getprocessfunction(self.args.processFunction)
        self.epoch_loss_avg               = tf.keras.metrics.Mean()
        self.status = []
        self.pred   = []
        
        self.model.compile(optimizer= self.optimizer, loss=self.loss, metrics=self.train_metric)
    
    
    @tf.function           
    def custom_loss(self, y_true, y_pred):
        loss_x = tf.square( tf.cast(y_true[:, 0], tf.float32) - tf.cast(y_pred[:, 0], tf.float32)) # along x-axis
        loss_y = tf.square( tf.cast(y_true[:, 1], tf.float32) - tf.cast(y_pred[:, 1], tf.float32)) # along y-axis
        loss   = loss_x + loss_y
        return loss    
        
    def getloss(self, loss):
        with tf.device(self.args.TrainingDevice):
            if loss == "mse":
                return tf.keras.losses.MeanSquaredError()
            if loss == "mae":
                return tf.keras.losses.MeanAbsoluteError()
            if loss == "logcosh":
                return tf.keras.losses.LogCosh()
            if loss == "huber":
                return tf.keras.losses.Huber()
            if loss == "mine":
                return self.custom_loss        
            
    def getmetric(self, metric):
        with tf.device(self.args.TrainingDevice):
            if metric == "mse":
                return tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanSquaredError()
            if metric == "mae":
                return tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsoluteError()
            if metric == "logcosh":
                return tf.keras.metrics.LogCoshError(), tf.keras.metrics.LogCoshError()        

    def getoptimizer(self, optimizer, learning_rate):
        with tf.device(self.args.TrainingDevice):
            if optimizer == "adam":
                return tf.keras.optimizers.Adam(learning_rate=learning_rate)
                
    def getTriangleLR(self, step, nstep):
        
        max_lr = self.args.max_lr 
        min_lr = self.args.min_lr 
        n      = int(nstep/4)
        delta  = (max_lr - min_lr)/n
        l      = min_lr
        self.lr = []
        for i in range(0, nstep):
            if i in range(0, n):
                l = l + delta
            if i in range(n, 2*n): 
                l = l - delta
            if i in range(2*n, 3*n):
                l = l + delta
            if i in range(3*n, 4*n):
                l = l - delta
            
            self.lr.append(l)
        
        return self.lr[step]        
                    
        
    def fit(self):
        reduce_on_plateau_loss = [-999]
        patience = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            if self.args.debug: print("\nStart of epoch %d" % (epoch,))
            if self.args.debug: print("Training...")
            i = 0
            for data, n in self.data_generator(self.dsid, 'Train', 'Train'):
                if self.debug: print("    Train on file %i/%i"%(i,n))
                train_batch = self.processfunction(data, self.batch_size)
                for step, (x_batch_train, yw_batch_train) in enumerate(train_batch):
                    if self.args.doTraingleLR:
                        new_lr = self.getTriangleLR(step, len(train_batch))
                        self.optimizer.lr = new_lr
                    if self.args.debug: print("        process batch %i/%i"%(step, len(train_batch)))
                    train_loss = self.on_epoch_fit(x_batch_train, yw_batch_train[0], yw_batch_train[1])
                    self.epoch_loss_avg.update_state(train_loss)
                    if self.debug: print("        training loss = %.4f"%( float(train_loss)))
                    i = i+1
                    
                del train_batch
                
            train_epoch_metric  = float(self.train_metric.result())
            train_epoch_loss    = float(self.epoch_loss_avg.result())
            
            self.train_metric.reset_states()
            self.epoch_loss_avg.reset_states()
            
            if self.debug: print("Validation...")
            i = 0
            for data, n in self.data_generator(self.dsid, 'Val', 'Train'):
                if self.debug: print("    Validate on file %i/%i"%(i,n))
                val_batch = self.processfunction(data, self.batch_size)
                for step, (x_batch_val, yw_batch_val) in enumerate(val_batch):
                    if self.debug: print("        process batch %i/%i"%(step, len(val_batch)))
                    val_loss = self.on_epoch_eval(x_batch_val, yw_batch_val[0], yw_batch_val[1])
                    self.epoch_loss_avg.update_state(val_loss)
                    if self.debug: print("        validation loss = %.4f"%( float(val_loss)))
                    
                del val_batch
                i = i+1
                    
            val_epoch_metric  = float(self.val_metric.result())
            val_epoch_loss    = float(self.epoch_loss_avg.result())
            
            self.val_metric.reset_states()
            self.epoch_loss_avg.reset_states()
            
            reduce_on_plateau_loss.append(val_epoch_loss)
            
            
            if self.args.ReduceOnPlateau:
                vl1 =     reduce_on_plateau_loss [-2]
                vl2 =   reduce_on_plateau_loss [-1]
                if vl1-vl2 <= 1e-3:
                    patience += 1
                    print(" Validation loss not improved: Old value = %.4f --> New value = %.4f,  patience = %i " %(vl1, vl2, patience))
                else:
                    patience = 0
                    print(" Validation loss improved: Old value = %.4f --> New value = %.4f,  patience = %i " %(vl1, vl2, patience))
                    
                if patience >= 3:
                    patience = 0
                    old_lr = self.optimizer.get_config()['learning_rate']
                    new_lr = 0.2*old_lr
                    self.optimizer.lr = new_lr
                    new_lr = "{:e}".format(new_lr)
                    old_lr = "{:e}".format(old_lr)
                    print(" Learning Rate reduced: Old value = %s --> New value = %s "%(old_lr, new_lr))                
            
            
            self.status.append([epoch, train_epoch_loss, train_epoch_metric, val_epoch_loss, val_epoch_metric])
            
            print("Epoch %i: training_loss: %.4f , training_metric: %.4f , validation_loss: %.4f , validation_metric: %.4f "%(epoch+1, 
            train_epoch_loss, train_epoch_metric,
            val_epoch_loss, val_epoch_metric)) 
            
            print(" Time : %.2fs" % (time.time() - start_time))
            
            if self.args.SaveOnEpoch: 
                print(" Save model weights on Epoch %i"%(epoch+1))
                self.model.save_weights(self.args.output_dir+"/model_weight_epoch_%i.h5"%(epoch+1))
            
        df = pd.DataFrame(self.status, columns=['epoch', 'train_loss', 'train_metric', 'val_loss', 'val_metric'])
        self.model.trainable = False
        return df
        
    @tf.function
    def computeLoss(self, label, prediction, w):
        with tf.device(self.args.TrainingDevice):
            reg_loss = self.loss(label, prediction)
            if self.args.UseReweighter: 
                reg_loss = self.loss(label, prediction, sample_weight=w)
            
            dist_loss = tf.zeros_like(reg_loss)    
            if self.args.UseSinKhorn:
                epsilon      = 0.001
                num_iterates = 100
                batch_size   = len(label)
                #dist_loss = skhorn.sinkhorn_loss(label, prediction, epsilon, batch_size, num_iterates)
                m = 0.5*(label+prediction)
                dist_loss = 0.5*self.distance_loss(label, m) + 0.5*self.distance_loss(prediction, m)
            return reg_loss + dist_loss        
        
    @tf.function            
    def on_epoch_fit(self, x, y, w):
        self.model.trainable = True
        with tf.device(self.args.TrainingDevice):
            with tf.GradientTape() as tape:
                train_y = self.model(x, training=True) # compute predicted y
                train_loss = self.computeLoss(y, train_y, w)  # evaluate the loss function between the predicted y and the truth y
            grads = tape.gradient(train_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.train_metric.update_state(y, train_y)
            return train_loss
    
    @tf.function
    def on_epoch_eval(self, x, y, w):
        self.model.trainable = False
        with tf.device(self.args.TrainingDevice):
            val_y = self.model(x, training=False)
            val_loss = self.computeLoss(y, val_y, w)
            self.val_metric.update_state(y, val_y)
            
            return val_loss
    
    
    @tf.function
    def on_eval(self, x):
        self.model.trainable = False
        with tf.device(self.args.TrainingDevice):
            pred_y = self.model(x, training=False)
            return pred_y    
        
    def eval(self, dir):
        
        if self.debug: print("Testing ...")
        i = 0
        for data, n in self.data_generator(self.dsid, dir, 'Test'):
            if self.debug: print("    Test on file %i/%i"%(i,n))
            test_batch = self.processfunction(data, self.batch_size)
            for step, (x_batch_test, yz_batch_test) in enumerate(test_batch, 1):
                if self.debug: print("        process batch %i/%i"%(step, len(test_batch)))
                pred_y = self.on_eval(x_batch_test)
                for y_p, y, z in zip(pred_y.numpy(), yz_batch_test[0].numpy(), yz_batch_test[1].numpy()):
                    if self.args.UseTransformer:
                        y   = self.data.transformer.inverse_transform(y)
                        y_p = self.data.transformer.inverse_transform(y_p)
                    vec = []
                    if self.args.strategy == 1:
                        met_p = y_p[0]
                        met_t = y
                        x_p   = np.zeros(met_p.shape)
                        x_t   = np.zeros(met_t.shape)
                        y_p   = np.zeros(met_p.shape)
                        y_t   = np.zeros(met_t.shape)
                        phi_p = np.zeros(met_p.shape)
                        phi_t = np.zeros(met_t.shape)
                        vec = [y_p, x_p, met_p, phi_p, y_t, x_t, met_t, phi_t]
                    elif self.args.strategy == 2:
                    
                        x_p   = y_p[1]
                        x_t   = y[1]
                        y_p   = y_p[0]
                        y_t   = y[0]
                        phi_p = np.arctan2(y_p, x_p)
                        phi_t = np.arctan2(y_t, x_t)
                        met_p = (x_p**2 + y_p**2)**0.5
                        met_t = (x_t**2 + y_t**2)**0.5
                        '''
                        if self.args.UsePhiTransformer:
                            phi_p = self.data.phiTransformer.inverse_tranform(phi_p)
                            phi_t = self.data.phiTransformer.inverse_tranform(phi_t)
                        '''
                        vec = [y_p, x_p, met_p, phi_p, y_t, x_t, met_t, phi_t]
                    elif self.args.strategy == 3:
                    
                        met_p = y_p[0]
                        met_t = y[0]
                        phi_p = np.arctan(y_p[1])
                        
                        phi_t = np.arctan(y[1])
                        x_p   = met_p * np.cos(phi_p)
                        x_t   = met_t * np.cos(phi_t)
                        y_p   = met_p * np.sin(phi_p)
                        y_t   = met_t * np.sin(phi_t)
                        vec = [y_p, x_p, met_p, phi_p, y_t, x_t, met_t, phi_t]
                        
                    elif self.args.strategy == 4:
                    
                        met_p = y_p[1]
                        met_t = y[1]
                        phi_p = y_p[0]
                        phi_t = y[0]
                        
                        x_p   = met_p * np.cos(phi_p)
                        x_t   = met_t * np.cos(phi_t)
                        y_p   = met_p * np.sin(phi_p)
                        y_t   = met_t * np.sin(phi_t)
                        vec = [y_p, x_p, met_p, phi_p, y_t, x_t, met_t, phi_t]

                    elif self.args.strategy == 5:
                    	alpha_p = y_p[0]
                    	alpha_t = y[0]
                    	trg   = z[self.data.Targetflag]
                    	met_p = trg*alpha_p
                    	met_t = trg*alpha_t
                    	x_p   = np.zeros(met_p.shape)
                    	x_t   = np.zeros(met_t.shape)
                    	y_p   = np.zeros(met_p.shape)
                    	y_t   = np.zeros(met_t.shape)
                    	phi_p = np.zeros(met_p.shape)
                    	phi_t = np.zeros(met_t.shape)
                    	vec = [y_p, x_p, met_p, phi_p, y_t, x_t, met_t, phi_t]

                    else:
                        print("No strategy ! this should not happen here")
                        exit()
                    for z_i in z:
                        vec.append(z_i)
                    self.pred.append(vec)        
            del test_batch
        
        columns=['predicted_y', 'predicted_x', 'predicted_met', 'predicted_phi', 'truth_y', 'truth_x', 'truth_met', 'truth_phi']
        for col in self.args.Z_names:
            columns.append(col)
            
        df = pd.DataFrame(self.pred, columns=columns)
        return df
    
    def execute(self, isTest=True):
        
        self.fit_df = self.fit()
        
        if isTest: self.test_df = self.eval('Test')
    
    def share(self):
        return self.fit_df, self.test_df, self.model
