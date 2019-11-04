import tensorflow as tf
import numpy as np
import h5py
import time
import glob
import scipy
import argparse
import os
from PIL import Image

from utils import *
from nets import FR_16L, FR_28L, FR_52L

class DUF(object):
    model_name = "VSR-DUF"     # name for checkpoint

    def __init__(self):
    	return

    #network
    def G(self,x, is_train):  
        # shape of x: [B,T_in,H,W,C]

        # Generate filters and residual
        # Fx: [B,1,H,W,1*5*5,R*R]
        # Rx: [B,1,H,W,3*R*R]
        FR=self.FR
        T_in=self.T_in
        R=self.R
        Fx, Rx = FR(x, is_train) 
        x_c = []
        for c in range(3):
            #t = DynFilter3D(x[:,T_in//2:T_in//2+1,:,:,c], Fx[:,0,:,:,:,:], [1,5,5]) # [B,H,W,R*R]
            t = DynFilter3D(x[:,T_in//2:T_in//2+1,:,:,c], Fx[:,0,:,:,:,:], [1,5,5],c) # [B,H,W,R*R]
            t = tf.depth_to_space(t, R) # [B,H*R,W*R,1]
            x_c += [t]
        x = tf.concat(x_c, axis=3)   # [B,H*R,W*R,3]
        x = tf.expand_dims(x, axis=1)

        Rx = depth_to_space_3D(Rx, R)   # [B,1,H*R,W*R,3]
        x += Rx
        
        return x

    def W(self,frame1,frame2,flow,is_train):  
        # shape of frame1/2: [B,1,H,W,C]
        # shape of flow: [B,2,H,W]
        frame1=tf.squeeze(frame1,axis=1)
        frame2=tf.squeeze(frame2,axis=1)
        # warp frame2 to frame1
        frame2_w=self.tf_warp(frame2, flow)
        #print(frame1[0].shape)
        '''
        #crop
        frame1_list = []
        frame2_list = []
        for i in range(self.batch_size):
            frame1_list.append(tf.image.central_crop(frame1[i], 108.0/128.0))
            frame2_list.append(tf.image.central_crop(frame2_w[i], 108.0/128.0))
        frame1 = tf.convert_to_tensor(frame1_list)
        frame2 = tf.convert_to_tensor(frame2_list)
            # frame1=tf.image.central_crop(frame1,108.0*108.0/128.0/128.0)
            # frame2=tf.image.central_crop(frame2_w,108.0*108.0/128.0/128.0)
        '''
        return frame1,frame2_w

    def get_pixel_value(self,img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W, )
        - y: flattened tensor of shape (B*H*W, )
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width)) #https://blog.csdn.net/LoseInVain/article/details/78994615?utm_source=blogxgwz0

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)

    def tf_warp(self,img, flow):
    #    H = 256
    #    W = 256
        x,y = tf.meshgrid(tf.range(self.width), tf.range(self.height))
        x = tf.expand_dims(x,0)
        x = tf.expand_dims(x,0)

        y  =tf.expand_dims(y,0)
        y = tf.expand_dims(y,0)

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        grid  = tf.concat([x,y],axis = 1)
    #    print grid.shape
        flows = grid+flow
        #print flows.shape
        max_y = tf.cast(self.height - 1, tf.int32)
        max_x = tf.cast(self.width - 1, tf.int32)
        zero = tf.zeros([], dtype=tf.int32)

        x = flows[:,0,:,:]
        y = flows[:,1,:,:]
        x0 = x
        y0 = y
        x0 = tf.cast(x0, tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(y0,  tf.int32)
        y1 = y0 + 1

        # clip to range [0, H/W] to not violate img boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        Ia = self.get_pixel_value(img, x0, y0)
        Ib = self.get_pixel_value(img, x0, y1)
        Ic = self.get_pixel_value(img, x1, y0)
        Id = self.get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)


        # calculate deltas
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return out


    # Gaussian kernel for downsampling
    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        # create nxn zeros
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen//2, kernlen//2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    def build_model(self):
        # some parameters
        bs = self.batch_size
        T_in=self.T_in
        R=self.R
        FR=self.FR

        """ Graph Input """
        # images
        self.inputs1 = tf.placeholder(tf.float32, shape=[bs, T_in, None, None, self.c_dim], name='real_images1')
        self.labels1 = tf.placeholder(tf.float32, shape=[bs, 1, None, None, self.c_dim], name='label_images1')
        self.inputs2 = tf.placeholder(tf.float32, shape=[bs, T_in, None, None, self.c_dim], name='real_images2')
        self.labels2 = tf.placeholder(tf.float32, shape=[bs, 1, None, None, self.c_dim], name='label_images2')
        self.flow = tf.placeholder(tf.float32, shape=[bs, 2, None, None], name='flow_from_1_to_2')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.is_train = tf.placeholder(tf.bool, shape=[])
        """ Loss Function """
        # Network
        with tf.variable_scope('G') as scope:
            self.outputs1 = self.G(self.inputs1, self.is_train)
        with tf.variable_scope('G', reuse = True) as scope:
            self.outputs2 = self.G(self.inputs2, self.is_train)
        self.frame1,self.frame2=self.W(self.outputs1,self.outputs2,self.flow,self.is_train)
        '''
        frame1l_list = []
        for i in range(self.batch_size):
            frame1l_list.append(tf.image.central_crop(self.labels1[i,0], 108.0/128.0))
        self.frame1l = tf.convert_to_tensor(frame1l_list)
        '''
        self.params_G = [v for v in tf.global_variables() if v.name.startswith('G/')]

        # loss
        self.loss1 = Huber(self.labels1, self.outputs1, 0.01)
        self.loss2 = Huber(self.labels2, self.outputs2, 0.01)
        #self.loss3 = tf.losses.mean_squared_error(self.frame1,self.frame2))
        self.loss3 = Huber(self.frame1, self.frame2, 0.01)
        self.loss4 = Huber(tf.squeeze(self.labels1,axis=1), self.frame2, 0.01)
        self.loss = self.loss1+self.loss2+1*self.loss3+1*self.loss4
        """ Training """
        # optimizers
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.params_G)

        """" Testing """
        # for test
        #with tf.variable_scope('G') as scope:
         #    self.test_outputs = G(self.test_inputs, is_training=False)

    def train(self):
        
          # Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        # initialize all variables
        with tf.Session(config=config) as self.sess:
            tf.global_variables_initializer().run()

            # saver to save model
            self.saver = tf.train.Saver()
            
            # Load parameters
            #self.load(self.checkpoint_dir)
            LoadParams(self.sess, [self.params_G], in_file='params_{}L_x{}.h5'.format(self.L, self.R))
            
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            #print(" [!] Load failed...")

            # loop for epoch
            start_time = time.time()
            learning_rate=0.01
            for epoch in range(start_epoch, self.epoch):

                if not os.path.exists(os.path.join(self.checkpoint_dir, 'training/Epoch{:02d}'.format(epoch))):
                    os.makedirs(os.path.join(self.checkpoint_dir, 'training/Epoch{:02d}'.format(epoch)))

                if epoch%10==0:
                    learning_rate=learning_rate*0.1
                
                # batch
                random_list=np.random.permutation(self.training_label1.shape[0])
                for idx in range(start_batch_id, self.num_batches):
                    #prepare for data for each batch
                    inputs1=self.training_input1.transpose(0,3,2,1)
                    labels1=self.training_label1.transpose(0,3,2,1)
                    inputs2=self.training_input2.transpose(0,3,2,1)
                    labels2=self.training_label2.transpose(0,3,2,1)
                    #inputs_tmp = np.lib.pad(inputs, pad_width=((0,0),(8,8),(8,8),(0,0)), mode='reflect')
                    
                    inputss1,labelss1=self.get_batch(inputs1,labels1,random_list,idx)
                    inputss2,labelss2=self.get_batch(inputs2,labels2,random_list,idx)
                    '''
                    inputs_tmp = inputs1
                    labels_tmp = labels1
                    inputss1=np.zeros([self.batch_size,self.T_in,inputs_tmp.shape[1],inputs_tmp.shape[2],self.c_dim], dtype= 'float32')
                    labelss1=np.zeros([self.batch_size,1,labels_tmp.shape[1],labels_tmp.shape[2],self.c_dim], dtype= 'float32')
                    for iidx in range(0,self.batch_size):
                        offset=random_list[idx*self.batch_size+iidx]
                        inputss1[iidx,:,:,:,:]=inputs_tmp[offset*self.T_in:offset*self.T_in+self.T_in,:,:,:]
                        labelss1[iidx,:,:,:,:]=labels_tmp[offset,:,:,:]
                    '''
                    
                    #self.height = labels1.shape[1]
                    #self.width = labels1.shape[2]
                    flow1=self.training_flow1.transpose(0,1,3,2)
                    flow2=self.training_flow2.transpose(0,1,3,2)
                    flow=np.zeros([self.batch_size, 2, self.height, self.width], dtype= 'float32')
                    for iidx in range(0,self.batch_size):
                        offset=random_list[idx*self.batch_size+iidx]
                        flow[iidx,0,:,:]=flow1[offset,:,:,:]
                        flow[iidx,1,:,:]=flow2[offset,:,:,:]
                    
                    


                    # update
                    _,loss,loss1,loss2,loss3,loss4,sample_output1,sample_output2,sample_frame1,sample_frame2= self.sess.run([self.optim, self.loss, self.loss1, self.loss2, self.loss3, self.loss4, self.outputs1, self.outputs2, self.frame1, self.frame2],
                                                   feed_dict={self.inputs1: inputss1,self.inputs2: inputss2, self.labels1: labelss1,self.labels2: labelss2,self.flow: flow, self.learning_rate: learning_rate, self.is_train: True})
                    
                    #save sampled image
                    self.sample_path=os.path.join(self.checkpoint_dir, 'training/Epoch{:02d}'.format(epoch))+'/Iter{:05d}.png'.format(idx)
                    #self.imsave(sample_frame1)
                    
                    # display training status
                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f = %.8f * 1 + %.8f * 1 + %.8f* 1 + %.8f* 1" % (epoch, idx, self.num_batches, time.time() - start_time, loss ,loss1, loss2, loss3 ,loss4))
                # After an epoch, start_batch_id is set to zero
                start_batch_id=0
                # save model
                self.save(self.checkpoint_dir, counter)
                #save sampled image
                self.sample_path=os.path.join(self.checkpoint_dir, 'training/Epoch{:02d}'.format(epoch))+'/output1.png'
                self.imsave(sample_output1)
                self.sample_path=os.path.join(self.checkpoint_dir, 'training/Epoch{:02d}'.format(epoch))+'/output2.png'
                self.imsave(sample_output2)
                self.sample_path=os.path.join(self.checkpoint_dir, 'training/Epoch{:02d}'.format(epoch))+'/frame1.png'
                self.imsave(sample_frame1)
                self.sample_path=os.path.join(self.checkpoint_dir, 'training/Epoch{:02d}'.format(epoch))+'/frame2.png'
                self.imsave(sample_frame2)
                # Test
                Ave_loss=[]
                Ave_loss1=[]
                Ave_loss2=[]
                Ave_loss3=[]
                Ave_loss4=[]
                random_list=range(self.valid_label1.shape[0])
                for idx in range(self.num_test_batches):
                    #prepare for data for each batch
                    inputs1=self.valid_input1.transpose(0,3,2,1)
                    labels1=self.valid_label1.transpose(0,3,2,1)
                    inputs2=self.valid_input2.transpose(0,3,2,1)
                    labels2=self.valid_label2.transpose(0,3,2,1)
                    #inputs_tmp = np.lib.pad(inputs, pad_width=((0,0),(8,8),(8,8),(0,0)), mode='reflect')
                    
                    inputss1,labelss1=self.get_batch(inputs1,labels1,random_list,idx)
                    inputss2,labelss2=self.get_batch(inputs2,labels2,random_list,idx)
                    
                    flow1=self.valid_flow1.transpose(0,1,3,2)
                    flow2=self.valid_flow2.transpose(0,1,3,2)
                    flow=np.zeros([self.batch_size, 2, self.height, self.width], dtype= 'float32')
                    for iidx in range(0,self.batch_size):
                        offset=random_list[idx*self.batch_size+iidx]
                        flow[iidx,0,:,:]=flow1[offset,:,:,:]
                        flow[iidx,1,:,:]=flow2[offset,:,:,:]


                    # test loss
                    '''
                    loss,loss1,loss2,loss3,sample_output1,sample_output2,sample_frame1,sample_frame2= self.sess.run([self.loss, self.loss1, self.loss2, self.loss3, self.outputs1, self.outputs2, self.frame1, self.frame1],
                                                   feed_dict={self.inputs1: inputss1,self.inputs2: inputss2, self.labels1: labelss1,self.labels2: labelss2,self.flow: flow, self.learning_rate: learning_rate, self.is_train: False})
                    '''
                    loss,loss1,loss2,loss3,loss4= self.sess.run([self.loss, self.loss1, self.loss2, self.loss3, self.loss4],
                                                   feed_dict={self.inputs1: inputss1,self.inputs2: inputss2, self.labels1: labelss1,self.labels2: labelss2,self.flow: flow, self.learning_rate: learning_rate, self.is_train: False})
                    Ave_loss.append(loss)
                    Ave_loss1.append(loss1)
                    Ave_loss2.append(loss2)
                    Ave_loss3.append(loss3)
                    Ave_loss4.append(loss4)
                    print("Epoch[%2d] test: Iter[%4d/%4d] Done, loss: %.8f = %.8f * 1 + %.8f * 1 + %.8f* 1 + %.8f* 1" % (epoch, idx, self.num_test_batches, loss ,loss1, loss2, loss3, loss4))
                loss_aver=np.mean(np.asarray(Ave_loss))
                loss1_aver=np.mean(np.asarray(Ave_loss1))
                loss2_aver=np.mean(np.asarray(Ave_loss2))
                loss3_aver=np.mean(np.asarray(Ave_loss3))
                loss4_aver=np.mean(np.asarray(Ave_loss4))
                print("Epoch: [%2d] , average test loss: %.8f = %.8f * 1 + %.8f * 1 + %.8f* 1 + %.8f* 1" % (epoch, loss_aver ,loss1_aver, loss2_aver, loss3_aver, loss4_aver))

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def get_batch(self,inputs_tmp,labels_tmp,random_list,idx):
        inputss1=np.zeros([self.batch_size,self.T_in,inputs_tmp.shape[1],inputs_tmp.shape[2],self.c_dim], dtype= 'float32')
        labelss1=np.zeros([self.batch_size,1,labels_tmp.shape[1],labels_tmp.shape[2],self.c_dim], dtype= 'float32')
        for iidx in range(0,self.batch_size):
            offset=random_list[idx*self.batch_size+iidx]
            inputss1[iidx,:,:,:,:]=inputs_tmp[offset*self.T_in:offset*self.T_in+self.T_in,:,:,:]
            labelss1[iidx,:,:,:,:]=labels_tmp[offset,:,:,:]
        return inputss1,labelss1

    def imsave(self,im):
        im = np.clip(im, 0, 1)
        if len(im.shape)==5:
            Image.fromarray(np.around(im[0,0]*255).astype(np.uint8)).save(self.sample_path)
        else:
            Image.fromarray(np.around(im[0]*255).astype(np.uint8)).save(self.sample_path)


    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_data(self,path):
        f = h5py.File(path,'r')                       
        inputs = f['data'][:]
        labels = f['label'][:]
        #[B,C,W,H]
        f.close()
        inputs *= 1.0/255.0
        labels *= 1.0/255.0
        return inputs,labels

    def load_flow(self,path):
        f = h5py.File(path,'r')                       
        inputs = f['data'][:]
        labels = f['label'][:]
        f.close()
        #inputs *= 1.0/255.0
        #labels *= 1.0/255.0
        return inputs,labels


parser = argparse.ArgumentParser()
parser.add_argument('L', metavar='L', type=int, help='Network depth: One of 16, 28, 52')
args = parser.parse_args()

net=DUF()
# image size
net.c_dim = 3
net.batch_size = 16
net.epoch=20
net.model_dir='model'
net.checkpoint_dir='./TFoutput1111'
# Size of input temporal radius
net.T_in = 7
# Upscaling factor
net.R = 4
# Selecting filters and residual generating network
if args.L == 16:
    FR = FR_16L
elif args.L == 28:
    FR = FR_28L
elif args.L == 52:
    FR = FR_52L
else:
    print('Invalid network depth: {} (Must be one of 16, 28, 52)'.format(args.L))
    exit(1)
net.FR=FR
net.L=args.L
#load dataset
net.training_input1,net.training_label1=net.load_data('/userhome/Baseline2/dataset/train_frame1.h5')
net.training_input2,net.training_label2=net.load_data('/userhome/Baseline2/dataset/train_frame2.h5')
net.training_flow1,net.training_flow2=net.load_flow('/userhome/Baseline2/dataset/train_flow.h5')

net.height=net.training_label1.shape[3]
net.width=net.training_label1.shape[2]
net.num_batches=int(net.training_input1.shape[0]/net.batch_size/7)
                     
net.valid_input1,net.valid_label1 = net.load_data('/userhome/Baseline2/dataset/valid_frame1.h5')
net.valid_input2,net.valid_label2 = net.load_data('/userhome/Baseline2/dataset/valid_frame2.h5')
net.valid_flow1,net.valid_flow2=net.load_flow('/userhome/Baseline2/dataset/valid_flow.h5')
net.num_test_batches=int(net.valid_input1.shape[0]/net.batch_size/7)

net.build_model()
net.train()