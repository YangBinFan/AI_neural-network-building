import tensorflow as tf

import numpy as np
import pickle
import time
import os
import pdb
import cv2
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from skimage.util.shape import view_as_blocks

from super_CNN import super_CNN
from utils import recursiveFileList


class Img_denoise(super_CNN):
    def __init__(self, sess,
                 learnRate=0.001,
                 shuffle=True,
                 batch_size=100,
                 epoch=1000,
                 input_dim=784,
                 output_dim=10,
                 showPlot=False,
                 main_save_folder='tmp_model/'):

        super(super_CNN, self).__init__()

        self.sess              = sess

        self.learnRate         = learnRate
        self.shuffle           = shuffle
        self.batch_size        = batch_size
        self.epoch             = epoch

        # self.input_dim=[None, dim. of cir. conv., width, height, input channels]
        # self.output_dim=[None, # of classes]

        self.input_dim         = input_dim    # [None, 3, 28, 28, 1]
        self.output_dim        = output_dim  # [None, 10]
        input_dim[0] = output_dim[0] = None
        # pdb.set_trace()

        self.showPlot          = showPlot

        self.main_save_folder  = main_save_folder

        self.plot_step         = 5
        self.num_channel       = self.input_dim[1]
        self.meta_data         = []


    def build_model(self):

        self.w1_channel   = 2
        self.w2_channel   = 4
        self.w3_channel   = 8

        self.input_data   = tf.placeholder(tf.float32, self.input_dim)
        self.ground_truth = tf.placeholder(tf.float32, self.output_dim)
        self.w1           = self.weight_variable([self.num_channel, 3, 3, 1, self.w1_channel])
        self.b1           = self.bias_variable([self.w1_channel])
        self.w2           = self.weight_variable([self.num_channel, 3, 3, self.w1_channel, self.w2_channel])
        self.b2           = self.bias_variable([self.w2_channel])
        self.w3           = self.weight_variable([self.num_channel, 3, 3, self.w2_channel, self.w3_channel])
        self.b3           = self.bias_variable([self.w3_channel])

        self.w_out        = self.weight_variable([self.num_channel, 3, 3, self.w3_channel, 1])
        self.b_out        = self.bias_variable([1])

        self.saver = tf.train.Saver()


        self.h1_conv      = tf.nn.selu(self.cconv_conv(self.input_data, self.w1) + self.b1)
        # self.h1_pool        = self.max_pool(self.h1_conv, [2, 2], [2, 2]) # output 14x14
        self.h2_conv      = tf.nn.selu(self.cconv_conv(self.h1_conv, self.w2) + self.b2)
        # self.h2_pool        = self.max_pool(self.h2_conv, [2, 2], [2, 2]) # output 7x7
        self.h3_conv      = tf.nn.selu(self.cconv_conv(self.h2_conv, self.w3) + self.b3)

        self.output       = tf.nn.selu(self.cconv_conv(self.h3_conv, self.w_out) + self.b_out)


        # Regression problem
        # self.loss           = tf.reduce_mean(tf.norm(self.output - self.ground_truth, axis=1, ord=2))

        # self.loss           = tf.nn.l2_loss(self.output - self.ground_truth)

        self.loss         = tf.reduce_mean(tf.pow(self.output - self.ground_truth, 2))

        # self.err_cost       =tf.reduce_mean(tf.sqrt(self.loss))

        # pdb.set_trace()

    def test(self, label, data):

        data=np.abs(data)
        label=np.abs(label)
        batch_num = data.shape[0] // self.batch_size

        avg_acc=0

        # aaa=np.array([[2,1,1], [0,1,1]], dtype=float)
        # bbb=np.array([[4,2,1], [3,1,1]], dtype=float)
        # diff=tf.norm(aaa-bbb)
        # ans=self.sess.run(diff)
        # print(ans)
        # pdb.set_trace()

        for idx in range(0, batch_num):
            start = idx * self.batch_size
            end = (idx+1) * self.batch_size
            pred_label = self.sess.run(self.output,
                    feed_dict={
                        self.input_data:   data[start:end, :],
                        self.ground_truth: label[start:end, :]
                    })
            pdb.set_trace()

            correct_pred=tf.equal(tf.argmax(pred_label, 1), tf.argmax(label[start:end], 1))
            acc=tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            avg_acc=avg_acc+(acc/batch_num)

        return self.sess.run(avg_acc)



def psnr(noise_img, clean_img):

    diff     = noise_img-clean_img
    mse      = diff*diff / (noise_img.shape[0]*noise_img.shape[1])
    psnr_val = 10*math.log ( (255.0*255.0/(mse.sum())) ,10)

    return psnr_val

if __name__=="__main__":

    # one_image = np.arange(16).reshape((4, 4))
    # # patch     = image.extract_patches_2d(one_image, (2, 2))
    # patch     = view_as_blocks(one, block_shape=(2, 2))
    # pdb.set_trace()

    # Load multidimensional dataset
    sparsity         = 0.15
    sigma            = 100
    dataPath         = "../dataset/complete_ms_data"

    PSNR_in          = 0

    training_set=[ 'face_ms',]

    for set_idx in range(len(training_set)):

        img_dataPath     = os.path.join(dataPath, training_set[set_idx])
        train_fileList   = recursiveFileList(img_dataPath, 'png')
        train_fileList   = train_fileList[25:30]

        for file_idx in range(np.shape(train_fileList)[0]):

            img                 = cv2.imread(train_fileList[file_idx]._path)
            img                 = np.float32(img[:,:,0])/255

            # Generate Gaussian noise
            mask                = np.zeros([1, img.shape[0]*img.shape[1]], dtype=float)
            n_idx               = np.random.choice(img.shape[0]*img.shape[1], np.int(img.shape[0]*img.shape[1]*sparsity))
            mask[0, n_idx]      = 1
            mask                = np.reshape(mask, [img.shape[0], img.shape[1]])

            gaussain_noise      = mask*sigma*np.random.randn(img.shape[0], img.shape[1])
            gaussian_noise_data = gaussain_noise + img


            PSNR_in             = PSNR_in + psnr(gaussian_noise_data, img)/10


            # overlap patches
            noise_patch         = image.extract_patches_2d(gaussian_noise_data, (32, 32))
            clean_patch         = image.extract_patches_2d(img, (32, 32))

            idx                 = np.arange(noise_patch.shape[0])
            idx                 = idx[np.mod(idx, 10)==0]
            noise_patch         = noise_patch[idx, :]
            clean_patch         = clean_patch[idx, :]

            dim                 = clean_patch.shape
            noise_patch         = np.reshape(noise_patch, [dim[0], 1, dim[1], dim[2], 1])
            clean_patch         = np.reshape(clean_patch, [dim[0], 1, dim[1], dim[2], 1])


            # # Non-overlap patches
            # noise_patch         = view_as_blocks(gaussian_noise_data, block_shape=(32, 32))
            # clean_patch         = view_as_blocks(img, block_shape=(32, 32))
            # dim                 = clean_patch.shape
            # noise_patch         = np.reshape(noise_patch, [dim[0]*dim[1], 1, dim[2], dim[3], 1])
            # clean_patch         = np.reshape(clean_patch, [dim[0]*dim[1], 1, dim[2], dim[3], 1])

            if file_idx == 0:
                train_data   = noise_patch
                train_label  = clean_patch
            else:
                train_data   = np.concatenate((train_data, noise_patch), axis=1)
                train_label  = np.concatenate((train_label, clean_patch), axis=1)

            print(train_fileList[file_idx]._path)



    modelSaveFolder='tmp_model/'

    # # Train a model
    with tf.Session() as sess:
        # Model initialization
        model=Img_denoise(sess, \
                learnRate=0.001, shuffle=True, batch_size=200, epoch=500, \
                input_dim=list(np.shape(train_data)), output_dim=list(np.shape(train_label)), \
                showPlot=True, main_save_folder=modelSaveFolder)

        model.build_model()

        s_time=time.time()
        model.train(train_label, train_data)
        e_time=time.time()

        model.save_model(modelSaveFolder)

    print('------ Training time: {}'.format((e_time-s_time)))



    # # Test a Model
    # with tf.Session() as sess:
    #     model=Img_denoise(sess, \
    #             learnRate=0.001, shuffle=True, batch_size=2000, epoch=500, \
    #             input_dim=list(np.shape(train_data)), output_dim=list(np.shape(train_label)), \
    #             showPlot=True)
    #
    #     model.build_model()
    #     model.load_model(modelSaveFolder)
    #     avgAcc=model.test(train_label, train_data)
    #
    # print('Avg accuracy is: ', avgAcc)
