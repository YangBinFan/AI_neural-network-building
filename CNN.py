import tensorflow as tf

import numpy as np
import pickle
import time
import os
import pdb


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CNN:
    def __init__(self, sess,
                 learnRate=0.001,
                 shuffle=True,
                 batch_size=100,
                 epoch=1000,
                 input_dim=784,
                 output_dim=10,
                 showPlot=False,
                 main_save_folder='tmp_model/'):

        self.sess=sess

        self.learnRate=learnRate
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.epoch=epoch

        # self.input_dim=[None, width, height, num. of input channels]
        # self.output_dim=[None, # of classes]

        self.input_dim=input_dim    # [None, 28, 28, 1]
        self.output_dim=output_dim  # [None, 10]
        input_dim[0]=output_dim[0]=None

        self.showPlot=showPlot

        self.main_save_folder=main_save_folder

        self.plot_step=5
        self.meta_data=[];


    def build_model(self):
        self.input_data   = tf.placeholder(tf.float32, self.input_dim)
        self.ground_truth = tf.placeholder(tf.float32, self.output_dim)
        self.w1           = self.weight_variable([5, 5, 1, 16])
        self.b1           = self.bias_variable([16])
        self.w2           = self.weight_variable([5, 5, 16, 32])
        self.b2           = self.bias_variable([32])
        # self.w3           = self.weight_variable([5, 5, 32, 64])
        # self.b3           = self.bias_variable([64])
        self.w_fc         = self.weight_variable([7*7*32, 1024])
        self.b_fc         = self.bias_variable([1024])
        self.w_fc_out     = self.weight_variable([1024, 10])
        self.b_fc_out     = self.bias_variable([10])

        self.saver = tf.train.Saver()

        self.h1_conv        = tf.nn.relu(self.conv_2d(self.input_data, self.w1) + self.b1)
        self.h1_pool        = self.max_pool_2x2(self.h1_conv) # output 14x14
        self.h2_conv        = tf.nn.relu(self.conv_2d(self.h1_pool, self.w2) + self.b2)
        self.h2_pool        = self.max_pool_2x2(self.h2_conv) # output 7x7
        # self.h3_conv        = tf.nn.relu(self.conv_2d(self.h2_pool, self.w3) + self.b3)
        # self.h3_pool        = self.max_pool_2x2(self.h3_conv)
        self.h2_pool_flat   = tf.reshape(self.h2_pool, [-1, 7*7*32])
        self.h_fc           = tf.nn.relu(tf.matmul(self.h2_pool_flat, self.w_fc) + self.b_fc)
        self.output         = tf.nn.relu(tf.matmul(self.h_fc, self.w_fc_out) + self.b_fc_out)

        # Classification problem
        self.cross_entropy  = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.ground_truth)
        self.loss           = tf.reduce_mean(self.cross_entropy)

        # Regression problem
       # self.loss           = tf.reduce_mean(tf.norm(self.output - self.ground_truth, axis=1, ord=2))
       # self.loss           = tf.nn.l2_loss(self.output - self.ground_truth)

       # self.loss           = tf.pow(self.output - self.ground_truth, 2)
       # self.err_cost       =tf.reduce_mean(tf.sqrt(self.loss))

    def conv_2d(self, x, w):
        # stride [1, x_movement, y_movement, 1]
        # stride[0]=stridep[3]=1 (according to tensorflow rules)
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        # ksize: kernel size
        # stride=[1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def weight_variable(self, shape):
        # shape=[filter hight, filter width, input channels, output channels]
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def train(self, train_label, train_data, val_label=[], val_data=[]):

        optim=tf.train.AdamOptimizer(self.learnRate) \
                .minimize(loss=self.loss, var_list=tf.trainable_variables())

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        order = [i for i in range(0, train_data.shape[0])]
        batch_num = train_data.shape[0] // self.batch_size

        print('[*] Start training...')

        if self.shuffle:
            np.random.shuffle(order)


        bestLoss=float('inf')
        recTrainAvgLoss=[]
        recValAvgLoss=[]


        for e in range(0, self.epoch+1):

            print ("Epoch: ", e)

            train_data = train_data[order, :]
            train_label = train_label[order, :]

            for idx in range(0, batch_num):
                start = idx * self.batch_size
                end   = (idx+1) * self.batch_size
                input_batch = train_data[start:end, :]
                label_batch = train_label[start:end, :]

                optim.run(feed_dict={
                            self.input_data:   input_batch,
                            self.ground_truth: label_batch
                        })

#                 self.sess.run(optim, feed_dict={
#                             self.input_data:   input_batch,
#                             self.ground_truth: label_batch
#                         })

            if np.mod(e, self.plot_step) == 0:

                avgTrainLoss=self.cal_cost(train_label, train_data)
                recTrainAvgLoss=np.append(recTrainAvgLoss, avgTrainLoss)

                print("Avg. train loss", avgTrainLoss)

                # avg_acc=self.test(train_label, train_data)
                # print("Avg. train acc. ", avg_acc)

                if np.shape(val_data)[0]!=0:
                    avgValLoss=self.cal_cost(val_label, val_data)
                    recValAvgLoss=np.append(recValAvgLoss, avgValLoss)

                    print("Avg. val. loss", avgValLoss)


                if avgTrainLoss < bestLoss:
                    bestLoss=avgTrainLoss
                    self.save_model(self.main_save_folder+'_epoch_'+str(e)+'/')

                if self.showPlot:
                    self.show_plot(self.plot_step*np.arange(0, np.shape(recTrainAvgLoss)[0]), \
                     recTrainAvgLoss, recValAvgLoss)


        self.meta_data=(recTrainAvgLoss, recValAvgLoss)
        # print("[v] Training {} epochs completed".format(epoch))

    def test(self, label, data):

        data=np.abs(data)
        label=np.abs(label)
        batch_num = data.shape[0] // self.batch_size

        avg_acc=0

        for idx in range(0, batch_num):
            start = idx * self.batch_size
            end = (idx+1) * self.batch_size
            pred_label = self.sess.run(self.h1_conv,
                    feed_dict={
                        self.input_data:   data[start:end, :],
                        self.ground_truth: label[start:end, :]
                    })
            pdb.set_trace()
            correct_pred=tf.equal(tf.argmax(pred_label, 1), tf.argmax(label[start:end], 1))
            acc=tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            avg_acc=avg_acc+(acc/batch_num)

        return self.sess.run(avg_acc)


    def cal_cost(self, label, data):

        data=np.abs(data)
        label=np.abs(label)
        batch_num = data.shape[0] // self.batch_size

        avg_loss=0

        for idx in range(0, batch_num):
            start = idx * self.batch_size
            end = (idx+1) * self.batch_size
            err_cost = self.sess.run(self.loss,
                    feed_dict={
                        self.input_data:   data[start:end, :],
                        self.ground_truth: label[start:end, :]
                    })
            avg_loss=avg_loss+err_cost/batch_num

        return avg_loss


    def show_plot(self, epoch, train_y, val_y=[]):

        fig=plt.figure()

        if np.shape(val_y)[0]==0:
            hdl_train, =plt.plot(epoch, train_y, color='blue', linewidth=1.0, linestyle='--')
            plt.legend(handles=[hdl_train,], labels=['Training Data'], loc='best')
            plt.xlabel('Num. of Iteration')
            plt.ylabel('Error Cost')
            plt.title('Cost Trend')
            plt.grid()
        else:
            hdl_train, hdl_val=plt.plot(epoch, train_y, 'blue', \
                                        epoch,   val_y,  'red', \
                                        linewidth=1.0, linestyle='--')
            plt.legend(handles=[hdl_train, hdl_val], \
                        labels=['Training Data', 'Val. Data'], loc='best')
            plt.xlabel('Num. of Iteration', fontsize=14)
            plt.ylabel('Error Cost', fontsize=14)
            plt.title('Cost Trend', fontsize=14)
            plt.grid()

        fig.savefig('ErrorCost.png')
        # plt.pause(0.001)

    def save_model(self, save_path):

        print("[*] Saving model...")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.saver.save(self.sess, save_path+'model')

        with open(save_path+'meta.pickle', 'wb') as fout:
            pickle.dump(self.meta_data, fout)

        print("[v] Saved to {}.".format(save_path))

    def load_model(self, load_path):

        print("[*] Reading model {}...".format(load_path))

        self.saver.restore(self.sess, load_path+'model')

        with open(load_path+'meta.pickle', 'rb') as fin:
            self.meta_data=pickle.load(fin)

        print("[v] Model loaded.")

if __name__=="__main__":

    # Load mnist dataset
    (train_x, train_y), (test_x, test_y)=tf.keras.datasets.mnist.load_data(path='mnist.npz')

    train_x=np.float32(train_x)/255
    test_x=np.float32(test_x)/255

    # Transform labels into one-hot representation
    numOfTrainData, _, _=np.shape(train_x)
    one_hot_rpt=np.zeros([numOfTrainData, 10])
    one_hot_rpt[np.arange(numOfTrainData), train_y]=1;
    train_y=one_hot_rpt

    numOfTestData, _, _=np.shape(test_x)
    one_hot_rpt=np.zeros([numOfTestData, 10])
    one_hot_rpt[np.arange(numOfTestData), test_y]=1;
    test_y=one_hot_rpt

    train_input_dim=list(np.shape(train_x))
    train_output_dim=list(np.shape(train_y))

    train_x=train_x.reshape(numOfTrainData, train_input_dim[1], train_input_dim[2], 1)
    test_x=test_x.reshape(numOfTestData, train_input_dim[1], train_input_dim[2], 1)


    modelSaveFolder='tmp_model/'

    # # Train a model
    # with tf.Session() as sess:
    #     # Model initialization
    #     model=CNN(sess, \
    #             learnRate=0.000001, shuffle=True, batch_size=2000, epoch=200, \
    #             input_dim=list(np.shape(train_x)), output_dim=list(np.shape(train_y)), \
    #             showPlot=True, main_save_folder=modelSaveFolder)
    #
    #     model.build_model()
    #
    #     s_time=time.time()
    #     model.train(train_y, train_x, test_y, test_x)
    #     e_time=time.time()
    #
    #     model.save_model(modelSaveFolder)
    #
    # print('------ Training time: {}'.format((e_time-s_time)))



    # Test a Model
    with tf.Session() as sess:
        model=CNN(sess, \
                learnRate=0.000001, shuffle=True, batch_size=2000, epoch=10, \
                input_dim=list(np.shape(test_x)), output_dim=list(np.shape(test_y)), \
                showPlot=True)

        model.build_model()
        model.load_model(modelSaveFolder)
        avgAcc=model.test(test_y, test_x)

    print('Avg accuracy is: ', avgAcc)
