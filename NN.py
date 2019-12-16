import tensorflow as tf

import numpy as np
import pickle
import time
import os
import pdb


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NN:
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


        self.input_dim=input_dim
        self.output_dim=output_dim


        self.showPlot=showPlot

        self.main_save_folder=main_save_folder

        self.plot_step=5
        self.meta_data=[];


    def build_model(self):
        self.input_data   = tf.placeholder(tf.float32, [None, self.input_dim])
        self.ground_truth = tf.placeholder(tf.float32, [None, self.output_dim])
        self.w1           = tf.Variable(tf.truncated_normal([self.input_dim, 1024], stddev = 0.1), name='w1')
        # self.w1           = tf.get_variable("w1", shape=[self.input_dim, 1024], initializer=tf.contrib.layers.xavier_initializer())
        self.b1           = tf.Variable(tf.constant(0.1, shape=[1024]), name='b1')
        self.w2           = tf.Variable(tf.truncated_normal([1024, 1024], stddev = 0.1), name='w2')
        # self.w2           = tf.get_variable("w2", shape=[1024, 1024], initializer=tf.contrib.layers.xavier_initializer())
        self.b2           = tf.Variable(tf.constant(0.1, shape=[1024]), name='b2')
        self.w3           = tf.Variable(tf.truncated_normal([1024, 1024], stddev = 0.1), name='w3')
        # self.w3           = tf.get_variable("w4", shape=[1024, 1024], initializer=tf.contrib.layers.xavier_initializer())
        self.b3           = tf.Variable(tf.constant(0.1, shape=[1024]), name='b3')
        self.w_out        = tf.Variable(tf.truncated_normal([1024, self.output_dim], stddev = 0.1), name='w_out')
        # self.w_out        = tf.get_variable("w_out", shape=[1024, self.output_dim], initializer=tf.contrib.layers.xavier_initializer())
        self.b_out        = tf.Variable(tf.constant(0.1, shape=[self.output_dim]), name='b_out')

        self.saver = tf.train.Saver()

        self.h1             = tf.nn.sigmoid(tf.matmul(self.input_data, self.w1) + self.b1)
        self.h2             = tf.nn.sigmoid(tf.matmul(self.h1, self.w2) + self.b2)
        self.h3             = tf.nn.sigmoid(tf.matmul(self.h2, self.w3) + self.b3)
        self.output         = tf.nn.sigmoid(tf.matmul(self.h3, self.w_out) + self.b_out)

        # Classification problem
        self.cross_entropy  = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.ground_truth)
        self.loss           = tf.reduce_mean(self.cross_entropy)
        # Regression problem
       # self.loss           = tf.reduce_mean(tf.norm(self.output - self.ground_truth, axis=1, ord=2))
       # self.loss           = tf.nn.l2_loss(self.output - self.ground_truth)

       # self.loss           = tf.pow(self.output - self.ground_truth, 2)
       # self.err_cost       =tf.reduce_mean(tf.sqrt(self.loss))

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
            pred_label = self.sess.run(self.output,
                    feed_dict={
                        self.input_data:   data[start:end, :],
                        self.ground_truth: label[start:end, :]
                    })
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

    train_x=train_x.reshape(np.shape(train_x)[0], np.shape(train_x)[1]*np.shape(train_x)[2])
    test_x=test_x.reshape(np.shape(test_x)[0], np.shape(test_x)[1]*np.shape(test_x)[2])

    train_x=np.float32(train_x)/255
    test_x=np.float32(test_x)/255

    # Transform labels into one-hot representation
    numOftrainData, _=np.shape(train_x)
    one_hot_rpt=np.zeros([numOftrainData, 10])
    one_hot_rpt[np.arange(numOftrainData), train_y]=1;
    train_y=one_hot_rpt

    numOftestData, _=np.shape(test_x)
    one_hot_rpt=np.zeros([numOftestData, 10])
    one_hot_rpt[np.arange(numOftestData), test_y]=1;
    test_y=one_hot_rpt


    _, train_input_dim=np.shape(train_x)
    _, train_output_dim=np.shape(train_y)

    modelSaveFolder='tmp_model/'

    # Train a model
    with tf.Session() as sess:
        # Model initialization
        model=NN(sess, \
                learnRate=0.0001, shuffle=True, batch_size=2000, epoch=500, \
                input_dim=train_input_dim, output_dim=train_output_dim, \
                showPlot=True, main_save_folder=modelSaveFolder)

        model.build_model()

        s_time=time.time()
        model.train(train_y, train_x, test_y, test_x)
        e_time=time.time()

        model.save_model(modelSaveFolder)

    print('------ Training time: {}'.format((e_time-s_time)))



    # # Test a Model
    # with tf.Session() as sess:
    #     model=NN(sess, \
    #             learnRate=0.0001, shuffle=True, batch_size=1000, epoch=10, \
    #             input_dim=train_input_dim, output_dim=train_output_dim, \
    #             showPlot=True)
    #
    #     model.build_model()
    #     model.load_model(modelSaveFolder)
    #     avgAcc=model.test(test_y, test_x)
    #
    # print('Avg accuracy is: ', avgAcc)
