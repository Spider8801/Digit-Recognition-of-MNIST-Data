import os
import numpy as np
import tensorflow.compat.v1 as tf
from model import alexnet
from evals import calc_loss_acc, train_op
from tensorboard.plugins.hparams import api as hp
import input_data
from scipy import misc
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_integer('valid_steps', 11, 'The number of validation steps ')

flags.DEFINE_integer('max_steps', 300, 'The number of maximum steps for traing')

flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch during training')

flags.DEFINE_float('base_learning_rate', 0.0001, "base learning rate for optimizer")

flags.DEFINE_integer('input_shape', 784 , 'The inputs tensor shape')
##flags.DEFINE_integer('input_shape', 784, 'The inputs tensor shape')

flags.DEFINE_integer('num_classes', 10, 'The number of label classes')

flags.DEFINE_string('save_dir', './outputs', 'The path to saved checkpoints')

flags.DEFINE_float('keep_prob', 0.75, "the probability of keeping neuron unit")

flags.DEFINE_string('tb_path', './tb_logs/First/', 'The path points to tensorboard logs ')





def train(FLAGS):
    """Training model

    """
    valid_steps = FLAGS.valid_steps
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    base_learning_rate = FLAGS.base_learning_rate
    input_shape = FLAGS.input_shape  # image shape = 28 * 28
    num_classes = FLAGS.num_classes
    keep_prob = FLAGS.keep_prob
    save_dir = FLAGS.save_dir
    tb_path = FLAGS.tb_path 

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    tf.reset_default_graph()
    # define default tensor graphe 
    with tf.Graph().as_default():
        images_pl = tf.placeholder(tf.float32, shape=[None, input_shape])
        labels_pl = tf.placeholder(tf.float32, shape=[None, num_classes])

        # define a variable global_steps
        global_steps = tf.Variable(0, trainable=False)

        # build a graph that calculate the logits prediction from model
        logits = alexnet(images_pl, num_classes, keep_prob)

        loss, acc, _ = calc_loss_acc(labels_pl, logits)

        # build a graph that trains the model with one batch of example and updates the model params 
        training_op = train_op(loss, global_steps, base_learning_rate)
        validing_op = train_op(loss, global_steps, base_learning_rate)
        # define the model saver
        saver = tf.train.Saver(tf.global_variables())
        
        # define a summary operation 
        summary_op = tf.summary.merge_all()
        summ_op=tf.summary.merge_all()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # print(sess.run(tf.trainable_variables()))
            # start queue runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            train_writter = tf.summary.FileWriter(tb_path, sess.graph)
            train_writter2 = tf.summary.FileWriter('./tb_logs/Second/', sess.graph)
            #train_writter = tf.summary.create_file_writer(tb_path)

            # start training
            for step in range(max_steps):

                train_image_batch, train_label_batch = mnist.train.next_batch(batch_size)
                train_feed_dict = {images_pl: train_image_batch, labels_pl: train_label_batch}

                _, _loss, _acc, _summary_op = sess.run([training_op, loss, acc, summary_op], feed_dict = train_feed_dict)

                # store loss and accuracy value
                train_loss.append(_loss)
                train_acc.append(_acc)
                print("Iteration " + str(step) + ", Mini-batch Loss= " + "{:.6f}".format(_loss) + ", Training Accuracy= " + "{:.5f}".format(_acc))
                train_writter.add_summary(_summary_op, global_step= step)
                print("brrr",step)
                if step % 100 == 0:
                    _valid_loss, _valid_acc = [], []
                    print('Start validation process')

                    for itr in range(valid_steps):
                        valid_image_batch, valid_label_batch = mnist.test.next_batch(batch_size)

                        valid_feed_dict = {images_pl: valid_image_batch, labels_pl: valid_label_batch}

                        _,_loss, _acc,_validing_op = sess.run([validing_op,loss, acc,summ_op], feed_dict = valid_feed_dict)
                        train_writter2.add_summary(_validing_op, global_step= itr)

                        _valid_loss.append(_loss)
                        _valid_acc.append(_acc)
                        #train_writter.add_summary(_summary_op, global_step= step)

                    valid_loss.append(np.mean(_valid_loss))
                    valid_acc.append(np.mean(_valid_acc))

                    #print("Iteration {}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(itr, train_loss[-1], train_acc[-1], valid_loss[-1], valid_acc[-1]))
                    print("Iteration {}: Train Loss {}, Train Acc {}, Val Loss {}, Val Acc {}",itr, train_loss[-1], train_acc[-1], valid_loss[-1], valid_acc[-1])
            
                    #train_writter.add_summary(_summary_op, global_step= step)
                    #print("brrr",step)
                    #train_writter.
                    #with train_writter.as_default():
                    #tf.summary.scalar('loss', _valid_loss)
                    #tf.summary.scalar('accuracy', _valid_acc)


            np.save(os.path.join(save_dir, 'accuracy_loss', 'train_loss'), train_loss)
            np.save(os.path.join(save_dir, 'accuracy_loss', 'train_acc'), train_acc)
            np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_loss'), valid_loss)
            np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_acc'), valid_acc)
            checkpoint_path = os.path.join(save_dir, 'model', 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)
    sess.close()
if __name__ == '__main__':
    train(FLAGS)


                # get train image / label batch
                #training_data = np.array([image.reshape(28, 28, 1) for image in mnist.train.images])
                #training_label = mnist.train.labels
                #k=np.array([image.reshape(28, 28, 1) for image in mnist.train.next_batch(batch_size)])
                #mnist=k
                #train = {'X': resize_images(mnist.train.images.reshape(-1, 28, 28)),
                 #'y': mnist.train.labels}
                #scale_percent = 60 # percent of original size
                #width = int(img.shape[1] * scale_percent / 100)
                #height = int(img.shape[0] * scale_percent / 100)
                #dim = (width, height)
                # resize image
                #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                
"""
                dim=(batch_size,1600)
                #train_image_batch=np.resize(mnist.train.images,dim)
                
                for image in mnist.train.images:
                    train_image_batch=np.resize(image,dim)
                
                train_data=[]
                for image in mnist.train.images:
                    resized_img=np.resize(image,dim)
                    train_data.append(resized_img)
                 """   
                ##train_dataa=train_data.next_batch(batch_size)
                #train_image_batch=misc.imresize(mnist.train.images,dim)
                #train_label_batch=mnist.train.labels
                
                                #dim=(40,40)
"""
                train_data=[]
                for image in mnist.train.images:
                    resized_img=np.resize(image,dim)
                    train_data.append(resized_img)
                """
                #tf.image.resize([train_image_batch,28,28,128], dim,name=None)
                
                #FLAGS = flags.FLAGS


#dim=(40,40)
"""
train_data=[]
for image in mnist.train.images:
    resized_img=np.resize(image,dim)
    train_data.append(resized_img)
"""
#tf.image.resize([mnist.train.images,28,28,55000], dim,name=None)