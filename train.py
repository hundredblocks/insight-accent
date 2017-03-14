import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from data_fetch import preprocess, preprocess_and_load
from model import SoundCNN

# num_classes, trainX, trainYa, valX, valY, testX, testY = preprocess('organized_sound/wav/')
num_classes, trainX, trainYa, valX, valY, testX, testY = preprocess_and_load('sorted_sound/', data_limit=50,
                                                                             used_genders=['male'])

n = [np.copy(valX[0])]
valX = [n]

n = [np.copy(testX[0])]
testX = [n]
train_accuracies = []
val_accuracies = []


def train_conv_net(max_iter, batch_size):
    model = SoundCNN(num_classes)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        iterations = 0
        training_set = [[trainX[i, :, :], trainYa[i]] for i in range(len(trainYa))]

        while iterations < max_iter:
            perms = np.random.permutation(training_set)

            for i in range(len(training_set) / batch_size):
                batch = perms[i * batch_size:(i + 1) * batch_size, :]
                batch_x = [a[0] for a in batch]
                batch_x = [[a] for a in batch_x]
                batch_y = [a[1] for a in batch]
                # batch_y = to_one_hot(batch_y_nat)
                sess.run(model.train_step, feed_dict={model.x: batch_x, model.y_: batch_y, model.keep_prob: 0.5})
                if iterations % 50 == 0:
                    train_accuracy = model.accuracy.eval(session=sess,
                                                         feed_dict={model.x: batch_x, model.y_: batch_y,
                                                                    model.keep_prob: 1.0,
                                                                    model.is_train: True})
                    train_accuracies.append(train_accuracy)
                    print("Step %d, Training accuracy: %g" % (iterations, train_accuracy))
                if iterations % 500 == 0:
                    val_accuracy = model.accuracy.eval(session=sess, feed_dict={model.x: valX, model.y_: valY,
                                                                                model.keep_prob: 1.0,
                                                                                model.is_train: False})
                    val_accuracies.append(val_accuracy)
                    print("Step %d, Validation accuracy: %g" % (iterations, val_accuracy))
                iterations += 1
        test_accuracy = model.accuracy.eval(session=sess,
                                            feed_dict={model.x: testX, model.y_: testY, model.keep_prob: 1.0,
                                                       model.is_train: False})
        print("Test accuracy: %g" % (test_accuracy))
        save_path = saver.save(sess, "./model.ckpt")
        # plt.figure()
        # plt.plot(train_accuracies)
        # plt.show()
        print(train_accuracies)
        print(val_accuracies)

if __name__ == '__main__':
    train_conv_net(500, 5)
