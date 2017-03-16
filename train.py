import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from data_fetch import preprocess_and_load
from model import SoundCNN


def train_conv_net(max_iter, batch_size, num_classes, learning_rate, trainX, trainYa, valX, valY, testX, testY):
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
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
                learning_rate = 1e-4
                if 0 < iterations < max_iter / 2:
                    learning_rate = 1e-2
                if max_iter / 2 <= iterations < max_iter - max_iter / 4:
                    learning_rate = 1e-3
                sess.run(model.train_step,
                         feed_dict={model.x: batch_x, model.y_: batch_y, model.keep_prob: 0.5,
                                    model.is_train: True,
                                    model.learning_rate: learning_rate})
                if iterations % 5 == 0:
                    train_accuracy = model.accuracy.eval(session=sess,
                                                         feed_dict={model.x: batch_x,
                                                                    model.y_: batch_y,
                                                                    model.keep_prob: 1.0,
                                                                    model.is_train: False})
                    train_loss = model.cross_entropy.eval(session=sess,
                                                          feed_dict={model.x: batch_x,
                                                                     model.y_: batch_y,
                                                                     model.keep_prob: 1.0,
                                                                     model.is_train: False})
                    train_accuracies.append(train_accuracy)
                    train_losses.append(train_loss)
                    print("Step %d, Training accuracy: %g, Loss %s" % (iterations, train_accuracy, train_loss))
                if iterations % 50 == 0:
                    val_accuracy = model.accuracy.eval(session=sess, feed_dict={model.x: valX,
                                                                                model.y_: valY,
                                                                                model.keep_prob: 1.0,
                                                                                model.is_train: False,
                                                                                })
                    val_loss = model.cross_entropy.eval(session=sess,
                                                        feed_dict={model.x: batch_x,
                                                                   model.y_: batch_y,
                                                                   model.keep_prob: 1.0,
                                                                   model.is_train: False})
                    val_accuracies.append(val_accuracy)
                    val_losses.append(val_loss)
                    print("Step %d, Validation accuracy: %g, Loss %s" % (iterations, val_accuracy, val_loss))
                iterations += 1
        test_accuracy = model.accuracy.eval(session=sess,
                                            feed_dict={model.x: testX,
                                                       model.y_: testY,
                                                       model.keep_prob: 1.0,
                                                       model.is_train: False})
        print("Test accuracy: %g" % test_accuracy)
        save_path = saver.save(sess, "./model.ckpt")
        # plt.figure()
        # plt.plot(train_accuracies)
        # plt.figure()
        # plt.plot(val_accuracies)
        # plt.show()
        # plt.plot(train_losses)
        # plt.show()
        # plt.plot(val_losses)
        # plt.show()
        print(train_accuracies)
        print(val_accuracies)
        print(train_losses)
        print(val_losses)


if __name__ == '__main__':
    num_classes, trainX, trainYa, valX, valY, testX, testY = preprocess_and_load('sorted_sound/', data_limit=5,used_genders=['male'])
    valX = [[a] for a in valX]
    testX = [[a] for a in testX]

    import numpy as np

    # fake_input_8 = np.zeros([1025, 130])
    # fake_input_8 = np.zeros([130, 1025])
    # line1 = np.ones([130]) * 2
    # fake_input_8[:, 8] = line1
    # fake_input_300 = np.zeros([130, 1025])
    # line2 = np.ones([130]) * 2
    # fake_input_300[:, 300] = line2
    # trainX = np.ones([2, 130, 1025])
    # trainX[0, :, :] = fake_input_8
    # trainX[1, :, :] = fake_input_300
    # trainYa = [[0, 1], [1, 0]]
    # valX = trainX
    # valY = trainYa
    # testX = trainX
    # testY = trainYa
    # num_classes = 2
    # valX = [[a] for a in valX]
    # testX = [[a] for a in testX]
    train_conv_net(max_iter=500, batch_size=2, num_classes=num_classes, learning_rate=1e-2, trainX=trainX,
                   trainYa=trainYa, valX=valX, valY=valY, testX=testX, testY=testY)
