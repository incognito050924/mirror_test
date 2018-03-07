import os
import tensorflow as tf
from .models.inception_v3 import inception_v3
from .utils.pre_processing import preprocess_input
from ml import inception_model_v3 as inception

save_dir = os.path.join('/', 'Users', 'incognito', 'dev', 'datasets', 'train_dir')


def predict_emotion_old(img, text_label=True, checkpoint=None):
    x_data = preprocess_input(img)
    num_classes = 7
    if checkpoint is None:
        checkpoint = os.path.join(save_dir, 'model.ckpt-112728')
    print('Initialize trainable weights using checkpoint file: %s' % checkpoint)

    # Variables
    x = tf.placeholder(tf.float32, [None, 128, 128, 1])
    y = tf.placeholder(tf.float32, [None, num_classes])
    is_training = tf.placeholder(tf.bool)

    # Prediction
    prediction, end_points = inception_v3(x, num_classes=num_classes, is_training=is_training)
    # Loss & Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    # Initialize Variables
    init = tf.global_variables_initializer()

    # Saver
    saver = tf.train.Saver(max_to_keep=100)

    # Generate Session & Build Graph
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, os.path.join(save_dir, checkpoint))
        preds = sess.run(tf.argmax(prediction, 1), feed_dict={x: x_data, is_training: False})

    if text_label:
        text_ground_truth = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        pred = text_ground_truth[preds[0]]

    if text_label:
        emotion_text2score = {'angry': 0, 'sad': 1, 'disgust': 2, 'fear': 3,
                              'surprise': 4, 'neutral': 5, 'happy': 6}
        order = emotion_text2score[pred]
    else:
        emotion_logits2score = {0: 0, 1: 2, 2: 3, 3: 6, 4: 1, 5: 4, 6: 5}
        order = emotion_logits2score[preds[0]]

    return pred, order


def predict_emotion(img, text_label=True):
    x_data = preprocess_input(img)

    x = tf.placeholder(tf.float32, [None, 299, 299, 1])

    logits, endpoints = inception.inference(x, 8)
    prediction = tf.nn.softmax(logits, 1)

    # Initialize Variables
    init = tf.global_variables_initializer()

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    # Saver
    saver = tf.train.Saver(variables_to_restore)

    # Generate Session & Build Graph
    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, os.path.join(save_dir, 'model.ckpt-15000'))

        preds = sess.run(tf.argmax(prediction, 1), feed_dict={x: x_data})
        print(preds, type(preds))

    if text_label:
        text_ground_truth = ['unknown', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        pred = text_ground_truth[preds[0]]

    if text_label:
        emotion_text2score = {'angry': 0, 'sad': 1, 'disgust': 2, 'fear': 3,
                              'surprise': 4, 'neutral': 5, 'happy': 6}
        order = emotion_text2score[pred]
    else:
        emotion_logits2score = {0: 0, 1: 1, 2: 4, 3: 7, 4: 2, 5: 5, 6: 6}
        order = emotion_logits2score[preds[0]]
    print(pred, order)
    return pred, order
