from nets.vgg import vgg_16, vgg_19, vgg_arg_scope
from nets.resnet_v1 import resnet_v1_50, resnet_v1_101, resnet_v1_152, resnet_arg_scope
from nets.resnet_v2 import resnet_v2_50, resnet_v2_101, resnet_v2_152
from nets.inception_v1 import inception_v1, inception_v1_arg_scope
from nets.inception_v2 import inception_v2, inception_v2_arg_scope
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
from nets.inception_v4 import inception_v4, inception_v4_arg_scope
from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import tensorflow.contrib.slim as slim
import tensorflow as tf
import cv2


def load_imagenet_names(file_name):
    names = {}
    with open(file_name) as f:
        for id_, name in enumerate(f):
            names[id_] = name.split('\n')[0]
    return names


def test_vgg_16(img_dir):
    """
    Test VGG-16 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(vgg_arg_scope()):
        _, _ = vgg_16(inputs, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/vgg_16.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred + 1]

    print('Result of VGG-16:', name, prob)
    return name, prob


def test_vgg_19(img_dir):
    """
    Test VGG-19 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(vgg_arg_scope()):
        _, _ = vgg_19(inputs, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/vgg_19.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('vgg_19/fc8/squeezed:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred + 1]

    print('Result of VGG-19:', name, prob)
    return name, prob


def test_resnet_v1_50(img_dir):
    """
    Test ResNet-V1-50 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(resnet_arg_scope()):
        _, _ = resnet_v1_50(inputs, 1000, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/resnet_v1_50.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('resnet_v1_50/SpatialSqueeze:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred + 1]

    print('Result of ResNet-V1-50:', name, prob)
    return name, prob


def test_resnet_v1_101(img_dir):
    """
    Test ResNet-V1-101 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(resnet_arg_scope()):
        _, _ = resnet_v1_101(inputs, 1000, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/resnet_v1_101.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('resnet_v1_101/SpatialSqueeze:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred + 1]

    print('Result of ResNet-V1-101:', name, prob)
    return name, prob


def test_resnet_v1_152(img_dir):
    """
    Test ResNet-V1-152 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(resnet_arg_scope()):
        _, _ = resnet_v1_152(inputs, 1000, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/resnet_v1_152.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('resnet_v1_152/SpatialSqueeze:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred + 1]

    print('Result of ResNet-V1-152:', name, prob)
    return name, prob


def test_resnet_v2_50(img_dir):
    """
    Test ResNet-V1-50 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) / 255
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(resnet_arg_scope()):
        _, _ = resnet_v2_50(inputs, 1001, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/resnet_v2_50.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('resnet_v2_50/SpatialSqueeze:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred]

    print('Result of ResNet-V1-50:', name, prob)
    return name, prob


def test_resnet_v2_101(img_dir):
    """
    Test ResNet-V1-101 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) / 255
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(resnet_arg_scope()):
        _, _ = resnet_v2_101(inputs, 1001, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/resnet_v2_101.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('resnet_v2_101/SpatialSqueeze:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred]

    print('Result of ResNet-V1-101:', name, prob)
    return name, prob


def test_resnet_v2_152(img_dir):
    """
    Test ResNet-V1-152 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) / 255
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(resnet_arg_scope()):
        _, _ = resnet_v2_152(inputs, 1001, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/resnet_v2_152.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('resnet_v2_152/SpatialSqueeze:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred]

    print('Result of ResNet-V1-152:', name, prob)
    return name, prob


def test_inception_v1(img_dir):
    """
    Test Inception-V1 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) / 255
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(inception_v1_arg_scope()):
        _, _ = inception_v1(inputs, 1001, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/inception_v1.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('InceptionV1/Logits/SpatialSqueeze:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred]

    print('Result of Inception-V1:', name, prob)
    return name, prob


def test_inception_v2(img_dir):
    """
    Test Inception-V2 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) / 255
    img = img.reshape((1, 224, 224, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 224, 224, 3], dtype=tf.float32)
    with slim.arg_scope(inception_v2_arg_scope()):
        _, _ = inception_v2(inputs, 1001, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/inception_v2.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('InceptionV2/Logits/SpatialSqueeze:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred]

    print('Result of Inception-V2:', name, prob)
    return name, prob


def test_inception_v3(img_dir):
    """
    Test Inception-V3 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299)) / 255
    img = img.reshape((1, 299, 299, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 299, 299, 3], dtype=tf.float32)
    with slim.arg_scope(inception_v3_arg_scope()):
        _, _ = inception_v3(inputs, 1001, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/inception_v3.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
        pred = tf.argmax(tf.nn.softmax(outputs), axis=1)[0]
        prob = tf.reduce_max(tf.nn.softmax(outputs), axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred]

    print('Result of Inception-V3:', name, prob)
    return name, prob


def test_inception_v4(img_dir):
    """
    Test Inception-V4 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299)) / 255
    img = img.reshape((1, 299, 299, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 299, 299, 3], dtype=tf.float32)
    with slim.arg_scope(inception_v4_arg_scope()):
        _, _ = inception_v4(inputs, 1001, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/inception_v4.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')
        pred = tf.argmax(outputs, axis=1)[0]
        prob = tf.reduce_max(outputs, axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred]

    print('Result of Inception-V4:', name, prob)
    return name, prob


def test_inception_resnet_v2(img_dir):
    """
    Test Inception-ResNet-V2 with a single image.
    :param img_dir: Path of the image to be classified
    :return: classification result and probability of a single image
    """
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299)) / 255
    img = img.reshape((1, 299, 299, 3))

    tf.reset_default_graph()
    inputs = tf.placeholder(name='input_images', shape=[None, 299, 299, 3], dtype=tf.float32)
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        _, _ = inception_resnet_v2(inputs, 1001, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './models/inception_resnet_v2.ckpt')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        outputs = sess.graph.get_tensor_by_name('InceptionResnetV2/Logits/Predictions:0')
        pred = tf.argmax(outputs, axis=1)[0]
        prob = tf.reduce_max(outputs, axis=1)[0]

        pred, prob = sess.run([pred, prob], feed_dict={inputs: img})
        name = label_dict[pred]

    print('Result of Inception-ResNet-V2:', name, prob)
    return name, prob


if __name__ == '__main__':
    label_dict = load_imagenet_names('imagenet.names')
    test_vgg_16('cock.jpg')
    test_vgg_19('cock.jpg')
    test_resnet_v1_50('cock.jpg')
    test_resnet_v1_101('cock.jpg')
    test_resnet_v1_152('cock.jpg')
    test_resnet_v2_50('cock.jpg')
    test_resnet_v2_101('cock.jpg')
    test_resnet_v2_152('cock.jpg')
    test_inception_v1('cock.jpg')
    test_inception_v2('cock.jpg')
    test_inception_v3('cock.jpg')
    test_inception_v4('cock.jpg')
    test_inception_resnet_v2('cock.jpg')
