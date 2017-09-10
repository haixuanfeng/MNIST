#coding:utf-8
import numpy
from PIL import Image
import tensorflow as tf

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.
EVAL_BATCH_SIZE=1

img = Image.open("1.png").convert('L')
if img.size[0]!=28 or img.size[1]!=28:
        img= img.resize((28,28),Image.ANTIALIAS)
arr=[]
for i in xrange(28):
    for j in xrange(28):
            pixel = 1.0-float(img.getpixel((j,i)))/255.0

            arr.append(pixel)
arr=numpy.array(arr).reshape(1,28,28,1)


#神经网络

eval_data = tf.placeholder(
      tf.float32,
      shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

# 下面这些变量是网络的可训练权值
# conv1 权值维度为 32 x channels x 5 x 5, 32 为特征图数目
conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
# conv1 偏置
conv1_biases = tf.Variable(tf.zeros([32]))
# conv2 权值维度为 64 x 32 x 5 x 5
conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 64],
                          stddev=0.1,
                          seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
# 全连接层 fc1 权值，神经元数目为512
fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
          stddev=0.1,
          seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
# fc2 权值，维度与标签类数目一致
fc2_weights = tf.Variable(
      tf.truncated_normal([512, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

# 两个网络：训练网络和评测网络
# 它们共享权值

# 实现 LeNet-5 模型，该函数输入为数据，输出为fc2的响应
# 第二个参数区分训练网络还是评测网络
def model(data, train=False):
 """The Model definition."""
 # 二维卷积，使用“不变形”补零（即输出特征图与输入尺寸一致）。
 conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
 # 加偏置、过激活函数一块完成
 relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
 # 最大值下采样
 pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
 # 第二个卷积层
 conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
 relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
 pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
 # 特征图变形为2维矩阵，便于送入全连接层
 pool_shape = pool.get_shape().as_list()
 reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
 # 全连接层，注意“+”运算自动广播偏置
 hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
 # 训练阶段，增加 50% dropout；而评测阶段无需该操作

 return tf.matmul(hidden, fc2_weights) + fc2_biases

eval_prediction = tf.nn.softmax(model(eval_data))
print eval_prediction


def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    predictions = numpy.ndarray(shape=(size,NUM_LABELS), dtype=numpy.float32)
    #end = begin + EVAL_BATCH_SIZE

    predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data})

    return predictions

with tf.Session() as sess:
    #tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph("./model/my_model.ckpt-8593.meta")
    #model_file = tf.train.latest_checkpoint("./model/checkpoint")
    saver.restore(sess,"./model/my_model.ckpt-8593")
    predictions = eval_in_batches(arr,sess)
    prediction = numpy.argmax(predictions[0])
    print prediction
