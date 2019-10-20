import tensorflow as tf
from utils import batch
import numpy as np


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def matrix_add(ma, mb):
    tf.compat.v1.reset_default_graph()
    X = tf.compat.v1.placeholder(tf.float32, [len(ma), len(ma[0])])
    Y = tf.compat.v1.placeholder(tf.float32, [len(mb), len(mb[0])])

    m = tf.add(X, Y)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    x = sess.run(m, {X: ma, Y: mb})
    sess.close()
    tf.compat.v1.reset_default_graph()
    return x


def matrix_dot(ma, mb):
    tf.compat.v1.reset_default_graph()
    X = tf.compat.v1.placeholder(tf.float32, [len(ma), len(ma[0])])
    Y = tf.compat.v1.placeholder(tf.float32, [len(mb), len(mb[0])])

    m = tf.matmul(X, tf.transpose(Y))

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    x = sess.run(m, {X: ma, Y: mb})
    sess.close()
    tf.compat.v1.reset_default_graph()
    return x



def matrix_dot_batches(ma,mb,batch_size=10000):
    x = None

    for i_batch, mbatch in enumerate(batch(mb, batch_size)):
        if x is None:
            x = matrix_dot(ma, mbatch)
        else:
            x = np.concatenate((x, matrix_dot(ma, mbatch)), axis=1)

    return x


def k_top(ma, k):
    tf.compat.v1.reset_default_graph()
    X = tf.compat.v1.placeholder(tf.float32, [len(ma), len(ma[0])])

    _, indexes = tf.nn.top_k(ma, k)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    x = sess.run(indexes, {X: ma})
    sess.close()
    tf.compat.v1.reset_default_graph()
    return x


def cosine_knn(ma, mb, k):
    tf.compat.v1.reset_default_graph()
    X = tf.compat.v1.placeholder(tf.float32, [len(ma), len(ma[0])])
    Y = tf.compat.v1.placeholder(tf.float32, [len(mb), len(mb[0])])

    m = tf.matmul(X, tf.transpose(Y))
    _, indexes = tf.nn.top_k(m, k)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    x = sess.run(indexes, {X: ma, Y: mb})
    sess.close()
    tf.compat.v1.reset_default_graph()
    return x


def cosine_knn_batches(ma, mb, k, batch_size=10000):

    #mDot = matrix_dot_batches(ma, mb, batch_size=batch_size)

    x = None

    for i_batch, mbatch in enumerate(batch(mb, batch_size)):
        if x is None:
            x = matrix_dot(ma, mbatch)
        else:
            x = np.concatenate((x, matrix_dot(ma, mbatch)), axis=1)

    top = k_top(x, k)
    return top


def matrix_analogy(ma,mb,mc,mM):
    tf.compat.v1.reset_default_graph()
    a = tf.compat.v1.placeholder(tf.float32, [len(ma), len(ma[0])])
    b = tf.compat.v1.placeholder(tf.float32, [len(mb), len(mb[0])])
    c = tf.compat.v1.placeholder(tf.float32, [len(mc), len(mc[0])])
    M = tf.compat.v1.placeholder(tf.float32, [len(mM), len(mM[0])])

    ag = tf.add(tf.subtract(c, a), b)
    nn = tf.matmul(ag, tf.transpose(M))

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    x = sess.run(nn, {a: ma, b: mb, c: mc, M:mM})
    sess.close()
    tf.compat.v1.reset_default_graph()

    return x

