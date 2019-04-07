import tensorflow as tf

"""
This makes it seem as though we can use py_function to do reentrant usage of a Session.
Makes sense since Sessions are supposed to be thread safe.  
"""

if __name__ == '__main__':
    def tf_func(sess):
        def func(x):
            with sess.graph.as_default():
                y = x.numpy()
                print("numpy",y)
                return sess.run(tf.constant(y))
        return func



    with tf.Session(graph=tf.Graph()) as sess:

        x = tf.constant(0.)

        p = tf.stack([tf.py_function(tf_func(sess),[x],[x.dtype]),
                      tf.py_function(tf_func(sess), [x+1.], [x.dtype]),
                      tf.py_function(tf_func(sess), [x+2.], [x.dtype])])
        print(sess.run(p))

