import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

d = np.array([range(10), range(10)], dtype=np.float32)

x = tf.get_variable(name="x", initializer=d)
sess.run(tf.initialize_all_variables())

y = x.eval()

y[1, 0] = 2.
x = tf.assign(x, y)

print(sess.run(x))