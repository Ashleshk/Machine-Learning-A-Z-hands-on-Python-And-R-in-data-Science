#SIMPLEST TENSORFLOW CODE
import tensorflow as tf

a = tf.Variable(1, name="a")
b = tf.Variable(2, name="b")
f = a + b

tf.print("The sum of a and b is", f)