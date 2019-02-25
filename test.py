import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import struct

# read = tf.train.NewCheckpointReader("./nodemodel/nodemodel.ckpt")
# a = read.get_tensor('conv/weights')
# b = read.get_tensor('conv/bias')
# print(a, b)

#     linkread=tf.train.NewCheckpointReader("./linkmodel/linkmodel.ckpt")
#     a=linkread.get_tensor('conv/weights')
#     b=linkread.get_tensor('conv/bias')
#     print(a, b)