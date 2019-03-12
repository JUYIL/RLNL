import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import math

# read = tf.train.NewCheckpointReader("./nodemodel/nodemodel.ckpt")
# a = read.get_tensor('conv/weights')
# b = read.get_tensor('conv/bias')
# print(a, b)

# linkread=tf.train.NewCheckpointReader("./linkmodel/linkmodel.ckpt")
# a=linkread.get_tensor('conv/weights')
# b=linkread.get_tensor('conv/bias')
# print(a, b)


# print(os.system("cd /home/lm/ns_allinone_3_27/ns ; echo '123' | sudo -S  /home/lm/PycharmProjects/hello/ns_allinone_3_27/ns/waf --run mycc"))

# print(os.system("cd /home/lm/ns_allinone_3_27/ns/examples/tutorial ; echo '123' | sudo -S  /home/lm/PycharmProjects/hello/ns_allinone_3_27/ns/waf --pyrun examples/tutorial/my.py"))
