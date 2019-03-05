#nodepolicy=nodetrain(50, 10, 30) 2018/02/27
# RLN   rec:0.1515  re:72127.83843000006   co:160394.97538499985  rc:0.4496888899223302  nu:0.15374800861348137 lu:0.09680522472645106
# linktrain failed and RLNL failed

#nodepolicy=nodetrain(10, 10, 30) 2018/02/27
# RLN   0.1795  91836.10234 118945.05815 0.7720884227944529  	0.1179 0.0520006
# linktrain(10,10,30) RLNL 55126     0.207   0.7034400519832924  0.13819189 0.074741

#nodepolicy=nodetrain(100, 10, 30) 2018/02/28
# RLN 0.1795 0.77 0.11 0.05
# linktrain(10,10,30) RLNL 0.207 0.70 0.13 0.07

#new nodepolicy 10
# RLN 0.3845  0.5088150393206318  	0.24096158401880854 	0.22385629194839932

import time
from train import *
from onlinemap import *


# def train(n):
#     recs, rcs=[], []
#     while n<31:
#         nodepolicy = nodetrain(n, 10, 30)
#         nodesaver = tf.train.Saver()
#         nodesaver.save(nodepolicy.sess, "./nodemodel/nodemodel.ckpt")
#         nof(n)
#         rec, rc = RLN()
#         recs.append(rec)
#         rcs.append(rc)
#         n+=10
#     plt.figure(1)
#     x=[i for i in range(10,40,10)]
#     plt.plot(x, recs)
#     plt.xlabel("eponum", fontsize=12)
#     plt.ylabel('acceptance', fontsize=12)
#     plt.title('acp change ', fontsize=15)
#     plt.savefig('Results/acp.jpg')
#     plt.figure(2)
#     x = [i for i in range(10,40,10)]
#     plt.plot(x, rcs)
#     plt.xlabel("eponum", fontsize=12)
#     plt.ylabel('rc', fontsize=12)
#     plt.title('rc change ', fontsize=15)
#     plt.savefig('Results/rc.jpg')





# def train(n):
#     linkpolicy=linktrain(n, 10, 30)
#     linksaver=tf.train.Saver()
#     linksaver.save(linkpolicy.sess, "./linkmodel/linkmodel.ckpt")
#     lif(n)
#     rec, rc = RLNL()
#     print(rec,rc)
#


def train(n):
    trs=get_training_set(1000)
    nodep=NodePolicy(sub1,sub1.number_of_nodes(),5,n,100)
    nodep.train(trs)
    nodesaver = tf.train.Saver()
    nodesaver.save(nodep.sess, "./nodemodel/nodemodel.ckpt")
    nof(n)
    rec, rc = RLN()
    print(rec,rc)

train(10)

