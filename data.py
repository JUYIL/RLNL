
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
from compare1 import *







def train(n):
    trs=get_training_set(1000)
    linkp=LinkPolicy(sub1,len(getallpath(sub1)),2,n,100)
    linkp.train(trs)
    linksaver=tf.train.Saver()
    linksaver.save(linkp.sess, "./linkmodel/linkmodel.ckpt")
    lif(n)
    rec, rc = RLNL()
    print(rec,rc)



# def train(n):
#     trs=get_training_set(1000)
#     nodep=NodePolicy(sub1,sub1.number_of_nodes(),5,n,100)
#     nodep.train(trs)
#     nodesaver = tf.train.Saver()
#     nodesaver.save(nodep.sess, "./nodemodel/nodemodel.ckpt")
#     nof(n)
#     rec, rc = RLN()
#     print(rec,rc)
# train(10)

# def comp1train(n):
#     trs=get_training_set(1000)
#     nodep=RL(sub1,sub1.number_of_nodes(),4,learning_rate=0.05,num_epoch=1,batch_size=100)
#     nodep.train(trs)
#     nodesaver = tf.train.Saver()
#     nodesaver.save(nodep.sess, "./nodemodel/cp1nodemodel.ckpt")
#     nof(n)
#     rec, rc = RLN()
#     print(rec,rc)


rec, rc = RLNL()
print(rec,rc)