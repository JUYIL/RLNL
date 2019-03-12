

#new nodepolicy 10
# RLN 0.3845  0.5088150393206318  	0.24096158401880854 	0.22385629194839932

#new nodepolicy 50
# RLN 0.36 0.5 0.23 0.21

# com1 nodepolicy 10
# RLN 0.616   0.8145042073466895  	0.56277633633813    	0.2089386380405062
#new nodepolicy 1
# 0.359  0.7720884227944529  	0.23586201305238322 	0.10413807585350601


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
train(10)

# def comp1train(n):
#     trs=get_training_set(1000)
#     nodep=RL(sub1,sub1.number_of_nodes(),4,learning_rate=0.05,num_epoch=n,batch_size=100)
#     nodep.train(trs)
#     nodesaver = tf.train.Saver()
#     nodesaver.save(nodep.sess, "./nodemodel/cp1nodemodel.ckpt")
#     c1nof(n)
#     rec, rc = rl()
#     print(rec,rc)
#
# comp1train(100)
