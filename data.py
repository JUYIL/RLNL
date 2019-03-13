#new nodepolicy 10 1
# 0.654  0.823961394505712   	0.5879847293853341  	0.22512982974686352
#new nodepolicy 10
# 0.3825   0.4766732825651454  	0.24553206442286216 	0.25038998752121683
#new nodepolicy 100
# 0.369   0.4678079419403824  	0.23806351840837997 	0.2513172203726005



# com1 nodepolicy 10
# 0.651   0.8292801774766457  	0.5989195749644476  	0.22983675836980566







import time
from train import *
from onlinemap import *
from compare1 import *

# def train(n):
#     trs=get_training_set(1000)
#     linkp=LinkPolicy(sub1,len(getallpath(sub1)),2,n,100)
#     linkp.train(trs)
#     linksaver=tf.train.Saver()
#     linksaver.save(linkp.sess, "./linkmodel/linkmodel.ckpt")
#     lif(n)
#     rec, rc = RLNL()
#     print(rec,rc)



# def train(n):
#     trs=get_training_set(1000)
#     nodep=NodePolicy(sub1,sub1.number_of_nodes(),5,learning_rate=0.05,num_epoch=n,batch_size=100)
#     nodep.train(trs)
#     nodesaver = tf.train.Saver()
#     nodesaver.save(nodep.sess, "./nodemodel/nodemodel.ckpt")
#     nof(n)
#     rec, rc = RLN()
#     print(rec,rc)
# train(1)

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
# comp1train(10)
rec, rc = RLNL()
print(rec, rc)
