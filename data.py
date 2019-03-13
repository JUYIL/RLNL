#new nodepolicy 1
# 0.359  0.7720884227944529  	0.23586201305238322 	0.10413807585350601
#new nodepolicy 10 1
# 0.654  0.823961394505712   	0.5879847293853341  	0.22512982974686352
#new nodepolicy 10 10
# 0.3825   0.4766732825651454  	0.24553206442286216 	0.25038998752121683
#new nodepolicy 50
#
#new nodepolicy 1 100
#

# com1 nodepolicy 1
# 0.5005   0.7320228301273158  	0.4548032593131334  	0.18348912180836724
# com1 nodepolicy 10
# 0.6775  0.8310122922892651  	0.5961650631852115  	0.24341457773604092
# com1 nodepolicy 10 5
# 0.641   0.8364807638775762  	0.5580915558902976  	0.21167498581043667
# com1 nodepolicy 100
# 0.642  0.8261037199687731  	0.573066819646115   	0.2178665125388119





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



def train(n):
    trs=get_training_set(1000)
    nodep=NodePolicy(sub1,sub1.number_of_nodes(),5,learning_rate=0.05,num_epoch=n,batch_size=100)
    nodep.train(trs)
    nodesaver = tf.train.Saver()
    nodesaver.save(nodep.sess, "./nodemodel/nodemodel.ckpt")
    nof(n)
    rec, rc = RLN()
    print(rec,rc)
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
# comp1train(10)
