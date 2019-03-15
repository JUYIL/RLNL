#new nodepolicy 10 5
# 0.654  0.823961394505712   	0.5879847293853341  	0.22512982974686352
# lt 1 RLNL:0.8015  0.7420461195942332  	0.7488780886922155  	0.3867336932756678
#new nodepolicy 10 6
# 0.645 0.825 0.561 0.20
#new nodepolicy 100
#

# com1 nodepolicy 10
# 0.651   0.8292801774766457  	0.5989195749644476  	0.22983675836980566
# com1 nodepolicy 100 time=7h
# 0.721   0.844027820090658   	0.6329730216219035  	0.2750193531636147
# com1 nodepolicy 10 k=5 start=14:37 end=14:40
# 0.7565  0.766040337035084   	0.7157667933089912  	0.3312687045914222

# from train import *
from onlinemap import *
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
#
# train(5)
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

rec, rc = grc()
print(rec,rc)