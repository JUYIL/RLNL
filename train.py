# from policy import *
# from environment import *
# from network import *
# import networkx as nx
#
#
#
# def nodetrain(eponum,epinum,reqnum):
#     # create a substrate network
#
#     sub2 = create_sub('sub.txt')
#
#     env1=NodeEnv(sub2)
#     RL = NodePolicy(env1.action_space.n,
#                         env1.observation_space.shape)
#
#     with open('losslog.txt', 'w') as f:
#         f.truncate()
#
#     for i_epoch in range(eponum):
#         print('%d epoch is start' % i_epoch)
#         lrclog = []
#         losslog=[]
#         j=0
#         bach=0
#         for i_episode in range(epinum):
#             sub = create_sub('sub.txt')
#             reqs = getreqs(reqnum,j)
#             env = NodeEnv(sub)
#             obsreset = env.reset()
#             rc_r=0
#             rc_c=0
#             count=0
#             mapped_info={}
#             for req in reqs:
#                 if req.graph['type'] == 0:
#                     print('req%d is mapping... ' % req.graph['id'])
#                     print('node mapping...')
#                     reqr = 0
#                     node_map = {}
#                     env.set_vnr(req)
#                     for node in req.nodes:
#                         observation = obsreset
#                         action = RL.choose_action(observation, env.sub, req.nodes[node]['cpu'], req.number_of_nodes())
#                         if action == -1:
#                             break
#                         else:
#                             observation_next, reward, done, info = env.step(action)
#                             RL.store_transition(observation, action, reward)
#                             obsreset = observation_next
#                             reqr += req.nodes[node]['cpu']
#                             node_map.update({node: action})
#                     reqc = reqr
#                     if len(node_map) == req.number_of_nodes():
#                         print('link mapping...')
#                         link_map = {}
#                         for link in req.edges:
#                             vn_from = link[0]
#                             vn_to = link[1]
#                             sn_from = node_map[vn_from]
#                             sn_to = node_map[vn_to]
#                             if nx.has_path(env.sub, sn_from, sn_to):
#                                 for path in nx.all_shortest_paths(env.sub, source=sn_from, target=sn_to):
#                                     if minbw(env.sub,path) >= req[vn_from][vn_to]['bw']:
#                                         link_map.update({link: path})
#                                         reqr += req[vn_from][vn_to]['bw']
#                                         reqc += req[vn_from][vn_to]['bw'] * (len(path) - 1)
#                                         i = 0
#                                         while i < len(path) - 1:
#                                             env.sub[path[i]][path[i + 1]]['bw_remain'] -= req[vn_from][vn_to]['bw']
#                                             i += 1
#                                         break
#                                     else:
#                                         continue
#                         if len(link_map) == req.number_of_edges():
#                             print('req%d is mapped ' % req.graph['id'])
#                             count += 1
#                             rc_r += reqr
#                             rc_c += reqc
#                             mapped_info.update({req.graph['id']: (node_map, link_map)})
#                         else:
#                             obsreset = env.statechange(node_map)
#                             for vl,pl in link_map.items():
#                                 vfr,vto=vl[0],vl[1]
#                                 i=0
#                                 while i < len(pl) - 1:
#                                     env.sub[pl[i]][pl[i + 1]]['bw_remain'] += req[vfr][vto]['bw']
#                                     i += 1
#                             print('req%d mapping is failed ' % req.graph['id'])
#                     else:
#                         obsreset = env.statechange(node_map)
#                         print('req%d mapping is failed ' % req.graph['id'])
#
#                 if req.graph['type'] == 1:
#                     if mapped_info.__contains__(req.graph['id']):
#                         print('req%d is leaving... ' % req.graph['id'])
#                         env.set_vnr(req)
#                         reqid = req.graph['id']
#                         nodemap = mapped_info[reqid][0]
#                         linkmap = mapped_info[reqid][1]
#                         for vl, path in linkmap.items():
#                             i = 0
#                             while i < len(path) - 1:
#                                 env.sub[path[i]][path[i + 1]]['bw_remain'] += req[vl[0]][vl[1]]['bw']
#                                 i += 1
#                         obsreset = env.statechange(nodemap)
#                         mapped_info.pop(reqid)
#                     else:
#                         pass
#
#             j+=reqnum
#             if rc_c==0:
#                 lrcre =0
#             else:
#                 lrcre=rc_r/rc_c
#             recnum=count/reqnum
#             if lrcre==0 or recnum==0:
#                 loss=0
#             else:
#                 loss = RL.learn(lrcre, recnum)
#             print('%d episode is done and the loss is' %i_episode,loss)
#             losslog.append(loss)
#             print('lrc is',lrcre)
#             lrclog.append(lrcre)
#             print('recnum is',recnum)
#             print(losslog)
#             print(lrclog)
#         print('%d epoch is done' % i_epoch)
#         meanloss=np.mean(losslog)
#         with open('losslog.txt', 'a') as f:
#             f.write(str(meanloss))
#             f.write('\n')
#
#     return RL
#
# def linktrain(eponum,epinum,reqnum):
#     # create a substrate network
#
#     sub3 = create_sub('sub.txt')
#
#     nodeenv1 = NodeEnv(sub3)
#     linkenv1 = LinkEnv(sub3)
#     LP = LinkPolicy(linkenv1.action_space.n,
#                         linkenv1.observation_space.shape)
#     nodep=nodepolicy(nodeenv1.action_space.n,
#                         nodeenv1.observation_space.shape)
#     linkpath = getallpath(sub3)
#
#     with open('linklosslog.txt', 'w') as f:
#         f.truncate()
#
#     for i_epoch in range(eponum):
#         lrclog, losslog = [], []
#         j=0
#         for i_episode in range(epinum):
#             sub = create_sub('sub.txt')
#             reqs = getreqs(reqnum, j)
#             linkenv = LinkEnv(sub)
#             nodeenv = NodeEnv(sub)
#             obsreset = linkenv.reset()
#             nodeobreset=nodeenv.reset()
#             rc_r=0
#             rc_c=0
#             count=0
#             mapped_info={}
#             for req in reqs:
#                 if req.graph['type']==0:
#                     print('req%d is mapping... ' % req.graph['id'])
#                     print('node mapping...')
#                     reqr=0
#                     node_map={}
#                     nodeenv.set_vnr(req)
#                     for node in req.nodes:
#                         observation=nodeobreset
#                         action=nodep.choose_max_action(observation,nodeenv.sub,req.nodes[node]['cpu'],req.number_of_nodes())
#                         if action==-1:
#                             break
#                         else:
#                             observation_next,reward,done,info=nodeenv.step(action)
#                             nodeobreset=observation_next
#                             reqr+=req.nodes[node]['cpu']
#                             node_map.update({node:action})
#                     reqc = reqr
#                     if len(node_map) == req.number_of_nodes():
#                         print('link mapping...')
#                         linkenv.set_vnr(req)
#                         link_map = {}
#                         for link in req.edges:
#                             linkenv.set_link(link)
#                             vn_from = link[0]
#                             vn_to = link[1]
#                             sn_from = node_map[vn_from]
#                             sn_to = node_map[vn_to]
#                             bw=req[vn_from][vn_to]['bw']
#                             if nx.has_path(linkenv.sub, sn_from, sn_to):
#                                 linkob=obsreset
#                                 linkaction = LP.choose_action(linkob,linkenv.sub,bw,linkpath,sn_from,sn_to)
#                                 if linkaction == -1:
#                                     break
#                                 else:
#                                     lobservation_next, lreward, ldone, linfo = linkenv.step(linkaction)
#                                     LP.store_transition(linkob, linkaction, lreward)
#                                     obsreset = lobservation_next
#                                     path=list(linkpath[linkaction].values())[0]
#                                     link_map.update({link: path})
#                                     reqr += req[vn_from][vn_to]['bw']
#                                     reqc += req[vn_from][vn_to]['bw'] * (len(path) - 1)
#
#                         if len(link_map) == req.number_of_edges():
#                             print('req%d is mapped ' % req.graph['id'])
#                             count += 1
#                             rc_r += reqr
#                             rc_c += reqc
#                             mapped_info.update({req.graph['id']: (node_map, link_map)})
#                         else:
#                             nodeobreset = nodeenv.statechange(node_map)
#                             obsreset = linkenv.statechange(linkenv.sub,link_map)
#                             print('req%d mapping is failed ' % req.graph['id'])
#                     else:
#                         nodeobreset = nodeenv.statechange(node_map)
#                         print('req%d mapping is failed ' % req.graph['id'])
#
#                 if req.graph['type'] == 1:
#                     if mapped_info.__contains__(req.graph['id']):
#                         print('req%d is leaving... ' % req.graph['id'])
#                         linkenv.set_vnr(req)
#                         nodeenv.set_vnr(req)
#                         reqid = req.graph['id']
#                         nodemap = mapped_info[reqid][0]
#                         linkmap = mapped_info[reqid][1]
#                         nodeobreset = nodeenv.statechange(nodemap)
#                         obsreset = linkenv.statechange(linkenv.sub,linkmap)
#                         mapped_info.pop(reqid)
#                     else:
#                         pass
#
#             j += reqnum
#             if rc_c==0:
#                 lrcre =0
#             else:
#                 lrcre=rc_r/rc_c
#             recnum = count / reqnum
#             if lrcre==0 or recnum==0:
#                 loss=0
#             else:
#                 loss = LP.learn(lrcre,recnum)
#             print('%d episode is done and the loss is' % i_episode, loss)
#             losslog.append(loss)
#             print('lrc is', lrcre)
#             lrclog.append(lrcre)
#             print('recnum is', recnum)
#             print(losslog)
#             print(lrclog)
#         print('%d epoch is done' % i_epoch)
#         meanloss = np.mean(losslog)
#         with open('linklosslog.txt', 'a') as f:
#             f.write(str(meanloss))
#             f.write('\n')
#     return LP
