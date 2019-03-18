from policy import *
from environment import *
from analysis import *
from compare1 import *
from compare2 import *
from compare3 import *

#VNE with node and link traning
def RLNL():
    # create a substrate network
    sub = create_sub('sub.txt')
    # create a set of virtual network requests
    reqs = gettestreqs(2000)
    # get all path in sub
    linkpath=getallpath(sub)

    # get trained policy
    nodeenv=NodeEnv(sub)
    linkenv=LinkEnv(sub)
    nodeobsreset=nodeenv.reset()
    linkobsreset=linkenv.reset()
    nodep = nodepolicy(nodeenv.action_space.n,
                       nodeenv.observation_space.shape)
    linkp = linkpolicy(linkenv.action_space.n,
                       linkenv.observation_space.shape)
    arrived, count, rc_r, rc_c, nodeu, linku = 0, 0, 0, 0, 0, 0
    mapped_info={}
    evaluate={}
    for req in reqs:

        if req.graph['type'] == 0:
            arrived += 1
            print('req%d is mapping... ' % req.graph['id'])
            print('node mapping...')
            reqr = 0
            node_map = {}
            nodeenv.set_vnr(req)
            for node in req.nodes:
                observation = nodeobsreset
                action = nodep.choose_max_action(observation, nodeenv.sub, req.nodes[node]['cpu'], req.number_of_nodes())
                if action == -1:
                    break
                else:
                    observation_next, reward, done, info = nodeenv.step(action)
                    nodeobsreset = observation_next
                    reqr += req.nodes[node]['cpu']
                    node_map.update({node: action})
            reqc = reqr
            if len(node_map) == req.number_of_nodes():
                print('link mapping...')
                link_map = {}
                linkenv.set_vnr(req)
                for link in req.edges:
                    linkenv.set_link(link)
                    vn_from = link[0]
                    vn_to = link[1]
                    sn_from = node_map[vn_from]
                    sn_to = node_map[vn_to]
                    bw = req[vn_from][vn_to]['bw']
                    if nx.has_path(linkenv.sub, sn_from, sn_to):
                        linkob = linkobsreset
                        linkaction = linkp.choose_max_action(linkob, linkenv.sub, bw, linkpath, sn_from, sn_to)
                        if linkaction == -1:
                            break
                        else:
                            linkobservation_next, linkreward, linkdone, linkinfo = linkenv.step(linkaction)
                            linkobsreset = linkobservation_next
                            path=list(linkpath[linkaction].values())[0]
                            link_map.update({link: path})
                            reqr += req[vn_from][vn_to]['bw']
                            reqc += req[vn_from][vn_to]['bw'] * (len(path) - 1)
                if len(link_map) == req.number_of_edges():
                    print('req%d is mapped ' % req.graph['id'])
                    count += 1
                    rc_r += reqr
                    rc_c += reqc
                    mapped_info.update({req.graph['id']: (node_map, link_map)})

                    if rc_c == 0:
                        lrc = 0
                    else:
                        lrc = rc_r / rc_c
                    nodeu += nodeuti(nodeenv.sub)
                    linku += linkuti(linkenv.sub)
                    if count == 0:
                        nodeu, linku = 0, 0
                    times = req.graph['time']
                    evaluate.update({times: (count / arrived, rc_r, rc_c, lrc, nodeu / arrived, linku / arrived)})

                else:
                    nodeobsreset = nodeenv.statechange(node_map)
                    linkobsreset = linkenv.statechange(link_map)
                    print('req%d mapping is failed ' % req.graph['id'])
            else:
                print(node_map)
                nodeobsreset = nodeenv.statechange(node_map)
                print('req%d mapping is failed ' % req.graph['id'])


        if req.graph['type'] == 1:
            if mapped_info.__contains__(req.graph['id']):
                print('req%d is leaving... ' % req.graph['id'])
                linkenv.set_vnr(req)
                nodeenv.set_vnr(req)
                reqid = req.graph['id']
                nodemap = mapped_info[reqid][0]
                linkmap = mapped_info[reqid][1]
                nodeobsreset = nodeenv.statechange(nodemap)
                linkobsreset = linkenv.statechange(linkmap)
                mapped_info.pop(reqid)
            else:
                pass

    save_result('RLNL',evaluate)
    rec=count/arrived
    if rc_c==0:
        rc=0
    else:
        rc=rc_r / rc_c
    return rec,rc

# VNE without link training
def RLN():
    # create a substrate network
    sub = create_sub('sub.txt')
    # create a set of virtual network requests
    reqs = gettestreqs(2000)

    # start=time.time()

    # get node trained policy
    nodeenv=NodeEnv(sub)
    nodeobsreset=nodeenv.reset()
    nodep = nodepolicy(nodeenv.action_space.n,
                       nodeenv.observation_space.shape)
    arrived, count, rc_r, rc_c, nodeu, linku = 0, 0, 0, 0, 0, 0
    mapped_info={}
    evaluate={}

    for req in reqs:
        if req.graph['type'] == 0:
            arrived += 1
            print('req%d is mapping... ' % req.graph['id'])
            print('node mapping...')
            reqr = 0
            node_map = {}
            nodeenv.set_vnr(req)
            for node in req.nodes:
                observation = nodeobsreset
                action = nodep.choose_max_action(observation, nodeenv.sub, req.nodes[node]['cpu'], req.number_of_nodes())
                if action == -1:
                    break
                else:
                    observation_next, reward, done, info = nodeenv.step(action)
                    nodeobsreset = observation_next
                    reqr += req.nodes[node]['cpu']
                    node_map.update({node: action})
            reqc = reqr
            if len(node_map) == req.number_of_nodes():
                print('link mapping...')
                link_map = {}
                for link in req.edges:
                    vn_from = link[0]
                    vn_to = link[1]
                    sn_from = node_map[vn_from]
                    sn_to = node_map[vn_to]
                    if nx.has_path(nodeenv.sub, sn_from, sn_to):
                        for path in nx.all_shortest_paths(nodeenv.sub, source=sn_from, target=sn_to):
                            if minbw(nodeenv.sub, path) >= req[vn_from][vn_to]['bw']:
                                link_map.update({link: path})
                                reqr += req[vn_from][vn_to]['bw']
                                reqc += req[vn_from][vn_to]['bw'] * (len(path) - 1)
                                i = 0
                                while i < len(path) - 1:
                                    nodeenv.sub[path[i]][path[i + 1]]['bw_remain'] -= req[vn_from][vn_to]['bw']
                                    i += 1
                                break
                            else:
                                continue

                if len(link_map) == req.number_of_edges():
                    print('req%d is mapped ' % req.graph['id'])
                    count += 1
                    rc_r += reqr
                    rc_c += reqc
                    mapped_info.update({req.graph['id']: (node_map, link_map)})

                    if rc_c == 0:
                        lrc = 0
                    else:
                        lrc = rc_r / rc_c
                    nodeu += nodeuti(nodeenv.sub)
                    linku += linkuti(nodeenv.sub)
                    times = req.graph['time']
                    evaluate.update({times: (count / arrived, rc_r, rc_c, lrc, nodeu / arrived, linku / arrived)})

                else:
                    nodeobsreset = nodeenv.statechange(node_map)
                    for vl, pl in link_map.items():
                        vfr, vto = vl[0], vl[1]
                        i = 0
                        while i < len(pl) - 1:
                            nodeenv.sub[pl[i]][pl[i + 1]]['bw_remain'] += req[vfr][vto]['bw']
                            i += 1
                    print('req%d mapping is failed ' % req.graph['id'])
            else:
                nodeobsreset = nodeenv.statechange(node_map)
                print('req%d mapping is failed ' % req.graph['id'])


        if req.graph['type'] == 1:
            if mapped_info.__contains__(req.graph['id']):
                print('req%d is leaving... ' % req.graph['id'])
                nodeenv.set_vnr(req)
                reqid = req.graph['id']
                nodemap = mapped_info[reqid][0]
                linkmap = mapped_info[reqid][1]
                nodeobsreset = nodeenv.statechange(nodemap)
                for vl, path in linkmap.items():
                    i = 0
                    while i < len(path) - 1:
                        nodeenv.sub[path[i]][path[i + 1]]['bw_remain'] += req[vl[0]][vl[1]]['bw']
                        i += 1
                mapped_info.pop(reqid)
            else:
                pass

    save_result('RLN', evaluate)
    rec=count/arrived
    rc=rc_r / rc_c

    return rec,rc

def rl():
    # create a substrate network
    sub = create_sub('sub.txt')
    # create a set of virtual network requests
    reqs = gettestreqs(2000)

    # start=time.time()

    # get node trained policy
    nodeenv=Env(sub)
    nodeobsreset=nodeenv.reset()
    nodep = c1nodepolicy(nodeenv.action_space.n,
                       nodeenv.observation_space.shape)
    arrived, count, rc_r, rc_c, nodeu, linku = 0, 0, 0, 0, 0, 0
    mapped_info={}
    evaluate={}

    for req in reqs:
        if req.graph['type'] == 0:
            arrived += 1
            print('req%d is mapping... ' % req.graph['id'])
            print('node mapping...')
            reqr = 0
            node_map = {}
            nodeenv.set_vnr(req)
            for node in req.nodes:
                observation = nodeobsreset
                action = nodep.choose_max_action(observation, nodeenv.sub, req.nodes[node]['cpu'], req.number_of_nodes())
                if action == -1:
                    break
                else:
                    observation_next, reward, done, info = nodeenv.step(action)
                    nodeobsreset = observation_next
                    reqr += req.nodes[node]['cpu']
                    node_map.update({node: action})
            reqc = reqr
            if len(node_map) == req.number_of_nodes():
                print('link mapping...')
                link_map = {}
                for link in req.edges:
                    vn_from = link[0]
                    vn_to = link[1]
                    sn_from = node_map[vn_from]
                    sn_to = node_map[vn_to]
                    if nx.has_path(nodeenv.sub, sn_from, sn_to):
                        for path in nx.all_shortest_paths(nodeenv.sub, source=sn_from, target=sn_to):
                            if minbw(nodeenv.sub, path) >= req[vn_from][vn_to]['bw']:
                                link_map.update({link: path})
                                reqr += req[vn_from][vn_to]['bw']
                                reqc += req[vn_from][vn_to]['bw'] * (len(path) - 1)
                                i = 0
                                while i < len(path) - 1:
                                    nodeenv.sub[path[i]][path[i + 1]]['bw_remain'] -= req[vn_from][vn_to]['bw']
                                    i += 1
                                break
                            else:
                                continue

                if len(link_map) == req.number_of_edges():
                    print('req%d is mapped ' % req.graph['id'])
                    count += 1
                    rc_r += reqr
                    rc_c += reqc
                    mapped_info.update({req.graph['id']: (node_map, link_map)})

                    if rc_c == 0:
                        lrc = 0
                    else:
                        lrc = rc_r / rc_c
                    nodeu += nodeuti(nodeenv.sub)
                    linku += linkuti(nodeenv.sub)
                    times = req.graph['time']
                    evaluate.update({times: (count / arrived, rc_r, rc_c, lrc, nodeu / arrived, linku / arrived)})

                else:
                    nodeobsreset = nodeenv.statechange(node_map)
                    for vl, pl in link_map.items():
                        vfr, vto = vl[0], vl[1]
                        i = 0
                        while i < len(pl) - 1:
                            nodeenv.sub[pl[i]][pl[i + 1]]['bw_remain'] += req[vfr][vto]['bw']
                            i += 1
                    print('req%d mapping is failed ' % req.graph['id'])
            else:
                nodeobsreset = nodeenv.statechange(node_map)
                print('req%d mapping is failed ' % req.graph['id'])


        if req.graph['type'] == 1:
            if mapped_info.__contains__(req.graph['id']):
                print('req%d is leaving... ' % req.graph['id'])
                nodeenv.set_vnr(req)
                reqid = req.graph['id']
                nodemap = mapped_info[reqid][0]
                linkmap = mapped_info[reqid][1]
                nodeobsreset = nodeenv.statechange(nodemap)
                for vl, path in linkmap.items():
                    i = 0
                    while i < len(path) - 1:
                        nodeenv.sub[path[i]][path[i + 1]]['bw_remain'] += req[vl[0]][vl[1]]['bw']
                        i += 1
                mapped_info.pop(reqid)
            else:
                pass
    save_result('rl', evaluate)
    rec=count/arrived
    rc=rc_r / rc_c

    return rec,rc

def grc():
    # create a substrate network
    sub = create_sub('sub.txt')
    # create a set of virtual network requests
    reqs = gettestreqs(2000)
    arrived, count, rc_r, rc_c, nodeu, linku = 0, 0, 0, 0, 0, 0
    mapped_info={}
    evaluate={}
    agent=GRC(damping_factor=0.9,sigma=1e-6)

    for req in reqs:
        if req.graph['type'] == 0:
            arrived += 1
            print('req%d is mapping... ' % req.graph['id'])
            print('node mapping...')
            reqr = 0
            node_map = agent.run(sub,req)

            if len(node_map) == req.number_of_nodes():
                for v_id, s_id in node_map.items():
                    sub.nodes[s_id]['cpu_remain'] -= req.nodes[v_id]['cpu']
                    reqr+=req.nodes[v_id]['cpu']
                reqc = reqr
                print('link mapping...')
                link_map = {}
                for link in req.edges:
                    vn_from = link[0]
                    vn_to = link[1]
                    sn_from = node_map[vn_from]
                    sn_to = node_map[vn_to]
                    if nx.has_path(sub, sn_from, sn_to):
                        # for path in nx.all_shortest_paths(sub, source=sn_from, target=sn_to):
                        for path in k_shortest_path(sub, sn_from, sn_to):
                            if minbw(sub, path) >= req[vn_from][vn_to]['bw']:
                                link_map.update({link: path})
                                reqr += req[vn_from][vn_to]['bw']
                                reqc += req[vn_from][vn_to]['bw'] * (len(path) - 1)
                                i = 0
                                while i < (len(path) - 1):
                                    sub[path[i]][path[i + 1]]['bw_remain'] -= req[vn_from][vn_to]['bw']
                                    i += 1
                                break
                            else:
                                continue

                if len(link_map) == req.number_of_edges():
                    print('req%d is mapped ' % req.graph['id'])
                    count += 1
                    rc_r += reqr
                    rc_c += reqc
                    mapped_info.update({req.graph['id']: (node_map, link_map)})

                    if rc_c == 0:
                        lrc = 0
                    else:
                        lrc = rc_r / rc_c
                    nodeu += nodeuti(sub)
                    linku += linkuti(sub)
                    times = req.graph['time']
                    evaluate.update({times: (count / arrived, rc_r, rc_c, lrc, nodeu / arrived, linku / arrived)})

                else:
                    for vl, pl in link_map.items():
                        vfr, vto = vl[0], vl[1]
                        i = 0
                        while i < (len(pl) - 1):
                            sub[pl[i]][pl[i + 1]]['bw_remain'] += req[vfr][vto]['bw']
                            i += 1
                    print('req%d mapping is failed ' % req.graph['id'])
            else:
                print('req%d mapping is failed ' % req.graph['id'])


        if req.graph['type'] == 1:
            if mapped_info.__contains__(req.graph['id']):
                print('req%d is leaving... ' % req.graph['id'])
                reqid = req.graph['id']
                nodemap = mapped_info[reqid][0]
                linkmap = mapped_info[reqid][1]
                for v_id, s_id in nodemap.items():
                    sub.nodes[s_id]['cpu_remain'] += req.nodes[v_id]['cpu']
                for vl, path in linkmap.items():
                    i = 0
                    while i < len(path) - 1:
                        sub[path[i]][path[i + 1]]['bw_remain'] += req[vl[0]][vl[1]]['bw']
                        i += 1
                mapped_info.pop(reqid)
            else:
                pass

    save_result('GRC', evaluate)
    rec=count/arrived
    rc=rc_r / rc_c

    return rec,rc


def mcts():
    # create a substrate network
    sub = create_sub('sub.txt')
    # create a set of virtual network requests
    reqs = gettestreqs(2000)
    arrived, count, rc_r, rc_c, nodeu, linku = 0, 0, 0, 0, 0, 0
    mapped_info = {}
    evaluate = {}
    agent = MCTS(computation_budget=5,exploration_constant=0.5)

    for req in reqs:
        if req.graph['type'] == 0:
            arrived += 1
            print('req%d is mapping... ' % req.graph['id'])
            print('node mapping...')
            reqr = 0
            node_map = agent.run(sub,req)

            if len(node_map) == req.number_of_nodes():
                for v_id, s_id in node_map.items():
                    sub.nodes[s_id]['cpu_remain'] -= req.nodes[v_id]['cpu']
                    reqr+=req.nodes[v_id]['cpu']
                reqc = reqr
                print('link mapping...')
                link_map = {}
                for link in req.edges:
                    vn_from = link[0]
                    vn_to = link[1]
                    sn_from = node_map[vn_from]
                    sn_to = node_map[vn_to]
                    if nx.has_path(sub, sn_from, sn_to):
                        for path in nx.all_shortest_paths(sub, source=sn_from, target=sn_to):
                            if minbw(sub, path) >= req[vn_from][vn_to]['bw']:
                                link_map.update({link: path})
                                reqr += req[vn_from][vn_to]['bw']
                                reqc += req[vn_from][vn_to]['bw'] * (len(path) - 1)
                                i = 0
                                while i < (len(path) - 1):
                                    sub[path[i]][path[i + 1]]['bw_remain'] -= req[vn_from][vn_to]['bw']
                                    i += 1
                                break
                            else:
                                continue

                if len(link_map) == req.number_of_edges():
                    print('req%d is mapped ' % req.graph['id'])
                    count += 1
                    rc_r += reqr
                    rc_c += reqc
                    mapped_info.update({req.graph['id']: (node_map, link_map)})

                    if rc_c == 0:
                        lrc = 0
                    else:
                        lrc = rc_r / rc_c
                    nodeu += nodeuti(sub)
                    linku += linkuti(sub)
                    times = req.graph['time']
                    evaluate.update({times: (count / arrived, rc_r, rc_c, lrc, nodeu / arrived, linku / arrived)})

                else:
                    for vl, pl in link_map.items():
                        vfr, vto = vl[0], vl[1]
                        i = 0
                        while i < (len(pl) - 1):
                            sub[pl[i]][pl[i + 1]]['bw_remain'] += req[vfr][vto]['bw']
                            i += 1
                    print('req%d mapping is failed ' % req.graph['id'])
            else:
                print('req%d mapping is failed ' % req.graph['id'])


        if req.graph['type'] == 1:
            if mapped_info.__contains__(req.graph['id']):
                print('req%d is leaving... ' % req.graph['id'])
                reqid = req.graph['id']
                nodemap = mapped_info[reqid][0]
                linkmap = mapped_info[reqid][1]
                for v_id, s_id in nodemap.items():
                    sub.nodes[s_id]['cpu_remain'] += req.nodes[v_id]['cpu']
                for vl, path in linkmap.items():
                    i = 0
                    while i < len(path) - 1:
                        sub[path[i]][path[i + 1]]['bw_remain'] += req[vl[0]][vl[1]]['bw']
                        i += 1
                mapped_info.pop(reqid)
            else:
                pass

    save_result('MCTS', evaluate)
    rec=count/arrived
    rc=rc_r / rc_c
    return rec, rc