import networkx as nx
import copy
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def create_sub(path):
    """create a substrate network"""
    with open(path) as file_object:
        lines = file_object.readlines()

    G = nx.Graph()
    node_num, link_num = [int(x) for x in lines[0].split()]
    node_id = 0
    for line in lines[1:node_num + 1]:
        x, y, c = [float(x) for x in line.split()]
        G.add_node(node_id, x_coordinate=x, y_coordinate=y, cpu=c, cpu_remain=c)
        node_id = node_id + 1

    link_id = 0
    for line in lines[-link_num:]:
        src, dst, bw, dis = [float(x) for x in line.split()]
        G.add_edge(int(src), int(dst), link_id=link_id, bw=bw, bw_remain=bw, distance=dis)
        link_id = link_id + 1

    return G

def create_req(i, path):
    """create a new virtual network request"""
    with open(path) as file_object:
        lines = file_object.readlines()

    node_num, link_num, time, duration, maxD = [int(x) for x in lines[0].split()]
    graph = nx.Graph(id=i, type=0, time=time, duration=duration)
    node_id = 0
    for line in lines[1:node_num + 1]:
        x, y, c = [float(x) for x in line.split()]
        graph.add_node(node_id, x_coordinate=x, y_coordinate=y, cpu=c)
        node_id = node_id + 1

    link_id = 0
    for line in lines[-link_num:]:
        src, dst, bw, dis = [float(x) for x in line.split()]
        graph.add_edge(int(src), int(dst), link_id=link_id, bw=bw, distance=dis)
        link_id = link_id + 1

    return graph

def calculate_adjacent_bw(graph, u, kind='bw'):
    """calculate the bandwidth sum of the node u's adjacent links"""
    bw_sum = 0
    for v in graph.neighbors(u):
        if u <= v:
            bw_sum += graph[u][v][kind]
        else:
            bw_sum += graph[v][u][kind]
    return bw_sum

def get_g(path):
    with open(path, 'r') as f:
        list1 = list(map(float, f.readlines()[0].strip().split(' ')))
        linknum = list1[1]
    g = np.zeros(shape=(100, 100), dtype='int')
    j = 1
    listsublink = [0, ]
    while j < linknum + 1:
        with open(path, 'r') as f:
            listsublink.append(list(map(float, f.readlines()[j + 100].strip().split(' '))))
            linkfr = listsublink[j][0]
            linkto = listsublink[j][1]
            g[int(linkfr)][int(linkto)] = 1
            g[int(linkto)][int(linkfr)] = 1
            j += 1
    return g

def get_path(g, a):
        n = len(g)
        step = [0 for i in range(n)]
        step_path = [[] for i in range(n)]
        step_path[a] = [[a]]
        q = [a]
        while len(q) > 0:
            f = q.pop()
            s = step[f] + 1
            for i in range(0, n):
                if g[f][i] == 1:
                    if (step[i] == 0) or (step[i] > s):
                        step[i] = s
                        q.insert(0, i)
                        step_path[i] = deepcopy(step_path[f])
                        if len(step_path[i]) > 0:
                            for j in range(len(step_path[i])):
                                step_path[i][j].append(i)
                    elif step[i] == s:
                        dp = deepcopy(step_path[f])
                        if len(dp) > 0:
                            for j in range(len(dp)):
                                dp[j].append(i)
                        step_path[i] += dp
        step_path[a] = [[0]]
        return step_path

def caculate_avr_dis(action):

    dlz = np.zeros(shape=(100, 100), dtype='int')
    i = 0
    while i < 100:
        step_path = get_path(g, i)
        k = 0
        while k < 100:
            if i == k:
                dlz[i][k] = 0
            else:
                dlz[i][k] = len(step_path[k][0]) - 1
            k += 1
        i += 1
    averdis_=[]
    for i in range(100):
        averdis_.append(dlz[i][action])
        i+=1
    # averdis= [(s - np.min(averdis_)) / (np.max(averdis_) - np.min(averdis_)) for s in averdis_]
    return averdis_

def minbw(sub,path):
    """find the least bandwidth of a path"""
    bandwidth = 1000
    head = path[0]
    for tail in path[1:]:
        if sub[head][tail]['bw_remain'] <= bandwidth:
            bandwidth = sub[head][tail]['bw_remain']
        head = tail
    return bandwidth

def getallpath(sub):
    i = 0
    k=0
    linkaction = {}
    while i < 100:
        j = 0
        while j < 100:
            path = nx.shortest_simple_paths(sub, i, j)
            for p in path:
                if 1 < len(p) < 6:
                    linkaction.update({k:{(i,j):p}})
                    k+=1
                else:
                    break
            j += 1
        i += 1

    return linkaction

def getbtn(sub):
    linkpath=getallpath(sub)
    k=10000
    while k<50000:
        filename = 'btn%s.txt' % int((k+10000)/10000)
        with open(filename, 'w') as f:
            f.truncate()
        for j in range(k,k+10000):
            path = list(linkpath[j].values())[0]
            i = 0
            btnt = 0
            while i < (len(path) - 1):
                fr = path[i]
                to = path[i + 1]
                if nx.edge_betweenness_centrality(sub1).__contains__((fr, to)):
                    btnt += nx.edge_betweenness_centrality(sub1)[(fr, to)]
                else:
                    btnt += nx.edge_betweenness_centrality(sub1)[(to, fr)]
                i += 1

            with open(filename, 'a') as f:
                f.write(str(btnt / (len(path) - 1)))
                f.write('\n')
        k+=10000

    for j in range(50000, 59614):
        path = list(linkpath[j].values())[0]
        i = 0
        btnt = 0
        while i < (len(path) - 1):
            fr = path[i]
            to = path[i + 1]
            if nx.edge_betweenness_centrality(sub1).__contains__((fr, to)):
                btnt += nx.edge_betweenness_centrality(sub1)[(fr, to)]
            else:
                btnt += nx.edge_betweenness_centrality(sub1)[(to, fr)]
            i += 1

        with open('btn6.txt', 'a') as f:
            f.write(str(btnt / (len(path) - 1)))
            f.write('\n')

    print(len(linkpath))

    # return btn

def getbtns():
    btns=[]
    for i in range(1,7):
        path = 'btns/btn%s.txt' % i
        with open(path) as file_object:
            lines = file_object.readlines()
        for line in lines:
            btns.append(float(line))
    return btns

def getreqs(reqnum,j):
    reqs = []
    for i in range(j,j+reqnum):
        filename = 'trainVN/req%s.txt' % i
        vnr_arrive = create_req(i, filename)
        vnr_leave = copy.deepcopy(vnr_arrive)
        vnr_leave.graph['type'] = 1
        vnr_leave.graph['time'] = vnr_arrive.graph['time'] + vnr_arrive.graph['duration']
        reqs.append(vnr_arrive)
        reqs.append(vnr_leave)

    # sort the reqs by their time(including arrive time and depart time)
    reqs.sort(key=lambda r: r.graph['time'])

    return reqs

def get_training_set(reqnum):
    reqs = []
    for i in range(reqnum):
        filename = 'trainVN/req%s.txt' % i
        vnr_arrive = create_req(i, filename)
        vnr_leave = copy.deepcopy(vnr_arrive)
        vnr_leave.graph['type'] = 1
        vnr_leave.graph['time'] = vnr_arrive.graph['time'] + vnr_arrive.graph['duration']
        reqs.append(vnr_arrive)
        reqs.append(vnr_leave)

    # sort the reqs by their time(including arrive time and depart time)
    reqs.sort(key=lambda r: r.graph['time'])

    return reqs

def gettestreqs(reqnum):
    reqs = []
    for i in range(reqnum):
        filename = 'VNRequest/req%s.txt' % i
        vnr_arrive = create_req(i, filename)
        vnr_leave = copy.deepcopy(vnr_arrive)
        vnr_leave.graph['type'] = 1
        vnr_leave.graph['time'] = vnr_arrive.graph['time'] + vnr_arrive.graph['duration']
        reqs.append(vnr_arrive)
        reqs.append(vnr_leave)

    # sort the reqs by their time(including arrive time and depart time)
    reqs.sort(key=lambda r: r.graph['time'])

    return reqs

def nodeuti(sub):
    sum=0
    for i in range(sub.number_of_nodes()):
        sum+=(1-(sub.nodes[i]['cpu_remain'] / sub.nodes[i]['cpu']))
    nodeuu=sum/sub.number_of_nodes()

    return nodeuu

def linkuti(sub):
    sum = 0
    for path in nx.edges(sub):
        sum += (1 - (sub[path[0]][path[1]]['bw_remain'] / sub[path[0]][path[1]]['bw']))
    linkuu = sum / sub.number_of_edges()

    return linkuu

def nof(n):
    plt.figure()
    loss = []
    with open('Results/losslog-%s.txt' % n) as file_object:
        lines = file_object.readlines()
    for line in lines[1:]:
        loss.append(float(line))
    x = [i for i in range(n)]
    plt.plot(x, loss)
    plt.xlabel("eponum", fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.title('nodeloss change', fontsize=15)
    plt.savefig('Results/node%s.jpg' % n)

def lif(n):
    plt.figure()
    loss = []
    with open('Results/linklosslog-%s.txt' % n) as file_object:
        lines = file_object.readlines()
    for line in lines[1:]:
        loss.append(float(line))
    x = [i for i in range(n)]
    plt.plot(x, loss)
    plt.xlabel("eponum", fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.title('linkloss change', fontsize=15)
    plt.savefig('Results/link%s.jpg' % n)

def bfslinkmap(sub,req,node_map):
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
                    break
                else:
                    continue
    return link_map

# create a substrate network
sub1 = create_sub('sub.txt')
g=get_g('sub.txt')

# btns=getbtns()
