import gym
from gym import spaces
import copy
import networkx as nx
import numpy as np
from network import *

btns=getbtns()

class NodeEnv(gym.Env):

    def __init__(self, sub):
        self.count = -1
        self.n_action = sub.number_of_nodes()
        # when we reset,we should also reset the sub,that's why I save an original sub
        self.origin = copy.deepcopy(sub)
        # this sub is for us to change states when we make a step
        self.sub = copy.deepcopy(sub)
        # self.vnr = vnr
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_action, 5), dtype=np.float32)
        cpu_all, bw_all = [], []
        self.degree, self.closeness, self.avrdis = [], [], []
        self.reqas=[]
        for u in range(self.n_action):
            cpu_all.append(sub.nodes[u]['cpu'])
            bw_all.append(calculate_adjacent_bw(sub, u))

        # normalization
        self.cpu_all = (cpu_all - np.min(cpu_all)) / (np.max(cpu_all) - np.min(cpu_all))
        self.bw_all = (bw_all - np.min(bw_all)) / (np.max(bw_all) - np.min(bw_all))
        self.cpu_remain = self.cpu_all
        self.bw_all_remain = self.bw_all
        # degree centrality
        for i in nx.degree_centrality(sub).values():
            self.degree.append(i)
        # closeness centrality
        for j in nx.closeness_centrality(sub).values():
            self.closeness.append(j)
        # average distance to mapped nodes
        self.avrdis=[0.5 for k in range(100)]
        self.state = None

    def set_vnr(self,vnr):
        self.vnr=vnr
        self.count=-1

    def step(self, action):
        self.count = self.count + 1
        self.cpu_remain, self.bw_all_remain = [], []
        for u in range(self.n_action):
            adjacent_bw = calculate_adjacent_bw(self.sub, u, 'bw_remain')
            if u == action:
                self.sub.nodes[action]['cpu_remain'] -= self.vnr.nodes[self.count]['cpu']
                adjacent_bw -= calculate_adjacent_bw(self.vnr, self.count)
            self.cpu_remain.append(self.sub.nodes[u]['cpu_remain'])
            self.bw_all_remain.append(adjacent_bw)

        self.cpu_remain = (self.cpu_remain - np.min(self.cpu_remain)) / (
                np.max(self.cpu_remain) - np.min(self.cpu_remain))
        self.bw_all_remain = (self.bw_all_remain - np.min(self.bw_all_remain)) / (
                np.max(self.bw_all_remain) - np.min(self.bw_all_remain))
        self.reqas.append(action)



        if len(self.reqas)==self.vnr.number_of_nodes():
            self.avrdis=[0.5 for i in range(100)]
            self.reqas=[]

        else:
            sumdis=[0 for i in range(100)]
            for ac in self.reqas:
                b=caculate_avr_dis(ac)
                # print(b)
                for i in range(100):

                    sumdis[i]+=b[i]

            averdis_=[s/len(self.reqas) for s in sumdis]

            self.avrdis=[(s - np.min(averdis_)) / (np.max(averdis_) - np.min(averdis_)) for s in averdis_]


        self.state = (self.cpu_remain,
                      self.bw_all_remain,
                      self.degree,
                      self.closeness,
                      self.avrdis)

        # reward = self.sub.nodes[action]['cpu_remain'] / self.sub.nodes[action]['cpu']
        reward = 0.0
        # reward = self.vnr.nodes[self.count]['cpu'] / self.sub.nodes[action]['cpu_remain']
        return np.vstack(self.state).transpose(), reward, False, {}

    def statechange(self, nodemap):
        for vid, sid in nodemap.items():
            self.sub.nodes[sid]['cpu_remain'] += self.vnr.nodes[vid]['cpu']

        self.cpu_remain, self.bw_all_remain = [], []
        for u in range(self.n_action):
            self.cpu_remain.append(self.sub.nodes[u]['cpu_remain'])
            adjacent_bw = calculate_adjacent_bw(self.sub, u, 'bw_remain')
            self.bw_all_remain.append(adjacent_bw)

        for vid, sid in nodemap.items():
            self.bw_all_remain[sid]+=calculate_adjacent_bw(self.vnr, vid)

        self.cpu_remain = (self.cpu_remain - np.min(self.cpu_remain)) / (
                    np.max(self.cpu_remain) - np.min(self.cpu_remain))
        self.bw_all_remain = (self.bw_all_remain - np.min(self.bw_all_remain)) / (
                    np.max(self.bw_all_remain) - np.min(self.bw_all_remain))

        self.state = (self.cpu_remain,
                      self.bw_all_remain,
                      self.degree,
                      self.closeness,
                      self.avrdis)
        return np.vstack(self.state).transpose()



    def reset(self):
        self.count = -1
        self.sub = copy.deepcopy(self.origin)
        self.cpu_remain = self.cpu_all
        self.bw_all_remain = self.bw_all
        self.avrdis=[0.5 for i in range(100)]
        self.state = (self.cpu_remain,
                      self.bw_all_remain,
                      self.degree,
                      self.closeness,
                      self.avrdis)
        return np.vstack(self.state).transpose()

    def render(self, mode='human'):
        pass

class LinkEnv(gym.Env):

    def __init__(self, sub):
        self.count = -1
        self.linkpath = getallpath(sub)
        self.n_action = len(self.linkpath)
        # when we reset,we should also reset the sub,that's why I save an original sub
        self.origin = copy.deepcopy(sub)
        # this sub is for us to change states when we make a step
        self.sub = copy.deepcopy(sub)
        # self.vnr = vnr
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_action, 2), dtype=np.float32)
        mbw = []
        btn=btns
        self.mbw, self.btn = [], []
        for paths in self.linkpath.values():
            path=list(paths.values())[0]
            mbw.append(minbw(sub,path))

        # normalization
        self.mbw = (mbw - np.min(mbw)) / (np.max(mbw) - np.min(mbw))
        self.btn = (btn - np.min(btn)) / (np.max(btn) - np.min(btn))
        self.mbw_remain = self.mbw

        self.state = None


    def set_vnr(self,vnr):
        self.vnr=vnr
        # self.count=-1

    def set_link(self,link):
        self.link=link

    def step(self, action):
        # self.count = self.count + 1
        self.mbw_remain = []
        thepath = list(self.linkpath[action].values())[0]

        # reward = self.vnr[self.link[0]][self.link[1]]['bw'] / minbw(self.sub, thepath)

        i = 0
        while i < len(thepath) - 1:
            fr = thepath[i]
            to = thepath[i+1]
            self.sub[fr][to]['bw_remain'] -= self.vnr[self.link[0]][self.link[1]]['bw']
            i += 1

        for paths in self.linkpath.values():
            path=list(paths.values())[0]
            self.mbw_remain.append(minbw(self.sub,path))
        self.mbw_remain = (self.mbw_remain - np.min(self.mbw_remain)) / (
                np.max(self.mbw_remain) - np.min(self.mbw_remain))

        self.state = (self.mbw_remain,self.btn)

        # reward = self.vnr[self.link[0]][self.link[1]]['bw'] / minbw(self.sub,thepath)
        reward=0.0
        return np.vstack(self.state).transpose(), reward, False, {}

    def statechange(self,sub,linkmap):

        self.mbw_remain = []
        for vlink, slink in linkmap.items():
            v_fr = vlink[0]
            v_to = vlink[-1]
            i=0
            while i < len(slink) - 1:
                self.sub[slink[i]][slink[i+1]]['bw_remain'] += self.vnr[v_fr][v_to]['bw']
                i += 1

        for paths in self.linkpath.values():
            path=list(paths.values())[0]
            self.mbw_remain.append(minbw(sub,path))
        self.mbw_remain = (self.mbw_remain - np.min(self.mbw_remain)) / (
                np.max(self.mbw_remain) - np.min(self.mbw_remain))

        self.state = (self.mbw_remain, self.btn)
        return np.vstack(self.state).transpose()



    def reset(self):
        self.count = -1
        self.sub = copy.deepcopy(self.origin)
        self.mbw_remain = self.mbw

        self.state = (self.mbw_remain,
                      self.btn)
        return np.vstack(self.state).transpose()

    def render(self, mode='human'):
        pass


class MyEnv(gym.Env):

    def __init__(self, sub):
        self.count = -1
        # self.n_action = sub.number_of_nodes()
        # when we reset,we should also reset the sub,that's why I save an original sub
        self.origin = copy.deepcopy(sub)
        # this sub is for us to change states when we make a step
        self.sub = copy.deepcopy(sub)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 3), dtype=np.float32)
        self.rcpu=self.vnr.nodes[self.vnode]['cpu']
        self.acpu=self.rcpu
        self.ucpu=0
        self.scpu=self.sub.nodes[self.snode]['cpu']
        self.scpur=self.sub.nodes[self.snode]['cpu_remain']
        self.r1=self.acpu / self.rcpu
        self.r2=self.ucpu / self.acpu
        self.r3=self.scpur / self.scpu

        self.state = None
    def set_vnr(self,vnr):
        self.vnr=vnr
        # self.count=-1

    def set_vnode(self,vnode):
        self.vnode=vnode
        # self.count=-1

    def set_snode(self,snode):
        self.snode=snode

    def step(self, action):
        self.count = self.count + 1
        self.cpu_remain, self.bw_all_remain = [], []


        self.cpu_remain = (self.cpu_remain - np.min(self.cpu_remain)) / (
                np.max(self.cpu_remain) - np.min(self.cpu_remain))
        self.bw_all_remain = (self.bw_all_remain - np.min(self.bw_all_remain)) / (
                np.max(self.bw_all_remain) - np.min(self.bw_all_remain))



        self.state = (self.cpu_remain,)

        # reward = self.sub.nodes[action]['cpu_remain'] / self.sub.nodes[action]['cpu']
        # reward = 0.0
        reward = self.vnr.nodes[self.count]['cpu'] / self.sub.nodes[action]['cpu_remain']
        return np.vstack(self.state).transpose(), reward, False, {}

    def statechange(self,nodemap):


        self.state = (self.r1, self.r2, self.r3)
        return np.vstack(self.state).transpose()



    def reset(self):
        self.count = -1
        self.sub = copy.deepcopy(self.origin)
        self.state = (self.r1, self.r2, self.r3)
        return np.vstack(self.state).transpose()

    def render(self, mode='human'):
        pass