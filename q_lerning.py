import gym
import numpy as np
import time
import random as ran


class Agent:
    def __init__(self):
        self.Q_a={}
        self.Q_b={}
        self.alpha=0.5
        self.gamma=0.3
        self.defautl_map={0:0,1:0}

    def get_action(self,init_state):
        next_a = self.max_q_a(self.Q_a, init_state)
        return next_a

    # def make_step(self,init_state,env):
    #     p = np.random.random()
        # if (p < .5):
        #     next_a, maxq = self.max_q_a(self.Q_a, init_state)
        #     new_state, reward, done, info=env.step(next_a)
        #     Q1[init_state][] = self.Q_a[init_state][new_state] + alpha * (reward + GAMMA * self.Q_b[new_state][next_a] - self.Q_a[s][a])
        # else:
        #     nxt_a, maxq = self.max_q_a(self.Q_b, init_state)
        #     Q2[s][a] = Q2[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q2[s][a])
    def update_q_table(self,reward,new_state,init_state,action):
        new_q=(1-self.alpha)*(self.Q_a.get(init_state,{0:0,1:0})[action])+self.alpha*(reward+self.gamma*self.max_q_for_state(new_state))
        if init_state in self.Q_a.keys():
            self.Q_a[init_state][action]=new_q
        else:
            self.Q_a[init_state]={0:0,1:0}
            self.Q_a[init_state][action]=new_q


    def max_q_a(self,q:map, state:np.array):
        max = -9999
        next_action = 0
        q_values = q.get(state, {0: 0, 1: 0})
        for action in q_values.keys():
            if (q_values[action] > max):
                max = q_values[action]
                next_action = action
        return next_action
    def max_q_for_state(self,state):
        q_values=self.Q_a.get(state,{0:0,1:0})
        return max(q_values.values())


env = gym.make('Pong-v0')
env2=gym.make("Copy-v0")
print(env2.action_space)
print(env2.action_space.sample())
agent=Agent()
start = time.time()
epizode=0


def e_gready(e,observ):
    agent.Q_a
    if observ not in agent.Q_a.keys() or 0 in agent.Q_a[observ].values():
        e_ost=0.9
    else:
        e_ost=e

    if ran.uniform(0,1)>e_ost:
        return agent.get_action(observ)
    else:
        return ran.randint(0,1)


epizode_mark=0
epizode_before_mark=10
epizodes_marks_list=list()
for i_episode in range(999999):
    observation: np.ndarray = env.reset()
    e=0.50


    if i_episode%1000==0 and e>0.1:
        epizode_before_mark=epizode_mark
        e=e-0.05
    for t in range(200):


        # print(observation)
        observation=np.round(observation,decimals=1)

        action=e_gready(e,tuple(observation))
        new_observation, reward, done, info = env.step(action)
        new_observation=np.round(new_observation,decimals=1)
        if done:
            epizode=epizode+1
            epizodes_marks_list.append(t)
            if time.time()-start>3:
                epizode_mark=np.array(epizodes_marks_list).mean()
                epizodes_marks_list = list()
                start=time.time()
                print("Episode finished after {} timesteps. epizode {}".format(epizode_mark,epizode))

            break
        else:
            agent.update_q_table(reward,tuple(new_observation),tuple(observation),action)
            observation=new_observation





