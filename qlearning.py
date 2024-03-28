# author: Xinglu Liu  
# Tsinghua University   
# 2022-01-30  
  
import pandas as pd  
import numpy as np  


class Q_Learning(object):

    network = {}  
    Nodes = []  
    _state_num = 0
    _actions = {}      
    MAX_action_num = 0  # 因为不是全连接网络，因此我们取最大的  

    def __init__(self, swithes, links, src, dst):
       self.network = {}
       self._get_network(links=links)     
       self.Nodes = sorted(swithes)  
       self._state_num = len(self.Nodes)    
        #    self._actions = 
       self.__get_actions()   
       self.MAX_action_num = self._state_num
       self._epsilon = 0.9   # greedy police  
       self._learning_rate = 0.1     # learning rate  
       self._discount_factor = 0.9    # discount factor  
       self.MAX_EPISODES = 100   # maximum episodes  
       self.Q_table = None
       self.org = src
       self.des = dst
       print(self.network)
       print(self._actions)
    #    self.build_Q_table()
    
    def _get_network(self, links):
        self.network={}
        for key, value in links.items():
            self.network[key]=value[1]
            
    def __get_actions(self):
        ac = {}
        for nd in self.Nodes:
            ac[nd]=[]
        for key, _ in self.network.items():
            ac[key[0]].append(key[1])  
        self._actions = ac 

    """ Generate Q-table """  
    def build_Q_table(self):  
        '''  
        生成一个全连接的矩阵，记录Q-value  
        :param _state_num:  
        :param Nodes:  
        :param MAX_action_num:  
        :return:  
        '''  
        self.Q_table = pd.DataFrame(  
            np.full((self._state_num, self.MAX_action_num),-1),  
            columns=self.Nodes,  
            index=self.Nodes  
        )
        for key, _ in self.network.items():
            if key[1]==self.des:
                self.Q_table.loc[key[0],key[1]]= 1
        for i in self.Nodes:
            self.Q_table.loc[i,i] = 0
        print(self.Q_table)

    
    def choose_action(self,state_current):  
        # This is how to choose an action  
        action = None  
        rand_num = np.random.uniform()  
        # candidate_actions = self._actions[state_current]  # 提取出下一个可以访问的节点  
        candidate_actions = self.Q_table.loc[state_current,:]
        if(rand_num > self._epsilon):   # act greedy  
            action_list = candidate_actions[candidate_actions >= 0].index.to_list()
            action = np.random.choice(action_list)  
        else:  # choose action with max Q-value  
            # max_Q_value = -np.inf  # 我们要选Q-value最大的  
            # for j in self.Nodes:  
            #     if(self.Q_table.loc[state_current, j] > max_Q_value and j in candidate_actions):   # 注意用index索引的话，就需要用.loc  
            #         max_Q_value = self.Q_table.loc[state_current, j]  # 注意用index索引的话，就需要用.loc  
            #         action = j  
            action = candidate_actions.idxmax()
    
        return action  
  
    def get_env_feedback(self,state_current, action):  
        # This is how agent will interact with the environment  
        reward = 0
        state_next = action
        if(state_next == self.des):
            reward = 10
        # arc = (state_current, action)  
        # max_length = max(list(self.network.values()))  
        # reward = max_length - self.network[arc]  # 奖励是做成这样，找出的是最短路  
        # reward = network[arc]  # 这样筛选出来的是最长路  
        return state_next, reward  
  
    def solve_SPP_with_Q_table(self):  
        solution = [self.org]  
        total_diatance = 0  
        current_node = self.org  
        while current_node != self.des:  
            # 注意，这里一定要是loc，因为我们的node下标是从1开始的，所用用loc索引Index,但是这里返回的下标是从0开始的，错了1，因此我们要加1  
            next_node = self.Q_table.loc[current_node, :].argmax() + 1 
            if current_node == next_node:
                break 
            solution.append(next_node)  
            total_diatance += self.network[current_node, next_node]  
            current_node = next_node  
    
        return solution, total_diatance  
  
  
  
    def Q_learning_algo(self):  
        '''  
        This is the main part of the Q_learning algorithm  
    
        :return: the resulted Q_table  
        '''  
        """ create initial Q-table """  
        self.build_Q_table()  
    
        """ Main loop to update the Q_table """  
        ## training  
        for episode in range(self.MAX_EPISODES):  
            if (episode % 10 == 0):  
                print('enter episode: {}'.format(episode), end='  ')  
    
            # initial state  
            state_current = self.org  
            is_terminated = False  
            if (episode % 10 == 0):  
                print('current position: {}'.format(state_current), end='   ')  
                print('next position: ', end='')  
    
            # 如果没有结束，则继续探索  
            step_counter = 0
            while not is_terminated:  
                # choose next action  
                action = self.choose_action(state_current)  
                state_next, reward = self.get_env_feedback(state_current, action)  # take action & get next state and reward  
                if (episode % 10 == 0):  
                    print(' {} '.format(state_next), end='')  
    
                # Update Q_table using temporal-difference (TD) method  
                Q_predict = self.Q_table.loc[state_current, action]  
                Q_target = 0  
                if(state_next != self.des):  
                    # # 注意用index索引的话，就需要用.loc  
                    Q_target = reward + self._discount_factor * self.Q_table.loc[state_next, :].max()   # next state is not terminal  
                else:  
                    Q_target = reward      # next state is terminal  
                    is_terminated = True   # terminate this episode  
    
                self.Q_table.loc[state_current, action] += self._learning_rate * (Q_target - Q_predict)  # update  
    
                state_current = state_next  # move to next state  
    
                step_counter += 1  
                if step_counter >= 100:
                    break
    
            if (episode % 10 == 0):  
                print()  
                print(self.Q_table, end='\n\n')  
     
 
  
# q = Q_Learning(None,None,None,None)
# if __name__ == "__main__":  
#     Q_table = build_Q_table(_state_num, Nodes, MAX_action_num)  
#     # print(Q_table)  
  
#     Q_table = Q_learning_algo()  
#     print('Final Q_table: \n', Q_table, end='\n\n')  
  
#     """ solve the problem with Q_table"""  
#     solution, total_diatance = solve_SPP_with_Q_table(org, des, Q_table)  
#     print('The solution is: {},   total_diatance: {}'.format(solution, total_diatance))  