# author: Xinglu Liu  
# Tsinghua University   
# 2022-01-30  
  
import pandas as pd  
import numpy as np  
  
network = {(1,2): 15  
        # ,(2,1):15
        ,(1,4): 25  
        ,(1,3): 45  
        ,(2,5): 30  
        ,(2,4): 2  
        ,(5,7): 2  
        ,(4,7): 50  
        ,(4,3): 2  
        ,(3,6): 25  
        ,(6,7): 1  
       }  
Nodes = [1, 2, 3, 4, 5, 6, 7]  


"""  
Gurobi得到的最优解：  
45.0  
x_1,2   1.0  
x_2,4   1.0  
x_4,3   1.0  
x_3,6   1.0  
x_6,7   1.0  
[1, 2, 4, 3, 6, 7]  
"""  
  
""" define input parameters """  
np.random.seed(2)  # reproducible  
  
_state_num = len(Nodes)   # the length of the 1 dimensional world  
# ACTIONS = ['left', 'right']     # available actions  
_actions = {  
    1:[2, 3, 4],  
    2:[4, 5],  
    3:[6],  
    4:[3, 7],  
    5:[7],  
    6:[7],  
    7:[]  
}  
org = 1  
des = 7  
MAX_action_num = len(Nodes)   # 因为不是全连接网络，因此我们取最大的  
_epsilon = 0.9   # greedy police  
_learning_rate = 0.1     # learning rate  
_discount_factor = 0.9    # discount factor  
MAX_EPISODES = 100   # maximum episodes  
  
  
""" Generate Q-table """  
def build_Q_table(_state_num, Nodes, MAX_action_num):  
    '''  
    生成一个全连接的矩阵，记录Q-value  
    :param _state_num:  
    :param Nodes:  
    :param MAX_action_num:  
    :return:  
    '''  
    Q_table = pd.DataFrame(  
        np.zeros((_state_num, MAX_action_num)),  
        columns=Nodes,  
        index=Nodes  
    )  
  
    return Q_table  
  
  
def choose_action(state_current, Q_table):  
    # This is how to choose an action  
    action = None  
    rand_num = np.random.uniform()  
    candidate_actions = _actions[state_current]  # 提取出下一个可以访问的节点  
    if(rand_num > _epsilon):   # act greedy  
        action = np.random.choice(candidate_actions)  
    else:  # choose action with max Q-value  
        max_Q_value = -np.inf  # 我们要选Q-value最大的  
        for j in Nodes:  
            if(Q_table.loc[state_current, j] > max_Q_value and j in candidate_actions):   # 注意用index索引的话，就需要用.loc  
                max_Q_value = Q_table.loc[state_current, j]  # 注意用index索引的话，就需要用.loc  
                action = j  
  
    return action  
  
  
def get_env_feedback(state_current, action):  
    # This is how agent will interact with the environment  
    arc = (state_current, action)  
    max_length = max(list(network.values()))  
    reward = max_length - network[arc]  # 奖励是做成这样，找出的是最短路  
    # reward = network[arc]  # 这样筛选出来的是最长路  
  
    state_next = action  
    return state_next, reward  
  
def solve_SPP_with_Q_table(org, des, Q_table):  
    solution = [org]  
    total_diatance = 0  
    current_node = org  
    while current_node != des:  
        # 注意，这里一定要是loc，因为我们的node下标是从1开始的，所用用loc索引Index,但是这里返回的下标是从0开始的，错了1，因此我们要加1  
        next_node = Q_table.loc[current_node, :].argmax() + 1  
        solution.append(next_node)  
        total_diatance += network[current_node, next_node]  
        current_node = next_node  
  
    return solution, total_diatance  
  
  
  
def Q_learning_algo():  
    '''  
    This is the main part of the Q_learning algorithm  
  
    :return: the resulted Q_table  
    '''  
    """ create initial Q-table """  
    Q_table = build_Q_table(_state_num=_state_num, Nodes=Nodes, MAX_action_num=MAX_action_num)  
  
    """ Main loop to update the Q_table """  
    
    ## training  
    for episode in range(MAX_EPISODES):  
        # if (episode % 10 == 0):  
        print('enter episode: {}'.format(episode), end='  ')  
  
        # initial state  
        state_current = org  
        is_terminated = False  
        if (episode % 10 == 0):  
            print('current position: {}'.format(state_current), end='   ')  
            print('next position: ', end='')  
  
        # 如果没有结束，则继续探索  
        step_counter = 0  
        total_reward = 0
        while not is_terminated:  
            # choose next action  
            action = choose_action(state_current, Q_table)  
            state_next, reward = get_env_feedback(state_current, action)  # take action & get next state and reward  
            if (episode % 10 == 0):  
                print(' {} '.format(state_next), end='')  
  
            # Update Q_table using temporal-difference (TD) method  
            Q_predict = Q_table.loc[state_current, action]  
            Q_target = 0  
            if(state_next != des):  
                # # 注意用index索引的话，就需要用.loc  
                Q_target = reward + _discount_factor * Q_table.loc[state_next, :].max()   # next state is not terminal  
            else:  
                Q_target = reward      # next state is terminal  
                is_terminated = True   # terminate this episode  
  
            Q_table.loc[state_current, action] += _learning_rate * (Q_target - Q_predict)  # update  
            if is_terminated:
                print("reward:{}".format(Q_table.loc[state_current, action]))
            state_current = state_next  # move to next state  

            total_reward += reward
            # _discount_factor_t *= _discount_factor
            step_counter += 1  
        # print("reward:{}".format(total_reward))
        if (episode % 10 == 0):  
            print()  
            print(Q_table, end='\n\n')  
  
    return Q_table  
  
  
  
if __name__ == "__main__":  
    Q_table = build_Q_table(_state_num, Nodes, MAX_action_num)  
    # print(Q_table)  
  
    Q_table = Q_learning_algo()  
    print('Final Q_table: \n', Q_table, end='\n\n')  
    
    """ solve the problem with Q_table"""  
    solution, total_diatance = solve_SPP_with_Q_table(org, des, Q_table)  
    print('The solution is: {},   total_diatance: {}'.format(solution, total_diatance))  
