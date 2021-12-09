import random

# 0 = rock, 1 = paper, 2 = scissors
from math import sqrt

import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque


def random_move():
    choice = random.randint(0, 2)
    return choice

# PLEASE NOTE: This algorithm looping 1000 times can take up to 40 minutes. Consider reducing the loop for quick review.
def mvd():
    # score keeping variables
    markov_wins = 0
    dq_wins = 0
    draws = 0

    window = 10  # window size for rate trending calc
    winRateTrend, tieRateTrend, lostRateTrend = 0, 0, 0
    winRateMovingAvg, tieRateMovingAvg, lostRateMovingAvg = 0, 0, 0
    winRateBuf, tieRateBuf, lostRateBuf = deque(maxlen=window), deque(maxlen=window), deque(
        maxlen=window)  # put all the observation state in here; shape in Keras input format
    state = np.array([[None, None, None, winRateTrend, tieRateTrend, lostRateTrend, winRateMovingAvg, tieRateMovingAvg,
                       lostRateMovingAvg]])
    memory = deque(maxlen=2000)
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.9910
    learning_rate = 0.005
    tau = .125
    q_max = []
    gamma = 0.9
    markov_moves, dql_moves = [], []

    final_two = '33'

    # Dictionary for markov order 3 (3 occurrences)
    RPS_count = {'000': 3, '001': 3, '002': 3, '010': 3, '011': 3, '012': 3, '020': 3, '021': 3, '022': 3, '100': 3,
                 '101': 3, '102': 3, '110': 3, '111': 3, '112': 3, '120': 3, '121': 3, '122': 3, '200': 3, '201': 3,
                 '202': 3, '210': 3, '211': 3, '212': 3, '220': 3, '221': 3,
                 '222': 3}

    def create_model():
        model = Sequential()
        state_shape = state.shape[1]
        model.add(Dense(24, input_dim=state_shape, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        # output = predicted target value
        model.add(Dense(len([0, 1, 2])))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
        print(model.summary())
        return model

    model = create_model()
    target_model = create_model()

    def act(s, e):
        # this is to take one action
        e = max(epsilon_min, e)
        if np.random.random() < e:
            # return a random move
            return random_move()

        else:
            # return a policy move
            q_max.append(max(model.predict(s)[0]))
            return np.argmax(model.predict(s)[0])

    def reset(): #reset all cumalitive/ trend values
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    def remember(cur_state, action, rwd, new_state):
        # store up a big pool of memory
        memory.append([cur_state, action, rwd, new_state])

    def target_train():
        # transfer weights  proportionally from the action/behave model to the target model
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = (1 - tau) * target_weights[i] + tau * weights[i]
        target_model.set_weights(target_weights)

    def replay():  # DeepMind "experience replay" method
        # the sample size from memory to learn from
        batch_size = 32
        # do nothing untl the memory is large enough
        if len(memory) < batch_size:
            return
        # get the samples   
        samples = random.sample(memory, batch_size)
        # DeepMind "Double" model (Mnih 2015)
        for sample in samples:
            state, action, reward, new_state = sample
            target = target_model.predict(state)
            # print('target at state is ', target)
            Q_future = max(target_model.predict(state)[0])
            TDtarget = reward + Q_future * gamma
            np.append(TDtarget, TDtarget)
            target_delta.append(TDtarget - target[0][action])
            target[0][action] = TDtarget

            # do one pass gradient descend using target as 'label' to train the action model
            model.fit(state, target, epochs=1, verbose=0)

    total_win_count = 0
    total_tie_count = 0
    total_lost_count = 0

    state = reset().reshape(1, state.shape[1])  # initial state

    target_model = create_model()
    # space created to collect TD target for instrumentation
    target_delta, target = [], []

    count = 0

    while count < 100:
        # dql move thrown here
        replay()
        dql_move = act(state, epsilon)
        epsilon *= epsilon_decay  # update epsilon value

        # markov move thrown here
        if final_two[0] == '3':
            m_move = random_move()  # not enough data, return random move
        else:
            r_count = RPS_count[final_two + '0']
            p_count = RPS_count[final_two + '1']
            s_count = RPS_count[final_two + '2']

            tot_count = r_count + p_count + s_count

            q_dist = [r_count / tot_count, p_count / tot_count, 1 - (r_count / tot_count) - (p_count / tot_count)]

            result = [max(q_dist[2] - q_dist[1], 0), max(q_dist[0] - q_dist[2], 0), max(q_dist[1] - q_dist[0], 0)]
            result_norm = sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2])
            result = [result[0] / result_norm, result[1] / result_norm,
                      1 - result[0] / result_norm - result[1] / result_norm]

            m_move = random.uniform(0, 1)

            if m_move <= result[0]:
                m_move = '0'
            elif m_move <= result[0] + result[1]:
                m_move = '1'
            else:
                m_move = '2'
            # update dictionary (memorise opponent dql's move thrown)
            # it is updated AFTER markov's move is already thrown, there is no biases.
            RPS_count[final_two + str(dql_move)] += 1
        # update final two
        final_two = final_two[1] + str(dql_move)

        m_move = int(m_move) # for win tie loss calc
        dql_win, tie, m_win = 0, 0, 0

        if (dql_move - m_move) == 0:
            draws += 1
            total_tie_count += 1
            tie = 1

        if dql_move == 0 and m_move == 1:
            markov_wins += 1
            total_lost_count += 1
            m_win = 1

        if dql_move == 0 and m_move == 2:
            dq_wins += 1
            total_win_count += 1
            dql_win = 1

        if dql_move == 1 and m_move == 0:
            dq_wins += 1
            total_win_count += 1
            dql_win = 1

        if dql_move == 1 and m_move == 2:
            markov_wins += 1
            total_lost_count += 1
            m_win = 1

        if dql_move == 2 and m_move == 0:
            markov_wins += 1
            total_lost_count += 1
            m_win = 1

        if dql_move == 2 and m_move == 1:
            dq_wins += 1
            total_win_count += 1
            dql_win = 1

        dql_moves.append(dql_move)
        markov_moves.append(m_move)

        cumWinRate = total_win_count / (count + 1)
        cumTieRate = total_tie_count / (count + 1)
        cumLostRate = total_lost_count / (count + 1)

        # update moving avg buffer
        winRateBuf.append(cumWinRate)
        tieRateBuf.append(cumTieRate)
        lostRateBuf.append(cumLostRate)

        # calculate trend
        trend = [0, 0, 0]

        if (count + 1) >= window:
            trend[0] = sum(winRateBuf[i] for i in range(window)) / window
            trend[1] = sum(tieRateBuf[i] for i in range(window)) / window
            trend[2] = sum(lostRateBuf[i] for i in range(window)) / window
            # win rate trend analysis
            if winRateMovingAvg < trend[0]:
                winRateTrend = 1  # win rate trending up. + for dql
            else:
                winRateTrend = 0  # win rate trending down. - for dql
            # tie rate trend analysis
            if tieRateMovingAvg < trend[1]:
                tieRateTrend = 1  # tie rate trending up. - for dql (we don't want ties)
            else:
                tieRateTrend = 0  # tie rate trending down.  neither good or bad
            # lost rate trend analysis
            if lostRateMovingAvg < trend[2]:
                lostRateTrend = 1  # lost rate trending up.  - for dql
            else:
                lostRateTrend = 0  # lost rate trending down. + for dql
            winRateMovingAvg, tieRateMovingAvg, lostRateMovingAvg = trend[0], trend[1], trend[2]

            # treat given to dq algorithm if win
            reward = dql_win
            # record the state + reshape it for Keras input format
            dim = state.shape[1]
            state = np.array([dql_win, tie, m_win, winRateTrend, tieRateTrend, lostRateTrend,
                              winRateMovingAvg, tieRateMovingAvg, lostRateMovingAvg]).reshape(1, dim)
            #update state for next move
            remember(state, dql_move, reward, dim)
            target_train()
        count += 1

    print("")
    print("Double-Q Learning Agent vs Markov Agent")
    print("Ties: " + str(draws))
    print("DQL Wins: " + str(dq_wins))
    print("Markov Wins: " + str(markov_wins))
    print("")
    print("")

    print("DQL moves")
    for dql_move in dql_moves:
        print(dql_move)

    print("")

    print("Markov Moves")
    for m_move in markov_moves:
        print(m_move)


mvd()
