'''
原文链接：https://blog.csdn.net/poisonchry/article/details/121600286
1.如何实现一个简单的Q Learning算法

'''

import numpy as np

# create q table
q_table = np.zeros((5, 5), dtype=np.float32)

# create rule table
rule_table = np.full_like(q_table, -1.0)

# set some col and row to 10, and (4, 4) to 100
#主对角线元素设置为10
for i in range(5):
    rule_table[i, i] = 10
print("rule_table:",rule_table)

# derive rule table with index
def derive_rule_val(state_idx, action_idx):
    rule_val = rule_table[state_idx, action_idx]
    if rule_val == -1:
        return False, rule_val
    else:
        return True, rule_val


# environment function
def derive_updated_q_val(state_idx, action_idx, alpha, gamma):
    # derive the q-value from q table
    q_val = q_table[state_idx, action_idx]

    # derive the rule value from rule table
    ret, rule_val = derive_rule_val(state_idx, action_idx)

    # compute the updated q-value
    if state_idx == 4:
        updated_q_val = (1 - alpha) * q_val + alpha * (rule_val + gamma * np.max(q_table[state_idx]))
    else:
        updated_q_val = (1 - alpha) * q_val + alpha * (rule_val + gamma * np.max(q_table[state_idx + 1]))

    # return the updated q-value
    return ret, updated_q_val


def choose_state_action(state_idx, epsilon, alpha, gamma):
    # choose action
    # if random number less than epsilon, choose random action
    # else choose the action with the highest q-value
    if np.random.random() < epsilon:
        action_idx = np.random.randint(0, 5)
    else:
        action_idx = np.argmax(q_table[state_idx])

    # derive updated q value
    ret, updated_q_val = derive_updated_q_val(state_idx, action_idx, alpha, gamma)

    # update q table
    if ret:
        q_table[state_idx, action_idx] = updated_q_val

    # return the ret
    return ret


if __name__ == "__main__":
    # set some paramters
    episodes = 200
    alpha = 0.1
    gamma = 0.5
    epsilon = 0.1

    # counting the number of steps
    step_count = 0

    # for each episode
    for episode in range(episodes):
        # set the current state
        state_idx = 0

        # set the step count to 0
        step_count = 0

        # while not reach the goal state
        while state_idx < 5:
            # choose action
            ret = choose_state_action(state_idx, epsilon, alpha, gamma)

            # if choose action successfully
            if ret:
                # set the next state
                state_idx = state_idx + 1

            # if choose action unsuccessfully
            else:
                # back to start point
                state_idx = 0

            # increase the step count.
            step_count = step_count + 1

        # print the episode, step count
        print('episode: {}, step count: {}\nq-table:\n{}'.format(episode, step_count, q_table))




