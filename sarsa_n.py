import matplotlib.pyplot as plt
import random
import sys

""" PARAMETERS """

# Actions left, right, up, and down which we label via coord change.
actions = [[1,0],[-1,0],[0,1],[0,-1]]

# Have states in 11 x 11 grid, marked 0 to 10.
states = [[i,j] for i in range(11) for j in range(11)]

# We combine actions and states into a list of pairs.
state_actions = [[state,action] for state in states for action in actions]

# We set our default start and goal as the state (note these will be random)
start = [1,5]
goal = [10,5]

# We set our greedy parameter and our discounting factor
epsilon = 0.01
gamma = .9
# The below formats our walls.
walls = [[[5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10]],
  [[7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9]],
  [[1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8]],
  [[9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7]],
  [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7]]]

X = [2*i+1 for i in range(5)]
for elem in walls:
    for i in elem:
        states.remove(i)
        for a in actions:
            state_actions.remove([i,a])

# The below loop removes any state action combinations that lead to a state 
# outside of our 11 x 11 grid. We then create a state-action values list which 
# matches this list one to one.
for i in range(5):
    for elem in state_actions:
        state = elem[0]
        action = elem[1]
        new_state = [state[0]+action[0],state[1]+action[1]]
        n = state_actions.index(elem)
        if new_state not in states:
            state_actions.remove(elem)

# We take a copy of our state-actions pairs (useful later)
record = [elem[0] for elem in state_actions]

# We now initialise our estimates Q.
S_A_values = [0 for i in range(len(state_actions))]

# This function initialises our policy.
def reset_policy():
    policy = []
    for state in states:
        while True:
            action = random.choice(actions)
            if [action[0]+state[0],action[1]+state[1]] in states:
                break
        policy.append(action)
    return policy
    
policy = reset_policy()

""" SARSA(N) IMPLEMENTATION """

# For when our agent needs to select a random action (for example exploring starts).
def random_action(state):
    new_state = 0
    while new_state not in states:
            action = random.choice(actions)
            new_state = [state[0]+action[0],state[1]+action[1]]
    return action

# Telling our learning agent how to choose actions (using policy).
def choose_action(state):
    r = random.random()
    if r > epsilon:
        n = states.index(state)
        action = policy[n]
    else: 
        action = random_action(state)
    return action

# Reward function.
def reward(state):
    if state == goal:
        return 0
    else:
        return -1
      
# This function improves our policy.
def improvement():
    record = [elem[0] for elem in state_actions]
    for state in states:
        options = []
        n = record.index(state)
        for i in range(record.count(state)):
            options.append(S_A_values[n+i])
        k = options.index(max(options))
        p = states.index(state)
        policy[p] = state_actions[n+k][1]
        
# This is the actual iteration of Sarsa(n), drawing on all the other functions.
def walk(start,goal,step,alpha = 0.5):
    n = step
    S = start
    A = choose_action(S)
    S_lst = [S]
    A_lst = [A]
    R_lst = []
    t = 0
    T = 1000000
    tau = -1
    while tau != T-1:
        if t < T:
            new_S = [S[0]+A[0],S[1]+A[1]]
            R = reward(new_S)
            R_lst.append(R)
            S_lst.append(new_S)
            S = new_S
            if new_S == goal:
                T = t+1
            else:
                new_A = choose_action(new_S)
                A_lst.append(new_A)
                A = new_A
        tau = t - n + 1
        if tau >= 0:
            G = sum([(gamma**(i-tau-1))*R_lst[i] for i in range(tau+1,min([tau+n,T]))])
            if tau + n < T:
                k = state_actions.index([S_lst[tau+n],A_lst[tau+n]])
                G += (gamma**n)*S_A_values[k]
            p = state_actions.index([S_lst[tau],A_lst[tau]])
            S_A_values[p] += alpha*(G - S_A_values[p])
            improvement()
        t += 1
    return t

# This generates a random starting state.
def starter():
    choices = states.copy()
    choices.remove(goal)
    return random.choice(choices)

# This function essentially iterates Sarsa(n) however many times as instructed,
# run(number of iterations, if you want to print all outputs, improvement plot?).
def run(n,print_,plot):
    x = [0]
    y = [0]    
    total = 0
    for i in range(1,n):
        #walk(start,goal)
        #print(walk(start,goal))
        step = 3
        count = walk(starter(),goal,step)
        if print_:
            print(count)
        total += count
        x.append(total)
        y.append(i)
        if i%1000 == 0:
            print(i)
    if plot == True:
        plt.title("Episodes against timesteps for Sarsa({}): Total = {}".format(step,total))
        plt.xlabel("Total timesteps")
        plt.ylabel("Episodes")
        plt.plot(x,y,'k-')
        plt.show()
    return total

# This records our agents movements so we can plot it back later and get a nice plot.
def recorder(start,goal):
    S = start
    n = states.index(S)
    A = policy[n]
    lst = [S]
    i = 0
    while S != goal:
        new_S = [S[0]+A[0],S[1]+A[1]]
        n = states.index(new_S)
        new_A = policy[n]
        S = new_S
        A = new_A 
        lst.append(S)
        if i > 5000:
            raise ValueError
        i += 1
    return lst

# Making the nice plots aforementioned.
def visual(X,start):
    start = start
    if X == True:
        try:
            lst = recorder(start,goal)
        except:
            print("Houston, we have a problem.")
            return None
        x = [state[0] for state in lst]
        y = [state[1] for state in lst]
        for i in range(len(lst)):
            plt.plot(x[i:i+2],y[i:i+2],'k-')
            plt.scatter(start[0],start[1],c='red',s=200)
    plt.scatter(goal[0],goal[1],c='green',s=200)
    for elem in walls:
        copy = elem.copy()
        ys = [i[1] for i in copy]
        if 0 in ys:
            copy.reverse()
            copy.append([copy[0][0],-1])
            copy.reverse()
        elif 10 in ys:
            copy.append([copy[0][0],11])
        plt.plot([x[0] for x in copy],[x[1] for x in copy],'k-', linewidth=3.0)
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10])
    # plt.xticks(color='w')
    # plt.yticks(color='w')
    plt.xlim(-1,11)
    plt.ylim(-1,11)
    plt.plot([-1,-1],[-1,11],'k-',linewidth=4.0)
    plt.plot([-1,11],[11,11],'k-',linewidth=4.0)
    plt.plot([11,11],[11,-1],'k-',linewidth=4.0)
    plt.plot([11,-1],[-1,-1],'k-',linewidth=4.0)
    plt.grid()
    plt.title("Optimal route")
    plt.show()

