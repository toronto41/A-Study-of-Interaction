import matplotlib.pyplot as plt
import random
import sys

""" PARAMETERS """
# We have actions left, right, up, and down which we label via coord change.
actions = [[1,0],[-1,0],[0,1],[0,-1]]
# Have states in 11 x 11 grid, marked 0 to 10..
states = [[i,j] for i in range(11) for j in range(11)]
# Create a list of state-action pairs.
state_actions = [[state,action] for state in states for action in actions]
# We create the greedy-selection parameter, step-size constant and discount factor.
epsilon = 0.1
alpha = 0.5
gamma = .95
# We set default start and goal of our maze, will be randomised each iteration.
start = [1,5]
goal = [10,5]

# The below code sets up our maze walls.
walls = [[[5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10]],
 [[7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9]],
 [[1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8]],
 [[9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7]],
 [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7]]]
for elem in walls:
    for i in elem:
        states.remove(i)
        for a in actions:
            state_actions.remove([i,a])

# Record of states for use in later algorithm
record = [elem[0] for elem in state_actions]

#The below loop removes any state action combinations that lead to a state 
#outside of our 11 x 11 grid. We then create a state-action values list which 
#matches this list one to one.      
for i in range(5):
    for elem in state_actions:
        state = elem[0]
        action = elem[1]
        new_state = [state[0]+action[0],state[1]+action[1]]
        n = state_actions.index(elem)
        if new_state not in states:
            state_actions.remove(elem)

# Record of states again
record = [elem[0] for elem in state_actions]
# Initiate Q initial estimates
S_A_values = [0 for i in range(len(state_actions))]
# Create empty policy
policy = []

# When creating a random policy we make sure to exclude unviable destinations
for state in states:
    n = record.index(state)
    poss_actions = []
    for i in range(record.count(state)):
        poss_actions.append(state_actions[n+i][1])
    policy.append(random.choice(poss_actions))
    
""" SARSA(0) IMPLENTATION """   

# Random action selection for exploring starts
def random_action(state):
    new_state = 0
    while new_state not in states:
            action = random.choice(actions)
            new_state = [state[0]+action[0],state[1]+action[1]]
    return action

# Action selection via policy
def choose_action(state):
    r = random.random()
    if r > epsilon:
        n = states.index(state)
        action = policy[n]
    else: 
        action = random_action(state)
    return action

# Reward function
def reward(state):
    if state == goal:
        return 0        
    else:
        return -1

# Policy improvement mechanism
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
 
# Implentation of the actual algorithm, drawing upon all other functions
def walk(start,goal):
    S = start
    A = random_action(S)
    count = 0
    while S != goal:
        new_S = [S[0]+A[0],S[1]+A[1]]
        R = reward(new_S)
        new_A = choose_action(new_S)
        n0 = state_actions.index([S,A])
        n1 = state_actions.index([new_S,new_A])
        S_A_values[n0] += alpha*(R + S_A_values[n1] - S_A_values[n0])
        #print(S,A,R,new_S,new_A)
        S = new_S
        A = new_A 
        count += 1
        improvement()
    return count

# Random start selection.
def starter():
    choices = states.copy()
    choices.remove(goal)
    return random.choice(choices)

# Iterates algorithm n times and then produced print of improvements    
def run(n,print_,plot):
    x = [0]
    y = [0]    
    total = 0
    for i in range(1,n):
        #walk(start,goal)
        #print(walk(start,goal))
        count = walk(starter(),goal)
        if print_:
            print(count)
        total += count
        x.append(total)
        y.append(i)
        if i%1000 == 0:
            print(i)
    if plot == True:
        plt.title("Episodes against timesteps for Sarsa(0): Total = {}".format(total))
        plt.xlabel("Total timesteps")
        plt.ylabel("Episodes")
        plt.plot(x,y,'k-')
        plt.show()
        
def performance(): 
    X = []
    run(10000,False,False)
    for j in range(100):
        lst = []
        for i in range(1000):
            count = walk(starter(),goal)
            lst.append(count)
        X.append(sum(lst)/len(lst))
    return X


# Records agents path through maze so we can create nice plots later
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

# Visualisation of our agents movement through the maze.
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