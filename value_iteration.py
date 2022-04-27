import matplotlib.pyplot as plt
import random as rand
import math

# We number our states 1 to 99 for each possible amount of capital
S = [i for i in range(1,100)]

# probability of winning the bet
prob = 0.4
# We use gamma = 1, i.e. a discounted problem, theta, the error very small
theta = 10*10**-10
gamma = 1

# This function returns the actions we can possibly take from a state, i.e. how
# much we can bet.
def action_filter(s):
    return [i for i in range(1, min(s, 100 - s) + 1)]

# We begin with a randomised policy.
pi = [rand.choice(action_filter(s)) for s in S]
# Define randomised value function
V = [1000*rand.random() for s in range(100)]
# make sure terminal state has value 0
V[0] = 0

def step():
    condition = True
    i= 0
    while condition:
        # We define error as delta
        delta = 0
        # i is our counter
        i += 1
        for s in S:
            # We copy original V estimate at s
            old_v = V[s]
            v = [0 for i in range(51)]
            # We iterate through all possible values from s
            for a in action_filter(s):
                v[a] = 0
                # We don't need to sum all the Bellman terms as they are all 
                # zero except where s' = s + a or s-a
                if a + s < 100:
                    v[a] += prob*gamma*V[s+a]
                    v[a] += (1-prob)*gamma*V[s-a]
                elif a + s == 100:
                    v[a] += prob
                    v[a] += (1-prob)*gamma*V[s-a]
            v = [round(v[i],3) for i in range(len(v))]
            # We take the maximum as new estimate, standard Bellman equation
            V[s] = max(v)
            # We define policy here to make use of the dummy value function for
            # s = 100. This is a coding nuance and not really relevant to the 
            # mathematics, ideally we improve policy once after policy evaluation
            pi[s-1] = v.index(max(v))
            # We update error value
            delta = max(delta,abs(old_v-V[s]))
        # We stop iteration when error is under a certain bound theta
        if delta < theta:
            condition = False
        #print('Index:{}. Delta:{}'.format(i,delta))
    return pi

# barchart plot
def plot():
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(4)  
    plt.bar(S,pi)
    plt.xlabel("Capital")
    plt.ylabel("Optimal bet")  
    plt.show()  
    
# prints policy action for each state
def visual():
    pairs = [[S[i],pi[i]] for i in range(len(S))]
    return pairs
