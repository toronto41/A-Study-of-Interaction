import random as rand
import matplotlib . pyplot as plt

""" PARAMETERS """
cards = [2,3,4,5,6,7,8,9,10,10,10,10,11]
S = [[i,j,k] for k in range(2) for j in range(2,12) for i in range(12,22)]
A = [0,1]
S_A = [[i,j] for i in S for j in A]
policy = [rand.choice([0,1]) for i in range(len(S))]

""" CLASS FOR DEALER """
# We set out a parent class for the dealer to play, this codes in the key functions
# we need to play blackjack, the player subclass then can iterate the dealers hand
# using this class and iterate the players hand using functions in the subclass
class dealer():
    def __init__(self,n=0):
        # Gives dealer 2 random cards
        self.hand = [rand.choice(cards) for i in range(2)]
        # Totals up dealers hand
        self.total = sum(self.hand)
        # Stores number of aces
        self.aces = self.hand.count(11)
        # Dealers card showing
        self.up = self.hand[0]
        self.starts = []
        self.n = n
    
    # Counts total cards in dealer hand
    def __total__(self):
        while self.total > 21 and self.aces > 0:
            self.total -= 10 
            self.aces -= 1
        return self.total
    
    # Iteration of episode for dealer
    def __play__(self):
        self.__total__()
        if 11 < self.total < 22:
            self.starts.append([self.total,self.aces])        
        while self.total < self.n:
            self.hand.append(rand.choice(cards))
            self.total += self.hand[-1]
            if self.hand[-1] == 11: self.aces += 1
            self.__total__()
            if 11 < self.total < 22: self.starts.append([self.total,self.aces])
        return self.total
            
            
""" INHERITOR CLASS FOR PLAYER """
class player(dealer):
    def __init__(self,start=0):
        # Inherit dealer class
        super().__init__()
        self.hit = 1
        # Exploring start
        if start != 0:
            self.total = start[0]
            self.aces = start[1]
            self.hit = start[2]
        # Dealer policy
        self.D = dealer(17)
        self.up = self.D.up
        # Run dealer episode
        self.D.__play__()
        self.episode = []
     
    # Check for aces
    def ace_check(self):
        if self.aces > 0: return 1
        else: return 0
     
    # Hit function, gives another card to player
    def __hit__(self):
        if self.total < 12: self.hit = 1
        elif self.total < 22: self.hit = policy[S.index([self.total,self.up,self.ace_check()])]
        else: self.hit = 0
        return self.hit
    
    # Reward function
    def __reward__(self):
        if self.total == self.D.total: return 0
        if self.total > 21:
            if self.D.total > 21: return 0
            elif self.D.total < 22: return -1
        elif self.total < 22: 
            if self.D.total > 21: return 1
            elif self.total > self.D.total: return 1
            else: return -1
    
    # Iteration of player episode
    def __play__(self):
        self.__total__()
        self.episode.append([[self.total,self.up,self.ace_check()],self.hit])
        if self.hit:
            self.hand.append(rand.choice(cards))
            self.total += self.hand[-1]
            if self.hand[-1] == 11: self.aces += 1
            self.__total__()
        if not self.hit:
            return self.__reward__()
        while self.hit and self.total < 22:
            self.__hit__()
            self.episode.append([[self.total,self.up,self.ace_check()],self.hit])
            if self.hit == 0: break
            self.hand.append(rand.choice(cards))
            self.total += self.hand[-1]
            if self.hand[-1] == 11: self.aces += 1
            self.__total__()
        return self.__reward__()

""" MONTE CARLO EXPLORING START IMPLEMENTATION """            
def starter(n):
    starts = []
    for i in range(n):
        D = dealer(21)
        D.__play__()
        for elem in D.starts:
            starts.append([elem[0],elem[1],rand.choice([0,1])])
    return starts
    
count = [0 for i in range(400)]
Q = [0 for i in range(400)]
starts = starter(1000000)

# A metric to tell us how function performing
def metric():
    P = player()
    avg = P.__play__()
    for i in range(1,1000000):
        P = player(rand.choice(starts))
        avg = (avg*i+P.__play__())/(i+1)
    return round(avg,4)

""" MONTE CARLO CONTROL ALGORITHM """
def run(n):
    for i in range(n*1000000):
        P = player(rand.choice(starts))
        reward = P.__play__()
        for x in reversed(P.episode):
            n = S_A.index(x)
            Q[n] = (count[n]*Q[n]+reward)/(count[n]+1)
            count[n] += 1
        for j in range(200):
            if Q[2*j] > Q[2*j+1]: policy[j] = 0
            else: policy[j] = 1
        if i % 1000000 == 0:
            print(metric())

""" PLOTS """            
def line(lst1,lst2):
    X = []
    Y = []
    for x in range(2,12):
        n = lst1.index(x)
        y = lst2[n]
        if len(Y) > 1:
            X.append(x-.5)
            Y.append(Y[-1])
            while y > Y[-1]:
                X.append(x-.5)
                Y.append(Y[-1]+1)
            X.append(x-.5)
            Y.append(Y[-1])
            while y < Y[-1]:
                X.append(x-.5)
                Y.append(Y[-1]-1)
        X.append(x)
        Y.append(y)
    for i in range(len(Y)):
        Y[i] -= .5
    return [X,Y]
     
def plot_no_ace():
    ax = plt.axes()
    lst1 = [S[i][1] for i in range(100) if not policy[i]]
    lst2 = [S[i][0] for i in range(100) if not policy[i]]
    L = line(lst1,lst2)
    ax.plot(L[0],L[1],'-k')
    ax.fill_between(L[0],L[1],y2=12)
    ax.set_title("Optimal policy, ace in hand")
    ax.set_ylabel("Player sum")
    ax.set_xlabel("Dealer's card")
    ax.set_yticks([12,13,14,15,16,17,18,19,20,21])
    ax.set_xticks([2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels([2,3,4,5,6,7,8,9,10,'A'])
    #ax.ylim([12,21])
    plt.show()

def plot_ace():
    ax = plt.axes()
    lst1 = [S[i][1] for i in range(100,200) if not policy[i]]
    lst2 = [S[i][0] for i in range(100,200) if not policy[i]]
    L = line(lst1,lst2)
    ax.plot(L[0],L[1],'-k')
    ax.fill_between(L[0],L[1],y2=12)
    ax.set_title("Optimal policy, ace in hand")
    ax.set_ylabel("Player sum")
    ax.set_xlabel("Dealer's card")
    ax.set_yticks([12,13,14,15,16,17,18,19,20,21])
    ax.set_xticks([2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels([2,3,4,5,6,7,8,9,10,'A'])
    #ax.ylim([12,21])
    plt.show()