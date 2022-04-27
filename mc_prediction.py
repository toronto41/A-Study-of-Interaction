import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

""" PARAMETERS """
cards = [0,2,3,4,5,6,7,8,9,10,10,10,10]
states = [[i,j,k] for k in range(2) for j in [0]+[2+i for i in range(9)] for \
          i in range(12,22)]

""" BLACKJACK CLASS """
# Essentially is code for the game of blackjack, each episode is run using this
class blackjack():
    def __init__(self):
        # We initialise game by giving the player and dealer 2 random cards
        self.P = random.choices(cards,k=2)
        self.D = random.choices(cards,k=2)
        # D._up is the card the dealer has showing
        self.D_up = self.D[0]
        self.lst = []
        
    # Function to total up a player/dealers hand
    def __total__(self,hand):
        total = sum(hand)
        aces = hand.count(0)
        for i in range(aces):
            if total + 11 <= 21: 
                total += 11
            else: 
                total += 1
        return total
    
    # Function to count how many aces player/dealer has
    def __package__(self):
        if 0 in self.P:
            ace = 1
        else: 
            ace = 0
        return [self.__total__(self.P),self.D_up,ace]
    
    # Playing the game of blackjack using above functions
    def __play__(self):
        cont = True
        while cont == True:
            total = self.__total__(self.P)
            #print(hand,total)
            if total < 20: 
                self.P.append(random.choice(cards))
            else: 
                cont = False
            if 22 > self.__total__(self.P) > 11 and cont == True:
                self.lst.append(self.__package__())
        if total > 21:
            total = 0
        P_score = total
        total = 0
        cont = True
        while cont == True:
            total = self.__total__(self.D)
            #print(hand,total)
            if total < 17: 
                self.D.append(random.choice(cards))
            else: 
                cont = False
        if total > 21:
            total = 0
        D_score = total
        if P_score > D_score:
            return 1
        elif P_score == D_score:
            return 0
        else: 
            return -1
     
""" MONTE CARLO PREDICTION ALGORITHM """
returns = [[] for i in range(400)]
# We iterate over 5 million games of blackjack
for i in range(5000000):
    # We initialise an episode of blackjack
    episode = blackjack()
    # We count reward
    reward = episode.__play__()
    lst = episode.lst
    for state in lst:
        n = states.index(state)
        returns[n].append(reward)
        
# Calculating our value function from final returns
values = [round(sum(i)/len(i),3) if len(i) != 0 else 0 for i in returns]

""" PLOT """

# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# defining all 3 axes
z = np.array(values)
y = [i[0] for i in states]
x = [i[1] for i in states]
 
# Plotting 3-d figure 4.3
ax.plot_trisurf(x, y, z, cmap = 'cividis')
ax.set_title('State values with policy n = 20')
ax.set_xlabel('Dealer\'s card', fontweight ='bold')
ax.set_ylabel('Player sum', fontweight ='bold')
ax.set_zlabel('State value', fontweight ='bold')
ax.set_zticks([-1,0,1])
ax.set_xticks([0,10])
ax.set_xticklabels(['A',10])
ax.set_yticks([12,21])
plt.show()
