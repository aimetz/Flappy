import sys
import random
import numpy as np
from NN import Layer
import pygame

class Bird:
    def __init__(self, x, y, num_birds):
        self.p = [Vector(x, y)] * num_birds
        self.v = [Vector(0.0, 0.0)] * num_birds
    
    def delete(self, i):
        self.p.pop(i)
        self.v.pop(i)
    
    def get(self, i):
        return (self.p[i], self.v[i])

    def update_v(self, i):
        self.v[i] = Vector(0.0, -.666)

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        if type(self) == type(other) == Vector:
            return Vector(self.x + other.x, self.y+other.y)
        else:
            raise TypeError
    
    def __repr__(self):
        return "Vector({}, {})".format(self.x, self.y)

class Wall:
    def __init__(self, width, height):
        self.w1 = random.randint(.3*height//1, .9*height//1)
        self.w2 = random.randint(.3*height//1, .9*height//1)
        self.x1 = width
        self.x2 = None
    
    def step(self, velocity, width, height):
        self.x1 += velocity
        if self.x2 is None and self.x1<width/2:
            self.x2 = width
        if self.x2 is not None:
            self.x2 += velocity
        if self.x1 < -75:
            self.w1 = random.randint(.3*height//1, .9*height//1)
            self.x1 = width
        if self.x2 is not None and self.x2 < -75:
            self.w2 = random.randint(.3*height//1, .9*height//1)
            self.x2 = width        

class NN:
    def __init__(self):
        self.l1 = Layer(4, 6)
        self.l2 = Layer(6, 1)
    def calc(self, inputs):
        self.l1.forward(inputs)
        self.l2.forward(self.l1.output)
        self.output = self.l2.output

class Strat:
    def __init__(self, l1w, l2w, l1b, l2b, i=0):
        self.w1 = l1w
        self.w2 = l2w
        self.b1 = l1b
        self.b2 = l2b
        self.fit = i
    
    def __lt__(self, other):
        return self.fit<other.fit
        
    def __add__(self, other):
        return Strat((self.w1+other.w1)/2,(self.w2+other.w2)/2, (self.b1+other.b1)/2, (self.b2+other.b2)/2)
        
    def __repr__(self):
        return "{}\n{}\n{}\n{}\n{}\n".format(self.w1, self.w2, self.b1, self.b2, self.fit) 
    

def draw_wall(wall_height, x, hole, color, screen, height):
    pygame.draw.rect(screen, color, (x, 0, hole, wall_height-2.4*hole))
    pygame.draw.rect(screen, color, (x, wall_height, hole, height - wall_height))



def train(num, strats=None):
    #pygame.init() # Uncomment for graphics

    width = 800
    height = 600
    #screen = pygame.display.set_mode((width, height)) # Uncomment for graphics

    green = (0,200,112)
    white = (255, 255, 255)
    red = (235, 0, 0)
    if strats is not None: # otherwise num wil be initial number of players
        print("Generation", num) # uses num to also store generation if not initial batch
        num = len(strats)
    best = []
    saved = open("saved_strategies.txt", "a")
    players = [None] * num
    for lmk in range(num):
        players[lmk] = NN()
        if strats is not None: # sets each player to strategy in strats
            players[lmk].l1.weights = strats[lmk].w1
            players[lmk].l2.weights = strats[lmk].w2
            players[lmk].l1.bias = strats[lmk].b1
            players[lmk].l2.bias = strats[lmk].b2
    #if strats is None: # Starts training with a strategy previously found
    #    for l in range(3):
    #        players[l].l1.weights = np.asmatrix(np.array([[ 0.13683738,  0.15444842, -0.21799054, -0.43945095, -1.35703313, -0.18130608],
    #                                     [-1.76681828,  0.85032577, -1.82620322, -1.36902304, -0.30476928,  0.77921532],
    #                                     [ 1.38061683, -2.12395059, -0.744464,    0.2213408,   0.31173999, -1.3975062 ],
    #                                     [ 0.5612241,  -0.15981751,  0.50040377, -0.7247977,   1.0656997,  -0.24192157]]))
    #        players[l].l2.weights = np.asmatrix(np.array([[-0.78295969],
    #                                 [ 2.77529467],
    #                                 [-0.81890514],
    #                                 [-1.09254459],
    #                                 [ 0.47598673],
    #                                 [ 0.11870738]]))
    #        players[l].l1.bias = np.asmatrix(np.array([[0.73889266, 0.51201123, 1.19816538, 1.00068603, 0.54369439, 0.98781531]]))
    #        players[l].l2.bias = np.asmatrix(np.array([[0.57109063]]))
    size = 25
    speed = .2
    gravity = Vector(0.0, .001666)
    birds = Bird(width /5, height/2, num)
    i = 0
    walls = Wall(width, height)
    done = []

    while len(players) > len(done):
        for lol in range(len(players)):
            if players[lol] is not None:
                #Plays game for each NN in players[], once it dies adds that player to done and sets its index to None in players[]
                if i % 10:
                    cpu(players[lol], birds, walls, width, height, lol)
                if -95<(walls.x1-width/5)<20:
                    if not (22<(walls.w1-birds.p[lol].y)<155):
                        done.append(Strat(players[lol].l1.weights, players[lol].l2.weights, players[lol].l1.bias, players[lol].l2.bias, i))
                        players[lol] = None
                elif walls.x2 is not None and -95<(walls.x2-width/5)<20:
                    if not (22<(walls.w2-birds.p[lol].y)<155):
                        done.append(Strat(players[lol].l1.weights, players[lol].l2.weights, players[lol].l1.bias, players[lol].l2.bias, i))
                        players[lol] = None
                elif not (25 < birds.p[lol].y < height - 25):
                    done.append(Strat(players[lol].l1.weights, players[lol].l2.weights, players[lol].l1.bias, players[lol].l2.bias, i))
                    players[lol] = None
                birds.v[lol] += gravity
                birds.p[lol] += birds.v[lol]
        if i > 100000:#Goal, if reached print and save strategy and exit
            for strat in players:
                if strat is not None:
                    print("{}\n{}\n{}\n{}\n\n".format(strat.l1.weights, strat.l2.weights, strat.l1.bias, strat.l2.bias))
                    saved.write("{}\n{}\n{}\n{}\n\n".format(strat.l1.weights, strat.l2.weights, strat.l1.bias, strat.l2.bias))
                    sys.exit()
        #Uncomment all below for graphics... Much slower
        #screen.fill(white) #fills screen black is actually white
        walls.step(-.333, width, height)
        #draw_wall(walls.w1, walls.x1, 75, red, screen, height)
        #if walls.x2 is not None:
        #    draw_wall(walls.w2, walls.x2, 75, red, screen, height)
        #for lmk in range(len(players)):
        #    if players[lmk] is not None:
        #        pygame.draw.circle(screen, green, (birds.p[lmk].x, birds.p[lmk].y), size)
        i += 1 # counts number of loops
        #pygame.display.update()
    for strat in done:
        if strat.fit > 50000:
            print(strat.fit)
        if strat.fit > 80000:
            print("{}\n{}\n{}\n{}\n{}\n\n".format(strat.w1, strat.w2, strat.b1, strat.b2, strat.fit))
            saved.write("{}\n{}\n{}\n{}\n{}\n\n".format(strat.w1, strat.w2, strat.b1, strat.b2, strat.fit))
    for a in range(5):
        good = max(done)
        done.remove(good)
        best.append(good)
    # Mutates best strategies
    best.append(best[0]+best[1])
    best.append(best[0]+best[2])
    best.append(best[1]+best[2])
    best.append(best[0]+best[3])
    best.append(best[0]+best[4])
    best.append(best[1]+best[3])
    best.append(best[1]+best[4])
    best.append(best[2]+best[3])
    best.append(best[2]+best[4])
    best.append(best[3]+best[4])
    for a in range(5):
        for b in range(3):
            #Adds 15 New randoms
            d = NN()
            best.append(Strat(d.l1.weights, d.l2.weights, d.l1.bias, d.l2.bias))
        for c in range(0, 100, 5):
            # 2nd mutation strategy
            best.append(Strat(best[a].w1+.001*c*np.random.randn(4, 6), best[a].w2+.001*c*np.random.randn(6, 1), 
                              best[a].b1+.001*c*np.random.randn(1, 6), best[a].b2+.001*c*np.random.randn(1, 1)))

    saved.close()
    return best


#calculates if individual bird should jump, pushes normalized inputs through corresponding NN
def cpu(cpu1, birds, walls, width, height, i):
    bird = birds.get(i)
    if walls.x2 is None or bird[0].x-100 < walls.x1 < walls.x2:
        wall = [(walls.x1-bird[0].x)/(width/2), (walls.w1)/height, (bird[0].y)/height, bird[1].y]
    else:
        wall = [(walls.x2-bird[0].x)/(width/2), (walls.w2)/height, (bird[0].y)/height, bird[1].y]    
    cpu1.calc(wall)
    if cpu1.output < 0:
        birds.v[i] = Vector(0, -.666)


# starts with 1000 in first batch and goes through 1000 generagtions or until goal is reached
strats = train(1000)
for i in range(1000):
    strats = train(i, strats)