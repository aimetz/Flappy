import pygame
from pygame.math import Vector2
import sys
import random
import numpy as np
from NN import Layer

class Bird:
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)

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

def draw_wall(wall_height, x, hole, color, screen, height):
    pygame.draw.rect(screen, color, (x, 0, hole, wall_height-2.4*hole))
    pygame.draw.rect(screen, color, (x, wall_height, hole, height - wall_height))


class State:
    def __init__(self, l1w, l2w, l1b, l2b, i=0):
        self.w1 = l1w
        self.w2 = l2w
        self.b1 = l1b
        self.b2 = l2b
        self.fit = i
    
    def __lt__(self, other):
        return self.fit<other.fit
        
    def __add__(self, other):
        return State((self.w1+other.w1)/2,(self.w2+other.w2)/2, (self.b1+other.b1)/2, (self.b2+other.b2)/2, "Comb, Meth1")
        
    def comb(self, other):
        return State(self.w1, other.w2, self.b1, other.b2, "Comb, Meth2")
        
    def __repr__(self):
        return "{}\n{}\n{}\n{}\n{}\n".format(self.w1, self.w2, self.b1, self.b2, self.fit) 
    

def main():
    best = []
    states = [None] * 75
    saved = open("saved_strategies.txt", "w")
    for q in range(75):
        print(q)
        pygame.init()

        width = 800
        height = 600
        screen = pygame.display.set_mode((width, height))

        game_over = False
        green = (0,200,112)
        white = (255, 255, 255)
        red = (235, 0, 0)
        size = 25
        speed = .2
        gravity = Vector2(0.0, .001666)
        bird = Bird(width /5, height/2)
        i = 0
        walls = Wall(width, height)
        cpu1 = NN()
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        bird.velocity = Vector2(0, -.666)
            screen.fill(white) #fills screen black is actually white
            walls.step(-.333, width, height)
            draw_wall(walls.w1, walls.x1, 75, red, screen, height)
            if i %10 == 0:    
                cpu(cpu1, bird, walls, width, height)
            if walls.x2 is not None:
                draw_wall(walls.w2, walls.x2, 75, red, screen, height)
            if -95<(walls.x1-width/5)<20:
                if not (22<(walls.w1-bird.position[1])<155):
                    game_over = True
            if walls.x2 is not None and -95<(walls.x2-width/5)<20:
                if not (22<(walls.w2-bird.position[1])<155):
                    game_over = True
            if not (25 < bird.position.y < height - 25):
                game_over = True
            bird.velocity += gravity
            bird.position += bird.velocity
            pygame.draw.circle(screen, green, bird.position, size)
            i += 1 # counts number of loops
            pygame.display.update()
        states[q] = State(cpu1.l1.weights, cpu1.l2.weights, cpu1.l1.bias, cpu1.l2.bias, i)
    for a in range(5):
        good = max(states)
        states.remove(good)
        best.append(good)
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
        for b in range(2):
            d = NN()
            best.append(State(best[a].w1+.1*np.random.randn(4, 6), best[a].w2+.1*np.random.randn(6, 1), best[a].b1+.1*np.random.randn(1, 6), best[a].b2+.1*np.random.randn(1, 1)))
            best.append(State(d.l1.weights, d.l2.weights, d.l1.bias, d.l2.bias))
    for q in range(50):
        print(q, 2)
        states = []
        for s in best:
            print(s.fit)
            pygame.init()

            width = 800
            height = 600
            screen = pygame.display.set_mode((width, height))

            game_over = False
            green = (0,200,112)
            white = (255, 255, 255)
            red = (235, 0, 0)
            size = 25
            speed = .2
            gravity = Vector2(0.0, .001666)
            bird = Bird(width /5, height/2)
            i = 0
            walls = Wall(width, height)
            cpu1 = NN()
            cpu1.w1 = s.w1
            cpu.w2 = s.w2
            cpu.b1 = s.b1
            cpu.b2 = s.b2
            while not game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            bird.velocity = Vector2(0, -.666)
                screen.fill(white) #fills screen black is actually white
                walls.step(-.333, width, height)
                draw_wall(walls.w1, walls.x1, 75, red, screen, height)
                if i %10 == 0:    
                    cpu(cpu1, bird, walls, width, height)
                if walls.x2 is not None:
                    draw_wall(walls.w2, walls.x2, 75, red, screen, height)
                if -95<(walls.x1-width/5)<20:
                    if not (22<(walls.w1-bird.position[1])<155):
                        game_over = True
                if walls.x2 is not None and -95<(walls.x2-width/5)<20:
                    if not (22<(walls.w2-bird.position[1])<155):
                        game_over = True
                if not (25 < bird.position.y < height - 25):
                    game_over = True
                bird.velocity += gravity
                bird.position += bird.velocity
                pygame.draw.circle(screen, green, bird.position, size)
                i += 1 # counts number of loops
                pygame.display.update()
            states.append(State(cpu1.l1.weights, cpu1.l2.weights, cpu1.l1.bias, cpu1.l2.bias, i))
            if i > 8000:
                print("{}\n{}\n{}\n{}\n{}\n\n".format(cpu1.l1.weights, cpu1.l2.weights, cpu1.l1.bias, cpu1.l2.bias, i))
                saved.write("{}\n{}\n{}\n{}\n{}\n\n".format(cpu1.l1.weights, cpu1.l2.weights, cpu1.l1.bias, cpu1.l2.bias, i))
        best = []
        for a in range(8):
            good = max(states)
            states.remove(good)
            best.append(good)
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
        best.append(best[1]+best[2]+best[3]+best[4]+best[0])
        best.append(best[0].comb(best[1]))
        best.append(best[1].comb(best[0]))
        best.append(best[2].comb(best[3]))
        best.append(best[0].comb(best[3]))
        best.append(best[0].comb(best[2]))
        best.append(best[1].comb(best[4]))

        for a in range(5):
            d = NN()
            best.append(State(d.l1.weights, d.l2.weights, d.l1.bias, d.l2.bias, "random"))
            for b in range(10):
                best.append(State(best[a].w1+.002*b*np.random.randn(4, 6), best[a].w2+.002*b*np.random.randn(6, 1), best[a].b1+.002*b*np.random.randn(1, 6), best[a].b2+.002*b*np.random.randn(1, 1), "tweaked"))
    print(best)
    saved.close()



def cpu(cpu1, bird, walls, width, height):
    if walls.x2 is None or bird.position.x-100 < walls.x1 < walls.x2:
        wall = [(walls.x1-bird.position.x)/(width/2), (walls.w1)/height, (bird.position.y)/height, bird.velocity.y]
    else:
        wall = [(walls.x2-bird.position.x)/(width/2), (walls.w2)/height, (bird.position.y)/height, bird.velocity.y]    
    cpu1.calc(wall)
    if cpu1.output < 0:
        bird.velocity = Vector2(0, -.666)


main()