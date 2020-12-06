import pygame
from pygame.math import Vector2
import sys
import random
from NN import Layer

class Bird:
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)


class NN:
    def __init__(self):
        self.l1 = Layer(2, 4)
        self.l2 = Layer(4, 4)
        self.l3 = Layer(4, 1)
    def calc(self, inputs):
        self.l1.forward(inputs)
        self.l1.lin_rect_act()
        self.l2.forward(self.l1.output_act)
        self.l2.lin_rect_act()
        self.l3.forward(self.l2.output_act)
        self.output = self.l3.output


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

def draw_wall(wall_height, x, hole, color, screen, height):
    pygame.draw.rect(screen, color, (x, 0, hole, wall_height-2.4*hole))
    pygame.draw.rect(screen, color, (x, wall_height, hole, height - wall_height))
    

def main():
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
    cpu1.l1.weights = [[-0.75693301, -0.15557033, -0.35002748, -0.16915722],
                       [-1.32813076,  1.17819187, -1.74604785, -0.16535369]]
    cpu1.l2.weights = [[ 2.53648758,  0.9660128, -0.01512735, -1.01101749],
                       [ 0.02425231, -1.32983301,  0.94976883, -0.63879104],
                       [-0.09090483,  1.423794,   -0.34958564,  1.33530054],
                       [-0.36168222, -2.11175652, -0.5490942,  -0.36594807]]
    cpu1.l3.weights = [[-0.38339659],
                       [ 0.46510374],
                       [ 0.51485256],
                       [-0.9898644]]
    cpu1.l1.bias = [[0.9173779,  1.07025258, 1.05870736, 0.86454876]]
    cpu1.l2.bias = [[0.98092574, 1.06569193, 0.91442385, 0.96617903]]
    cpu1.l3.bias = [[0.8556708]]
    cpu2 = NN()
    cpu2.l1.weights = [[-1.15516778,  0.21464661, -0.38963488,  0.4350685],
                       [-0.43415127,  0.91864609,  0.82807051, -0.28983477]]
    cpu2.l2.weights = [[ 0.37657839,  1.59273264, -0.71196742,  0.27452368],
                       [ 0.97264203,  0.65755141,  0.0532049,   1.44267865],
                       [-3.65941156, -2.24901854, -1.03273759, -0.0131793],
                       [ 0.83181616, -2.28187206,  0.00855971, -0.0928179]]
    cpu2.l3.weights = [[-0.65949501],
                       [-0.02305565],
                       [ 0.04065686],
                       [ 1.01944908]]
    cpu2.l1.bias = [[ 0.08277388,  0.03947458,  0.0962351,  -0.14355545]]
    cpu2.l2.bias = [[-0.13598934,  0.03012618,  0.1163572, -0.02422486]]
    cpu2.l3.bias = [[-0.10321254]]


    
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
        if i % 10:
            cpu(cpu2, bird, walls, width, height)
        if walls.x2 is not None:
            draw_wall(walls.w2, walls.x2, 75, red, screen, height)
        if -95<(walls.x1-width/5)<20:
            if not (22<(walls.w1-bird.position[1])<155):
                sys.exit()
        if walls.x2 is not None and -95<(walls.x2-width/5)<20:
            if not (22<(walls.w2-bird.position[1])<155):
                sys.exit()
        bird.velocity += gravity
        bird.position += bird.velocity
        pygame.draw.circle(screen, green, bird.position, size)      
        i += 1 # counts number of loops
        pygame.display.update()



def cpu(cpu1, bird, walls, width, height):
    if walls.x2 is None or bird.position.x-100 < walls.x1 < walls.x2 or bird.position.x-100 > walls.x2  :
        wall = [(walls.x1-bird.position.x)/(width/2), (walls.w1 - bird.position.y)/height]
    else:
        wall = [(walls.x2-bird.position.x)/(width/2), (walls.w2 - bird.position.y)/height]    
    cpu1.calc(wall)
    if cpu1.output < 0:
        bird.velocity = Vector2(0, -.666)


main()