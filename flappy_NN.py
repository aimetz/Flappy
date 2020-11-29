import pygame
from pygame.math import Vector2
import sys
import random
from NN import *

class Bird:
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)


class NN:
    def __init__(self):
        self.l1 = Layer(4, 6)
        self.l2 = Layer(6, 1)
    def calc(self, inputs):
        self.l1.forward(inputs)
        self.l2.forward(self.l1.output)
        self.output = self.l2.output


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
    cpu1.l1.weights = [[-0.62121055,  0.20920273, -1.04829624, -0.2086307,  -0.73016288,  1.12984472],
                         [ 3.2706438,   0.3521073, -0.46307007,  1.38133011, -0.90566804,  2.13188933],
                         [-1.01787012, -0.44951709, -0.55508441,  1.63629826, -0.84066305, -0.30657673],
                         [-0.11414878, -0.64114299, -1.17127957,  0.40250221, -0.39286808,  0.30460613]]
    cpu1.l2.weights = [[ 1.77925881],
                         [-0.12687917],
                         [-0.66028073],
                         [-1.63558957],
                         [ 0.58434142],
                         [ 0.73491652]]
    cpu1.l1.bias = [[0.69304097, 1.18463632, 1.2512064,  1.59333498, 1.17198556, 0.56375182]]
    cpu1.l2.bias = [[1.01504994]]
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
        cpu(cpu1, bird, walls, width, height)
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
    if walls.x2 is None or bird.position.x-100 < walls.x1 < walls.x2:
        wall = [(walls.x1-bird.position.x)/(width/2), (walls.w1)/height, (bird.position.y)/height, bird.velocity.y]
    else:
        wall = [(walls.x2-bird.position.x)/(width/2), (walls.w2)/height, (bird.position.y)/height, bird.velocity.y]    
    cpu1.calc(wall)
    if cpu1.output < 0:
        bird.velocity = Vector2(0, -.666)


main()