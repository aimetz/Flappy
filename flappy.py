import pygame
from pygame.math import Vector2
import sys
import random

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


main()