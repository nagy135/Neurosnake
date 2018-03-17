import time
import pygame
import math
import random

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (0, 255, 0)
WIDTH = 1000
HEIGHT = 1000

START_X = 10
START_Y = 10
MAX_X = 19
MAX_Y = 19
STEP_SIZE = 50

class Game(object):

    def __init__(self):
        pygame.init()
        self.gameDisplay = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('NeuroSnake')
        self.clock = pygame.time.Clock()
        self.body_parts = list()
        self.body_parts.append((START_X, START_Y))
        self.orientation = (1,1)
        self.time_step = 0.100
        self.last_time = time.time()

        self.food = (random.randint(0,19),random.randint(0,19))
        self.food_time = time.time()
        self.food_wait_time = 5


    def get_fitness(self):
        pass

    def draw(self):
        # pygame.draw.line(self.gameDisplay, black, (self.anchor_x, self.anchor_y),(self.head_x,self.head_y), 5)
        # pygame.draw.circle(self.gameDisplay, black, (int(self.head_x),int(self.head_y)), BALANCER_HEAD_RADIUS)
        # pygame.draw.circle(self.gameDisplay, red, (int(self.ball_x),int(self.ball_y)), BALL_RADIUS)
        # pygame.draw.line(self.gameDisplay, black, (self.ball_x-pri, self.ball_y+pro),(self.ball_x, self.ball_y), 5)
        for body_part in self.body_parts:
            pygame.draw.rect(self.gameDisplay, black, (STEP_SIZE*body_part[0], STEP_SIZE*body_part[1], STEP_SIZE, STEP_SIZE), 100)
        pygame.draw.rect(self.gameDisplay, green, (self.food[0]*STEP_SIZE, self.food[1]*STEP_SIZE, STEP_SIZE, STEP_SIZE), 100)

    def move(self):
        for i in range(len(self.body_parts-2), 0):
            #TODO
            self.body_parts[i] = self.body_parts[i-1]
        self.body.pop()
        self.body_parts[0] = (body_part[0] + self.orientation[0], body_part[1] + self.orientation[1])
        for i,body_part in enumerate(self.body_parts):
            self.body_parts[i] = (body_part[0] + self.orientation[0], body_part[1] + self.orientation[1])

    def tick(self):
        now = time.time()
        if abs(self.last_time - now) > self.time_step:
            self.last_time = now
            self.move()
            self.move_time()

    def move_time(self):
        time_now = time.time()
        if abs(self.food_time - time_now > self.food_wait_time):
            self.food_time = time_now
            self.food = (random.randint(0,19),random.randint(0,19))

    def start(self):
        end = False
        while not end:
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        if self.orientation[0] != 1:
                            self.orientation = (-1,0)
                    if event.key == pygame.K_RIGHT:
                        if self.orientation[0] != -1:
                            print('pressed it')
                            self.orientation = (1,0)
                    if event.key == pygame.K_UP:
                        if self.orientation[1] != 1:
                            self.orientation = (0,-1)
                    if event.key == pygame.K_DOWN:
                        if self.orientation[1] != -1:
                            self.orientation = (0,1)
                    if event.key == pygame.K_q:
                        pass
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                            print('pressed it')
                    if event.key == pygame.K_RIGHT:
                        pass
                if event.type == pygame.QUIT:
                    end = True
            self.tick()
            self.gameDisplay.fill(white)
            self.draw()
            pygame.display.update()
            self.clock.tick(60)
a = Game()
a.start()
