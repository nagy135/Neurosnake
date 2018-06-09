import tensorflow as tf
import copy
import time
import os
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
BODY = [(2,2), (3,2), (4,2), (5,2),(6,2)]
DX = 1
DY = 0
MAX_X = 19
MAX_Y = 19
STEP_SIZE = 50
TIME_STEP_SIZE = 0.04

# os.environ['SDL_VIDEO_WINDOW_POS'] = "{},{}".format(0,0)

class Game(object):

    def __init__(self):
        pygame.init()
        self.gameDisplay = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('NeuroSnake')
        self.clock = pygame.time.Clock()
        self.body_parts = copy.deepcopy(BODY)
        self.dx = DX
        self.dy = DY
        self.tick_time = time.time()
        self.food = (random.randint(0,19),random.randint(0,19))
        self.food_time = time.time()
        self.food_wait_time = 15

        ## NEURAL PARAMS
        self.classes = 3

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def maxpool2d(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def model(self, inner_x):
        self.x = tf.placeholder('float',[None, 400])
        self.y = tf.placeholder('float')
        self.keep_rate = 0.8
        self.keep_prob = tf.placeholder(tf.float32)
        weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
                   'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
                   'out':tf.Variable(tf.random_normal([1024, n_classes]))}

        biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                  'b_conv2':tf.Variable(tf.random_normal([64])),
                  'out':tf.Variable(tf.random_normal([n_classes]))}

        inner_x = tf.reshape(inner_x, shape=[-1, 20, 20, 1])

        conv1 = tf.nn.relu(conv2d(inner_x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = maxpool2d(conv1)

        conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = maxpool2d(conv2)

        fc = tf.reshape(conv2,[-1, 7*7*64])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out'])+biases['out']

        return output

    def cost_function(self, prediction):
        return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, self.y) )

    def train_neural_network(self, frame_x, frame_y):
        prediction = self.model(self.x)
        cost = self.cost_function(prediction)
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            _, c = sess.run([optimizer, cost], feed_dict={x: frame_x, y: frame_y})
            loss += c
            print('Loss after frame : ' + str(loss))

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    def draw(self):
        # pygame.draw.line(self.gameDisplay, black, (self.anchor_x, self.anchor_y),(self.head_x,self.head_y), 5)
        # pygame.draw.circle(self.gameDisplay, black, (int(self.head_x),int(self.head_y)), BALANCER_HEAD_RADIUS)
        # pygame.draw.circle(self.gameDisplay, red, (int(self.ball_x),int(self.ball_y)), BALL_RADIUS)
        # pygame.draw.line(self.gameDisplay, black, (self.ball_x-pri, self.ball_y+pro),(self.ball_x, self.ball_y), 5)
        for body_part in self.body_parts:
            # pygame.draw.rect(self.gameDisplay, black, (STEP_SIZE*body_part[0], STEP_SIZE*body_part[1], STEP_SIZE, STEP_SIZE), 100)
            self.gameDisplay.fill(black, (STEP_SIZE*body_part[0], STEP_SIZE*body_part[1], STEP_SIZE, STEP_SIZE))

        self.gameDisplay.fill(red, (self.food[0]*STEP_SIZE, self.food[1]*STEP_SIZE, STEP_SIZE, STEP_SIZE))


    def check_wall(self):
        if self.body_parts[0][0] < 0 or self.body_parts[0][0] > MAX_X:
            self.end = True
            return True
        if self.body_parts[0][1] < 0 or self.body_parts[0][1] > MAX_Y:
            self.end = True
            return True
        return False

    def change_orientation(self, direction):
        if direction == 'left':
            if self.dx != 1:
                self.dx = -1
                self.dy = 0
        if direction == 'right':
            if self.dx != -1:
                self.dx = 1
                self.dy = 0
        if direction == 'up':
            if self.dy != 1:
                self.dy = -1
                self.dx = 0
        if direction == 'down':
            if self.dy != -1:
                self.dy = 1
                self.dx = 0

    def move(self):
        self.body_parts.pop(0)
        last = self.body_parts[-1]
        self.body_parts.append((last[0]+self.dx,last[1]+self.dy))
        new_part = self.body_parts[-1]
        if new_part[0] < 0 or new_part[0] > MAX_X or new_part[1] < 0 or new_part[1] > MAX_Y:
            raise Exception
            return
    def check_food(self):
        head = self.body_parts[-1]
        if head[0] == self.food[0] and head[1] == self.food[1]:
            self.food = (random.randint(0,19),random.randint(0,19))
            self.food_time = time.time()

    def tick(self):
        now = time.time()
        if abs(self.tick_time - now) > TIME_STEP_SIZE:
            self.tick_time = now
            self.move()
            self.check_food()

    def start(self):
        self.end = False
        while not self.end:
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        self.change_orientation('left')
                    if event.key == pygame.K_RIGHT:
                        self.change_orientation('right')
                    if event.key == pygame.K_UP:
                        self.change_orientation('up')
                    if event.key == pygame.K_DOWN:
                        self.change_orientation('down')
                    if event.key == pygame.K_q:
                        self.end = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        pass
                    if event.key == pygame.K_RIGHT:
                        pass
                if event.type == pygame.QUIT:
                    self.end = True
            self.tick()
            self.gameDisplay.fill(white)
            self.draw()
            pygame.display.update()
            self.clock.tick(60)
a = Game()
a.start()
