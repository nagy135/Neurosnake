import tensorflow as tf
import copy
import time
import os
import pygame
import math
import random
import numpy as np

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
TIME_STEP_SIZE = 0.4

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

        ## Neural Network variables
        self.x = tf.placeholder('float',[None, 400])
        self.y = tf.placeholder('float')
        self.map = tf.placeholder('float',[None, 400])
        self.classes = 4
        self.keep_rate = 0.8
        self.keep_prob = tf.placeholder(tf.float32)
        self.training = False
        ##

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def maxpool2d(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def fully_connected(self, x):
        return tf.contrib.layers.fully_connected(x)

    def model(self, inner_x):
        n_nodes_hl1 = 100
        n_nodes_hl2 = 100
        n_nodes_hl3 = 100

        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([400, n_nodes_hl1]), name="layer1_weights"),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl1]), name="layer1_biases")}

        hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]), name="layer2_weights"),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl2]), name="layer2_biases")}

        hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]), name="layer3_weights"),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl3]), name="layer3_biases")}

        output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, self.classes]), name="out_weights"),
                        'biases':tf.Variable(tf.random_normal([self.classes]), name="out_biases")}


        l1 = tf.add(tf.matmul(inner_x, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

        return output

    def cost_function(self, predict):
        print('=======================================================================')
        print(predict)
        #[1,0,0,0] = Up
        #[0,1,0,0] = Right
        #[0,0,1,0] = Down
        #[0,0,0,1] = Left
        return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, self.y) )

    def outside_cost(self):
        #TODO create [0, 0, 0, 1] and return it
        return np.array([0, 0, 0, 1])

    def train_neural_network(self, frame_x):
        prediction = self.model(self.x)
        
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            frame_x = frame_x.reshape((1,400))
            predict = sess.run([prediction], feed_dict={self.x: frame_x})

            ## PERFORM MOVE
            new_orientation = np.argmax(predict)
            if new_orientation == 0:
                self.change_orientation('up')
            elif new_orientation == 1:
                self.change_orientation('right')
            elif new_orientation == 2:
                self.change_orientation('down')
            elif new_orientation == 3:
                self.change_orientation('left')
            self.move()

            print('==================================')
            variables = tf.trainable_variables()
            for var in variables:
                print(var)
            print('==================================')

            optimizer = tf.train.AdamOptimizer().minimize(cost)
            _, c = sess.run([optimizer, cost], feed_dict={prediction: predict})
            print("Loss : {}".format(str(c)))
            # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            # print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    def train(self):
        self.training = not self.training
        if self.training:
            print("Training begins")
        else:
            print("Training ends")

    def train_tick(self):
        frame_x = self.create_map()
        self.train_neural_network(frame_x)

    def create_map(self):
        arr = np.zeros((MAX_Y+1, MAX_X+1))
        for bodypart in self.body_parts:
            arr[bodypart[1]][bodypart[0]] = 1
        arr[self.body_parts[-1][1]][self.body_parts[-1][0]] = 2
        return arr

    def draw(self):
        for body_part in self.body_parts:
            self.gameDisplay.fill(black, (STEP_SIZE*body_part[0], STEP_SIZE*body_part[1], STEP_SIZE, STEP_SIZE))
        self.gameDisplay.fill(red, (self.food[0]*STEP_SIZE, self.food[1]*STEP_SIZE, STEP_SIZE, STEP_SIZE))

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
            # self.move()
            # self.check_food()
            if self.training:
                self.train_tick()

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
                    if event.key == pygame.K_m:
                        self.train()
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
