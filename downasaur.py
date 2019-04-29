# %%
import numpy as np
from PIL import Image
import cv2  # opencv
import io
import time
import pandas as pd
import numpy as np
from IPython.display import clear_output
from random import randint
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

# keras imports
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from collections import deque
import random
import pickle
from io import BytesIO
import base64
import json


# %%
# path variables
game_url = "chrome://dino"
chrome_driver_path = "./chromedriver"
loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
q_value_file_path = "./objects/q_values.csv"
scores_file_path = "./objects/scores_df.csv"

# scripts
# create id for canvas for faster selection from DOM
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

# get image from canvas
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"


# %%
'''
* Game class: Selenium interfacing between the python and browser
* __init__():  Launch the broswer window using the attributes in chrome_options
* get_crashed() : return true if the agent as crashed on an obstacles. Gets javascript variable from game decribing the state
* get_playing(): true if game in progress, false is crashed or paused
* restart() : sends a signal to browser-javascript to restart the game
* press_up(): sends a single to press up get to the browser
* get_score(): gets current game score from javascript variables.
* pause(): pause the game
* resume(): resume a paused game if not crashed
* end(): close the browser and end the game
'''


class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(
            executable_path=chrome_driver_path, options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get('chrome://dino')
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def get_score(self):
        score_array = self._driver.execute_script(
            "return Runner.instance_.distanceMeter.digits")
        # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()


# %%
class DinoAgent:
    def __init__(self, game):  # takes game as input for taking actions
        self._game = game
        self.jump()  # to start the game, we need to jump once

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


# %%
class Game_sate:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        # display the processed image on screen using openCV, implemented using python coroutine
        self._display = show_img()
        self._display.__next__()  # initiliaze the display coroutine

    def get_state(self, actions):
        # storing actions in a dataframe
        actions_df.loc[len(actions_df)] = actions[1]
        score = self._game.get_score()
        reward = 0.1
        is_over = False  # game over
        if actions[1] == 1:
            self._agent.jump()
        image = grab_screen(self._game._driver)
        self._display.send(image)  # display the image on screen
        if self._agent.is_crashed():
            # log the score when game is over
            scores_df.loc[len(loss_df)] = score
            self._game.restart()
            reward = -1
            is_over = True
        return image, reward, is_over  # return the Experience tuple


# %%
def save_obj(obj, name):
    with open('objects/' + name + '.pkl', 'wb') as f:  # dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def grab_screen(_driver):
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)  # processing image as required
    return image


def process_img(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:300, :500]  # Crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return image


def show_img(graphs=False):
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break


# %%
# Intialize log structures from file if exists else create new
loss_df = pd.read_csv(loss_file_path) if os.path.isfile(
    loss_file_path) else pd.DataFrame(columns=['loss'])
scores_df = pd.read_csv(scores_file_path) if os.path.isfile(
    loss_file_path) else pd.DataFrame(columns=['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(
    actions_file_path) else pd.DataFrame(columns=['actions'])
q_values_df = pd.read_csv(actions_file_path) if os.path.isfile(
    q_value_file_path) else pd.DataFrame(columns=['qvalues'])


# %%
# game parameters
ACTIONS = 2  # possible actions: jump, do nothing
GAMMA = 0.99  # decay rate of past observations original 0.99
OBSERVATION = 100.  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 16  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows, img_cols = 80, 80
img_channels = 4  # We stack 4 frames


# %%
# training variables saved as checkpoints to filesystem to resume training from the same step
def init_cache():
    """initial variable caching, done only once"""
    save_obj(INITIAL_EPSILON, "epsilon")
    t = 0
    save_obj(t, "time")
    D = deque()
    save_obj(D, "D")


# %%
'''
Call only once to init file structure
'''
# init_cache()


# %%
def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4),
                     input_shape=(img_cols, img_rows, img_channels)))  # 80*80*4
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    # create model file if not present
    if not os.path.isfile(loss_file_path):
        model.save_weights('model.h5')
    print("We finish building the model")
    return model


# %%
''' 
main training module
Parameters:
* model => Keras Model to be trained
* game_state => Game State module with access to game environment and dino
* observe => flag to indicate whether the model is to be trained(weight updates), else just play
'''


def trainNetwork(model, game_state, observe=False):
    last_time = time.time()
    # store the previous observations in replay memory
    D = load_obj("D")  # load from file system
    # get the first state by doing nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # 0 => do nothing,
    # 1=> jump

    # get next step after performing the action
    x_t, r_0, terminal = game_state.get_state(do_nothing)

    # stack 4 images to create placeholder input
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*20*40*4

    initial_state = s_t

    if observe:
        OBSERVE = 999999999     # We keep observe, never train
        epsilon = FINAL_EPSILON
        print("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print("Weight load successfully")
    else:                       # We go to training mode
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)

    # resume from the previous time step stored in file system
    t = load_obj("time")
    while (True):  # endless running
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0  # reward at 4
        a_t = np.zeros([ACTIONS])  # action at t

        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:  # parameter to skip frames for actions
            if random.random() <= epsilon:  # randomly explore an action
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:  # predict the output
                # input a stack of 4 images, get the prediction
                q = model.predict(s_t)
                # chosing index with maximum q value
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[action_index] = 1        # o=> do nothing, 1=> jump

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        # helpful for measuring frame rate
        print('fps: {0}'.format(1 / (time.time()-last_time)))
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1
        # append the new image to input stack and remove the first one
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:

            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            # 32, 20, 40, 4
            inputs = np.zeros(
                (BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]    # 4D stack of images
                action_t = minibatch[i][1]   # This is action index
                # reward at state_t due to action_t
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]   # next state
                # wheather the agent died or survided due the action
                terminal = minibatch[i][4]

                inputs[i:i + 1] = state_t

                targets[i] = model.predict(state_t)  # predicted q values
                # predict q values for next step
                Q_sa = model.predict(state_t1)

                if terminal:
                    # if terminated, only equals reward
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
        # reset game to initial frame if terminate
        s_t = initial_state if terminal else s_t1
        t = t + 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            game_state._game.pause()  # pause game while saving to filesystem
            model.save_weights("model.h5", overwrite=True)
            save_obj(D, "D")  # saving episodes
            save_obj(t, "time")  # caching time steps
            # cache epsilon to avoid repeated randomness in actions
            save_obj(epsilon, "epsilon")
            loss_df.to_csv("./objects/loss_df.csv", index=False)
            scores_df.to_csv("./objects/scores_df.csv", index=False)
            actions_df.to_csv("./objects/actions_df.csv", index=False)
            q_values_df.to_csv(q_value_file_path, index=False)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            clear_output()
            game_state._game.resume()
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,             "/ EPSILON", epsilon, "/ ACTION",
              action_index, "/ REWARD", r_t,             "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")


# %%
# main function
def playGame(observe=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_sate(dino, game)
    model = buildmodel()
    try:
        trainNetwork(model, game_state, observe=observe)
    except StopIteration:
        game.end()


# %%
playGame()
