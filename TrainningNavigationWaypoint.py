from pyrep import VRep
#import time   # For time.sleep(0.5)

import numpy as np
import random
import csv
#from nn import neural_net, LossHistory
#import os.path
import timeit

import math

# ----------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
#from keras.layers.recurrent import LSTM
from keras.callbacks import Callback

# from keras import losses
# model.compile(loss=losses.mean_squared_error, optimizer='sgd')
# ----------------------------------------

NUM_INPUT = 18#16+2
GAMMA = 0.9   # Forgetting.
d=0.415       # Meters
v_rob=0.8
num_goal_pos=6

# ------------------------------------
#               Functions and Classes
# ------------------------------------

class Robot:

    def __init__(self, api: VRep):
        self._api = api
        self._left_motor = api.joint.with_velocity_control("Pioneer_p3dx_leftMotor")
        self._right_motor = api.joint.with_velocity_control("Pioneer_p3dx_rightMotor")
        self._sensor1 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor1")
        self._sensor2 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor2")
        self._sensor3 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor3")
        self._sensor4 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor4")
        self._sensor5 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor5")
        self._sensor6 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor6")
        self._sensor7 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor7")
        self._sensor8 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor8")
        self._sensor9 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor9")
        self._sensor10 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor10")
        self._sensor11 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor11")
        self._sensor12 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor12")
        self._sensor13 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor13")
        self._sensor14 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor14")
        self._sensor15 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor15")
        self._sensor16 = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor16")
        self._robot_position = api.sensor.position("Pioneer_p3dx_visible")
        self._goal_position1 = api.sensor.position("GoalPosition1")
        self._goal_position2 = api.sensor.position("GoalPosition2")
        self._goal_position3 = api.sensor.position("GoalPosition3")
        self._goal_position4 = api.sensor.position("GoalPosition4")
        self._goal_position5 = api.sensor.position("GoalPosition5")
        self._goal_position6 = api.sensor.position("GoalPosition6")
        self._robot_orientation = api.sensor.position("Pioneer_p3dx_visible")
        
    def aply_angular_velocity(self, omega):
        v_right=v_rob+((d/2)*omega)
        v_left=v_rob-((d/2)*omega)
        self._left_motor.set_target_velocity(v_left)
        self._right_motor.set_target_velocity(v_right)
        
    def posicao_robo(self):
        return self._robot_position.get_position()
    
    def goal_position1(self):
        return self._goal_position1.get_position()
    
    def goal_position2(self):
        return self._goal_position2.get_position()
    
    def goal_position3(self):
        return self._goal_position3.get_position()
    
    def goal_position4(self):
        return self._goal_position4.get_position()
    
    def goal_position5(self):
        return self._goal_position5.get_position()
    
    def goal_position6(self):
        return self._goal_position6.get_position()
    
    def orientacao_robo(self):
        return self._robot_orientation.get_orientation()
        
    def sensor1_dist(self):
        return self._sensor1.read()[1].distance()
    
    def sensor2_dist(self):
        return self._sensor2.read()[1].distance()
    
    def sensor3_dist(self):
        return self._sensor3.read()[1].distance()
    
    def sensor4_dist(self):
        return self._sensor4.read()[1].distance()
    
    def sensor5_dist(self):
        return self._sensor5.read()[1].distance()
    
    def sensor6_dist(self):
        return self._sensor6.read()[1].distance()
    
    def sensor7_dist(self):
        return self._sensor7.read()[1].distance()
    
    def sensor8_dist(self):
        return self._sensor8.read()[1].distance()
    
    def sensor9_dist(self):
        return self._sensor9.read()[1].distance()
    
    def sensor10_dist(self):
        return self._sensor10.read()[1].distance()
    
    def sensor11_dist(self):
        return self._sensor11.read()[1].distance()
    
    def sensor12_dist(self):
        return self._sensor12.read()[1].distance()
    
    def sensor13_dist(self):
        return self._sensor13.read()[1].distance()
    
    def sensor14_dist(self):
        return self._sensor14.read()[1].distance()
    
    def sensor15_dist(self):
        return self._sensor15.read()[1].distance()
    
    def sensor16_dist(self):
        return self._sensor16.read()[1].distance()

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def NeuralNetwork(num_sensors, params, load=''):
    model = Sequential()

    # First layer.
    model.add(Dense(
        params[0], init='lecun_uniform', input_shape=(num_sensors,)
    ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second layer.
    model.add(Dense(params[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # Third layer.
    model.add(Dense(params[2], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(5, init='lecun_uniform'))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    if load:
        model.load_weights(load)

    return model

def log_results(filename, data_collect, loss_log):
    # Save the results to a file so we can graph it later.
    with open('results/sonar-frames/learn_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/sonar-frames/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)

def process_minibatch(minibatch, MainNetwork):
    # by Microos, improve this batch processing function 
    #   and gain 50~60x faster speed (tested on GTX 1080)
    #   significantly increase the training FPS

    mb_len = len(minibatch)

    old_states = np.zeros(shape=(mb_len, 18))
    actions = np.zeros(shape=(mb_len,))
    rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, 18))
    
    for i, m in enumerate(minibatch):
        old_state_m, action_m, reward_m, new_state_m = m
        old_states[i, :] = old_state_m[...]
        actions[i] = action_m
        rewards[i] = reward_m
        new_states[i, :] = new_state_m[...]

    old_qvals = MainNetwork.predict(old_states, batch_size=mb_len)
    new_qvals = MainNetwork.predict(new_states, batch_size=mb_len)

    maxQs = np.max(new_qvals, axis=1)
    y = old_qvals
    non_term_inds = np.where(rewards != -500)[0]
    term_inds = np.where(rewards == -500)[0]

    y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (GAMMA * maxQs[non_term_inds])
    y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]

    X_train = old_states
    y_train = y
    print('X_train', X_train)
    print('y_train', y_train)
    return X_train, y_train


def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])

def DistanceAngle2Goal(select_goal, rob_pos, gamma_euler_rad):
    if select_goal == 1:
        goal_pos=r.goal_position1()
    elif select_goal == 2:
        goal_pos=r.goal_position2()
    elif select_goal == 3:
        goal_pos=r.goal_position3()
    elif select_goal == 4:
        goal_pos=r.goal_position4()
    elif select_goal == 5:
        goal_pos=r.goal_position5()
    elif select_goal == 6:
        goal_pos=r.goal_position6()
           
    gamma_euler=(gamma_euler_rad*180)/(np.pi)
    
    x_rob, y_rob = rob_pos.get_x(), rob_pos.get_y()
    x_goal, y_goal = goal_pos.get_x(), goal_pos.get_y()

    dist_to_goal=math.sqrt((x_rob-x_goal)**2+(y_rob-y_goal)**2)
    
    psi=np.arctan((y_goal-y_rob)/(x_goal-x_rob))*(180/np.pi) # Angle from reference frame to vector from robot position to reference position 
    
    if np.abs(psi-gamma_euler)<180:#np.pi
        theta=psi-gamma_euler  # Angle from robot to reference point (waypoint)
    else:
        if psi<0:
            psi=360-np.abs(psi)
        if gamma_euler<0:
            gamma_euler=360-np.abs(gamma_euler)
        theta=psi-gamma_euler
    return dist_to_goal, theta

def UltrassonicSensors():
    ultrassonicdata=np.zeros(16)
    ultrassonicdata[0]=r.sensor1_dist()
    ultrassonicdata[1]=r.sensor2_dist()
    ultrassonicdata[2]=r.sensor3_dist()
    ultrassonicdata[3]=r.sensor4_dist()
    ultrassonicdata[4]=r.sensor5_dist()
    ultrassonicdata[5]=r.sensor6_dist()
    ultrassonicdata[6]=r.sensor7_dist()
    ultrassonicdata[7]=r.sensor8_dist()
    ultrassonicdata[8]=r.sensor9_dist()
    ultrassonicdata[9]=r.sensor10_dist()
    ultrassonicdata[10]=r.sensor11_dist()
    ultrassonicdata[11]=r.sensor12_dist()
    ultrassonicdata[12]=r.sensor13_dist()
    ultrassonicdata[13]=r.sensor14_dist()
    ultrassonicdata[14]=r.sensor15_dist()
    ultrassonicdata[15]=r.sensor16_dist()
    for i in range(16):
        if ultrassonicdata[i]<0.03 or ultrassonicdata[i]>10:
            ultrassonicdata[i]=1.0
    return ultrassonicdata

def Environment(action, dist_to_goal_back, sum_dist_to_obs_back, select_goal):
    if action == 0:      # Turn right hard
        omega=-1.0
    elif action == 1:    # Turn right soft
        omega=-0.3
    elif action == 2:    # Keep foward
        omega=0.0
    elif action == 3:    # Turn left soft
        omega=0.3
    elif action == 4:    # Turn left hard
        omega=1.0
    
    r.aply_angular_velocity(omega)
    
    readings=np.zeros(NUM_INPUT)
    
    dist_to_obs=UltrassonicSensors()
    
    rob_pos=r.posicao_robo()
    EulerAngles=r.orientacao_robo()
    gamma_euler_rad=EulerAngles.get_gamma()
    
    # Get distance and angle to gol
    
    dist_to_goal, theta = DistanceAngle2Goal(select_goal, rob_pos, gamma_euler_rad)
    
    for i in range(15):
        readings[i]=dist_to_obs[i]
    readings[16]=dist_to_goal
    readings[17]=theta
    
    state=np.array([readings])
    sum_dist_to_obs=np.sum(dist_to_obs)
    
    diff_dist_to_goal=dist_to_goal-dist_to_goal_back
    diff_sum_dist_to_obs=sum_dist_to_obs-sum_dist_to_obs_back
    k1, k2, k3 = 20.0, -100.0, -1
    
    if any(i<=0.2 for i in dist_to_obs):
       reward = -500
       api.simulation.restart() # Restart
       if select_goal == 1:
           dist_to_goal_back=dist_to_goal_back_init2
       elif select_goal == 2:
           dist_to_goal_back=dist_to_goal_back_init3
       elif select_goal == 3:
           dist_to_goal_back=dist_to_goal_back_init4
       elif select_goal == 4:
           dist_to_goal_back=dist_to_goal_back_init5
       elif select_goal == 5:
           dist_to_goal_back=dist_to_goal_back_init6
       elif select_goal == 6:
           dist_to_goal_back=dist_to_goal_back_init1
       sum_dist_to_obs_back=np.sum(dist_to_obs_init)
    else:
        if dist_to_goal<0.5 and np.abs(theta)<5:
            reward=50.0
        else:
            reward1=k1*diff_sum_dist_to_obs
            reward2=k2*diff_dist_to_goal
            reward3=k3*np.abs(theta)*(16/180)
            reward=reward1+reward2+reward3
        dist_to_goal_back=dist_to_goal
        sum_dist_to_obs_back=sum_dist_to_obs
    
    return reward, state, dist_to_goal_back, sum_dist_to_obs_back

# ------------------------------------
#               MAIN PROGRAM
# ------------------------------------

# ----------------------------------
# Initialize the neural network
# ----------------------------------

nn_param = [200, 200, 200] # 128, 128
params = {
    "batchSize": 100,
    "buffer": 50000,
    "nn": nn_param
}
MainNetwork = NeuralNetwork(NUM_INPUT, nn_param)

TargetNetwork=MainNetwork
# ----------------------------------
# Remote API connection
# ----------------------------------

api = VRep.connect("127.0.0.1", 19997)
r = Robot(api)                   

# ----------------------------------
# 
# ----------------------------------

x_rob_init, y_rob_init = -1.7000, -1.9250 # True
#x_rob_init, y_rob_init = -1.685, -1.884  # Alternative
x_goal_init=[1.625, -0.425, 1.125, -1.925, -1.925, -0.275]
y_goal_init=[1.7, 0.85, -1.75, -1.225, 1.7, -0.75]
dist_to_goal_back_init1=math.sqrt((x_rob_init-x_goal_init[0])**2+(y_rob_init-y_goal_init[0])**2)
dist_to_goal_back_init2=math.sqrt((x_rob_init-x_goal_init[1])**2+(y_rob_init-y_goal_init[1])**2)
dist_to_goal_back_init3=math.sqrt((x_rob_init-x_goal_init[2])**2+(y_rob_init-y_goal_init[2])**2)
dist_to_goal_back_init4=math.sqrt((x_rob_init-x_goal_init[3])**2+(y_rob_init-y_goal_init[3])**2)
dist_to_goal_back_init5=math.sqrt((x_rob_init-x_goal_init[4])**2+(y_rob_init-y_goal_init[4])**2)
dist_to_goal_back_init6=math.sqrt((x_rob_init-x_goal_init[5])**2+(y_rob_init-y_goal_init[5])**2)

dist_to_obs_init=9.55

# ----------------------------------
# Learning by Deep Reinforcement Learning
# ----------------------------------

filename = params_to_filename(params)

observe = 1000  # Number of frames to observe before training.
epsilon = 1
train_frames = 400000  # Number of frames to play.
batchSize = params['batchSize']
buffer = params['buffer']

t = 0
episodio = 0
elem_episodio = 0
max_elem_episodio = 0
data_collect = []
replay = []           # stores tuples of (S, A, R, S').

loss_log = []
    
#    # Start simulation
api.simulation.start()
    
select_goal=1
    
_, state, dist_to_goal_back, sum_dist_to_obs_back = Environment(2, dist_to_goal_back_init1, dist_to_obs_init, select_goal)

start_time = timeit.default_timer()

# Run the episodes/frames.
while t < train_frames:
    t += 1
    elem_episodio += 1

    # Select an action: e-greedy policy
    if random.random() < epsilon or t < observe:
        action = np.random.randint(0, 5)  # random
    else:
        # Observe the Q-value for each action
        qval = MainNetwork.predict(state, batch_size=1)
        action = (np.argmax(qval))  # melhor

    # Take an action, observe the new state and get a reward.
    reward, new_state, dist_to_goal_back, sum_dist_to_obs_back = Environment(action, dist_to_goal_back, sum_dist_to_obs_back, select_goal)

    # Experience replay storage.
    replay.append((state, action, reward, new_state))

    if t > observe:
        if len(replay) > buffer:
            replay.pop(0)

        # Sample randomly replay memory (experience replay memory)
        minibatch = random.sample(replay, batchSize)

        # Get training values
        X_train, y_train = process_minibatch(minibatch, MainNetwork)

        # Mini-batch training
        history = LossHistory()
        MainNetwork.fit(
                X_train, y_train, batch_size=batchSize,
                epochs=1, verbose=0, callbacks=[history]
        )
        loss_log.append(history.losses)

    # Update the states
    state = new_state

    # Decrease epsilon over time:
    if epsilon > 0.1 and t > observe:
        epsilon -= (1.0/train_frames)

    # If the robot collided, then::
    if reward == -500:
        data_collect.append([t, episodio, elem_episodio, max_elem_episodio])
        episodio += 1
        
        # Select the goal position for each episode by consecutive alternation
        if select_goal==num_goal_pos:
            select_goal=1
        else:
            select_goal+=1

        # Update max.
        if elem_episodio > max_elem_episodio:
            max_elem_episodio = elem_episodio

        # Time it.
        tot_time = timeit.default_timer() - start_time
        fps = elem_episodio / tot_time

        print(" Episode: %d of %d elements in sample: %d - max elem_episodio: % d , epsilon: %d and fps:%d" %
              (episodio, elem_episodio, t, max_elem_episodio, epsilon, fps))

        # Reset.
        elem_episodio = 0
        start_time = timeit.default_timer()

        # Save the model every 25,000 frames.
    if t % 50000 == 0:
        MainNetwork.save_weights('saved-models/' + filename + '-' +
                                 str(t) + '.h5',
                                 overwrite=True)
        print("Saving model %s - %d" % (filename, t))
    # Log results after we're done all frames.
    log_results(filename, data_collect, loss_log)

VRep.close_connection()