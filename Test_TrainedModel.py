"""
Once a model is learned, use this to play it.
"""

from pyrep import VRep
#import time
import numpy as np
from nn import neural_net

import math

NUM_INPUT = 18#16+2
d=0.415 # Metros
v_rob=0.8
select_goal=4

# ------------------------------------
#               FUNÇÕES
# ------------------------------------

class PioneerP3DX:

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

# =================================================================
    
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
        
#    print('select goal', select_goal)
    
    gamma_euler=(gamma_euler_rad*180)/(np.pi)
    
    x_rob, y_rob = rob_pos.get_x(), rob_pos.get_y()
    x_goal, y_goal = goal_pos.get_x(), goal_pos.get_y()

    dist_to_goal=math.sqrt((x_rob-x_goal)**2+(y_rob-y_goal)**2)
    
    psi=np.arctan((y_goal-y_rob)/(x_goal-x_rob))*(180/np.pi) # Angle from reference frame to vector from robot position to reference position 
    
    if np.abs(psi-gamma_euler)<180:#np.pi
        theta=psi-gamma_euler  # Angle from robto to reference point (waypoint)
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
# =================================================================    

def ActionState(action, select_goal):
    if action == 0:      # Turn left hard
        omega=-1.0
    elif action == 1:    # Turn left soft
        omega=-0.3
    elif action == 2:    # Keep foward
        omega=0.0
    elif action == 3:    # Turn right soft
        omega=0.3
    elif action == 4:    # Turn right hard
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
    
#    print('Estado e Ação')   
#    print(state, action)
    
    return state

def simular(model):

    api.simulation.start()

    state = ActionState(2, select_goal)

    # Move.
    while True:
        
#        stated = ActionState(1)            
        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))
#        print('State and Action:', state, action)
        
        new_state = ActionState(action, select_goal)
        
        state=new_state
        
    api.simulation.stop()

#api.simulation.start()
    
# ------------------------------------
#               PROGRAMA PRINCIPAL
# ------------------------------------

# ----------------------------------
# Inicializa a rede neural
# ----------------------------------
    
saved_model = 'saved-models/200-200-100-50000-400000.h5' #
model = neural_net(NUM_INPUT, [200, 200, 200], saved_model)

# ----------------------------------
# Conecta a Remote API do python no VREP
# ----------------------------------

api = VRep.connect("127.0.0.1", 19997)
r = PioneerP3DX(api)

# ----------------------------------
# Executa a aprendizagem usando Deep Reinforcement Learning
# ----------------------------------

simular(model)
