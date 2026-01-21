import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2

# ===============================
# INIT SIMULATION
# ===============================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)

plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("kuka_iiwa/model.urdf",[0,0,0],useFixedBase=True)

cube = p.loadURDF("cube_small.urdf",[0.5,0,0.05])
p.changeVisualShape(cube,-1,rgbaColor=[0,0,1,1])

# ===============================
# CAMERA
# ===============================
camTarget=[0.4,0,0]
camPos=[0.4,1,0.6]
view = p.computeViewMatrix(camPos,camTarget,[0,0,1])
proj = p.computeProjectionMatrixFOV(60,1,0.1,3)

# ===============================
# BLUE DETECTION
# ===============================
def detect_blue():
    w,h,rgba,_,_ = p.getCameraImage(320,320,view,proj)
    img = np.reshape(rgba,(h,w,4))[:,:,:3].astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower=np.array([100,150,50])
    upper=np.array([140,255,255])
    mask=cv2.inRange(hsv,lower,upper)

    M=cv2.moments(mask)
    if M["m00"]==0:
        return None

    cx=int(M["m10"]/M["m00"])
    x=0.5
    y=(cx-160)/320
    z=0.05
    return [x,y,z]

# ===============================
# FSM PICK & PLACE
# ===============================
state=0
cid=None

print("Robot prêt")

while True:
    pos=detect_blue()

    if pos is None:
        p.stepSimulation()
        time.sleep(1/240)
        continue

    above=[pos[0],pos[1],pos[2]+0.25]
    touch=[pos[0],pos[1],pos[2]+0.02]
    place=[0.3,-0.4,0.15]

    ee=p.getLinkState(robot,6)[0]

    # 1. Approche
    if state==0:
        joints=p.calculateInverseKinematics(robot,6,above)
        for i in range(7):
            p.setJointMotorControl2(robot,i,p.POSITION_CONTROL,joints[i])
        if np.linalg.norm(np.array(ee)-np.array(above))<0.03:
            state=1

    # 2. Descente
    elif state==1:
        joints=p.calculateInverseKinematics(robot,6,touch)
        for i in range(7):
            p.setJointMotorControl2(robot,i,p.POSITION_CONTROL,joints[i])
        if np.linalg.norm(np.array(ee)-np.array(touch))<0.01:
            state=2

    # 3. Saisie
    elif state==2:
        cid=p.createConstraint(robot,6,cube,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])
        print("Cube attrapé")
        state=3

    # 4. Transport
    elif state==3:
        joints=p.calculateInverseKinematics(robot,6,place)
        for i in range(7):
            p.setJointMotorControl2(robot,i,p.POSITION_CONTROL,joints[i])
        if np.linalg.norm(np.array(ee)-np.array(place))<0.05:
            state=4

    # 5. Dépose
    elif state==4:
        p.removeConstraint(cid)
        print("Cube déposé")
        state=5   # passer à état final pour garder simulation ouverte

    # 6. Simulation ouverte après dépôt
    elif state==5:
        # juste garder la fenêtre et la physique active
        p.stepSimulation()
        time.sleep(1/240)
        continue

    p.stepSimulation()
    time.sleep(1/240)

