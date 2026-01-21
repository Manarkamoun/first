#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import time
import random
import math
import matplotlib.pyplot as plt

# =====================
# PARAMÈTRES Q-LEARNING
# =====================
alpha = 0.2
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
episodes = 300
steps_per_episode = 200

# =====================
# TRAJECTOIRE CIBLE
# =====================
line_length = 4.0
amplitude = 0.15
frequency = 2

# =====================
# DISCRÉTISATION ÉTATS
# =====================
num_x_states = 20
num_error_states = 21

x_bins = [i * line_length / num_x_states for i in range(num_x_states)]
error_bins = [-0.2 + i * 0.02 for i in range(num_error_states)]

def get_target_y(x):
    return amplitude * math.sin(2 * math.pi * frequency * x / line_length)

def sense_state(x, y):
    error = y - get_target_y(x)
    x_idx = min(range(num_x_states), key=lambda i: abs(x - x_bins[i]))
    e_idx = min(range(num_error_states), key=lambda i: abs(error - error_bins[i]))
    return x_idx * num_error_states + e_idx

# =====================
# ACTIONS
# =====================
actions = [-0.03, -0.01, 0.0, 0.01, 0.03]

num_states = num_x_states * num_error_states
Q = [[0.0 for _ in actions] for _ in range(num_states)]

# =====================
# RÉCOMPENSE
# =====================
def get_reward(x, y):
    error = y - get_target_y(x)
    return - error ** 2

# =====================
# PYBULLET
# =====================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
robotId = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

# Trajectoire visuelle
for i in range(200):
    x1 = line_length * i / 200
    y1 = get_target_y(x1)
    x2 = line_length * (i + 1) / 200
    y2 = get_target_y(x2)
    p.addUserDebugLine([x1, y1, 0], [x2, y2, 0], [0, 0, 1], 2)

# =====================
# APPRENTISSAGE
# =====================
dx = line_length / steps_per_episode

print("🚀 Apprentissage Q-learning en cours...")
for ep in range(episodes):
    x, y = 0.0, 0.0
    state = sense_state(x, y)

    for step in range(steps_per_episode):
        if random.random() < epsilon:
            a = random.randint(0, len(actions) - 1)
        else:
            a = Q[state].index(max(Q[state]))

        y += actions[a]
        y = max(-0.2, min(0.2, y))
        x += dx

        new_state = sense_state(x, y)
        reward = get_reward(x, y)

        Q[state][a] += alpha * (reward + gamma * max(Q[new_state]) - Q[state][a])
        state = new_state

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if ep % 50 == 0:
        print(f"Episode {ep}/{episodes} | epsilon={epsilon:.3f}")

print("✅ Apprentissage terminé")

# =====================
# SIMULATION FINALE
# =====================
print("🎯 Simulation finale")

x, y = 0.0, 0.0
state = sense_state(x, y)

traj_x, traj_y, target_y = [], [], []

for step in range(steps_per_episode):
    a = Q[state].index(max(Q[state]))

    y += actions[a]
    y = max(-0.2, min(0.2, y))
    x += dx

    state = sense_state(x, y)

    traj_x.append(x)
    traj_y.append(y)
    target_y.append(get_target_y(x))

    print(f"Step {step:3d} | x={x:.2f} y={y:.3f} err={abs(y - target_y[-1]):.4f}")

    p.resetBasePositionAndOrientation(robotId, [x, y, 0.1], [0, 0, 0, 1])
    p.stepSimulation()
    time.sleep(0.01)

# =====================
# GRAPHIQUE
# =====================
plt.figure(figsize=(12, 5))
plt.plot(traj_x, traj_y, label="Robot")
plt.plot(traj_x, target_y, '--', label="Cible", linewidth=2)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Suivi de trajectoire sinusoïdale par Q-learning")
plt.legend()
plt.grid()
plt.show()

# =====================
# GARDER PYBULLET OUVERT
# =====================
while True:
    time.sleep(1)

