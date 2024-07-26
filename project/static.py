import mujoco
import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

model = mujoco.MjModel.from_xml_path('model.xml')
data = mujoco.MjData(model)

data.qvel[:] = np.zeros_like(data.qvel)
data.qacc[:] = np.zeros_like(data.qacc)

joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in range(model.njnt)]
joint_ranges = [model.jnt_range[joint_id] for joint_id in range(model.njnt)]
num_steps = 12
mujoco.mj_step(model, data)

joint_configs = itertools.product(*[np.linspace(joint_range[0], joint_range[1], num=num_steps) for joint_range in joint_ranges])

results = []

def check_no_collisions():
    mujoco.mj_forward(model, data)
    return data.ncon == 0

for config in joint_configs:
    data.qpos[:] = config
    mujoco.mj_forward(model, data)

    if check_no_collisions():
        mujoco.mj_inverse(model, data)
        torques = data.qfrc_inverse
        results.append(list(config) + list(torques))
    else:
        print(f"Collision detected for configuration: {config}")


with open('results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    header = joint_names + [f'torque_{name}' for name in joint_names]
    writer.writerow(header)
    writer.writerows(results)
data = np.genfromtxt('results.csv', delimiter=',', names=True)
plot_data = {name: data[f'torque_{name}'] for name in joint_names}


plt.figure(figsize=(10, 6))
sns.violinplot(data=list(plot_data.values()))
plt.xticks(range(len(joint_names)), joint_names)
plt.xlabel('Joint Name')
plt.ylabel('Torque')
plt.title('Torque Distribution Across Joints')
plt.show()
