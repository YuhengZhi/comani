from matplotlib import pyplot as plt

train_log = open("train_output", 'r')
lines = train_log.readlines()
train_log.close()
reward_history = []
for i in range(len(lines)):
    if(len(lines[i]) < 2):
        continue
    if(i%2 == 1):
        reward_history.append(float(lines[i].split(' ')[1][:-1]))
fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.set_ylim([-8,10])
ax.scatter([i+1 for i in range(len(reward_history))], reward_history, s=1)
plt.title("Total reward each episode")
plt.savefig("util/reward_history.png")