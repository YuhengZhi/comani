from matplotlib import pyplot as plt

smooth = 10

train_log = open("training_log", 'r')
lines = train_log.readlines()
train_log.close()
reward_history = []
for i in range(len(lines)):
    if(len(lines[i]) < 2):
        continue
    if(i%2 == 1):
        reward_history.append(float(lines[i].split(' ')[1][:-1]))

sum = 0
for i in range(smooth):
    sum += reward_history[i]
smoothed = []
for i in range(len(reward_history) - smooth):
    smoothed.append(sum / smooth)
    sum -= reward_history[i]
    sum += reward_history[i + smooth]
x_axis = [i+1 for i in range(len(smoothed))]
fig, ax = plt.subplots(1, 1, figsize=(12,10))
ax.set_ylim([0, 20])
ax.plot(x_axis, smoothed)
plt.title("Smoothed episode total reward " + str(smooth))
plt.savefig("reward_history_" + str(smooth) + ".png")

# Code for displaying the raw reward. Usually not very clear due to
# high number of episodes
# fig, ax = plt.subplots(1,1, figsize=(10,6))
# ax.set_ylim([-8,10])
# ax.scatter([i+1 for i in range(len(reward_history))], reward_history, s=1)
# plt.title("Total reward each episode")
# plt.savefig("util/reward_history.png")
