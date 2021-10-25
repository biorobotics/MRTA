import copy
import numpy as np
import matplotlib.pyplot as plt

from ergodicity import ErgCalculator
from agent import xyAgent
from team import optimize_team_erg

erg_calc = ErgCalculator(10)
terrain = np.zeros((50, 50), dtype=np.int32)
terrain[0:25, 0:25] = 1
terrain[0:25, 25:] = 2
terrain[25:, 25:] = 3

motion_scale = 0.3
terrain_traversability1 = np.array([0, .02, .04, .05])*motion_scale
agent1 = xyAgent('lime', erg_calc, 20, terrain_traversability1, weight=20)

terrain_traversability2 = np.array([.05, 0, 0, 0])*motion_scale
agent2 = xyAgent('lightblue', erg_calc, 20, terrain_traversability2, weight=20)

terrain_traversability3 = np.array([.05, .05, .05, .03])*motion_scale
agent3 = xyAgent('red', erg_calc, 20, terrain_traversability3, weight=20)


# agent3 = xyAgent('lightblue', erg_calc, dist_no_penalty=.1, nobs=10, weight=10)
# agent4 = xyAgent('magenta', erg_calc, dist_no_penalty=.03, nobs=30, weight=30)


# distribution1 = np.ones((50, 50))
distribution1 = np.zeros((50, 50))
# distribution1[25:, :25] = 1
distribution1[25:, 25:] = 1
distribution1 = distribution1 / np.sum(distribution1)

import time
start_time = time.time()

team1 = (
	[copy.deepcopy(agent1) for i in range(20)] +
	[copy.deepcopy(agent2) for i in range(0)] +
	[copy.deepcopy(agent3) for i in range(20)]
)
start_states = None #[[.2, .7] for i in range(6)] + [[.7, .2] for i in range(2)]
# team1 = [copy.deepcopy(agent2) for i in range(5)]
# start_states = [[np.random.random(), .25] for i in range(5)]

erg1, team_fourier1, dist_fourier1 = optimize_team_erg(team1, 4, terrain, distribution1,
													   erg_calc, start_states=start_states,
													   n_opt=300)

print("--- %s seconds ---" % (time.time() - start_time))

from matplotlib import colors
env_type_color = ['navy', 'saddlebrown', 'forestgreen', 'ghostwhite']
cmap = colors.ListedColormap(env_type_color)

fig, ax = plt.subplots()
im = ax.imshow(terrain.T, origin='lower', extent=(0, 1, 0, 1), cmap=cmap)
for agent in team1:
	agent.plot_self(ax)
print('erg team 1', erg1)
print(np.log(erg1))
save_location = False
if save_location:
	loc_list = []
	for agent in team1:
		loc_list.append([agent.color, agent.cur_traj])
	np.save("loc_t", np.array(loc_list))

# distribution2 = np.zeros((50, 50))
# distribution2[0:25, 0:25] = 1
# distribution2 = distribution2 / np.sum(distribution2)
# team2 = [agent3, agent4]

# erg2, team_fourier2, dist_fourier2 = optimize_team_erg(team2, distribution2, erg_calc)
# ax2.imshow(distribution2.T, origin='lower', extent=(0, 1, 0, 1), cmap='gray')
# for agent in team2:
# 	agent.plot_self(ax2)
# print('erg team 2', erg2)

# dist_weights = [50, 50]
# erg = erg_calc.calc_erg_multi_team(
# 	[team_fourier1, team_fourier2], [dist_fourier1, dist_fourier2], dist_weights
# )
# print('total ergodicity', erg)

# plt.colorbar(im)
plt.show()



