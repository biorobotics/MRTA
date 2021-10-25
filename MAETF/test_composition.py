import numpy as np
import matplotlib.pyplot as plt

from ergodicity import ErgCalculator
from agent import xyAgent
from team import optimize_team_erg
from scipy.special import softmax

erg_calc = ErgCalculator()


#################################################
###############   calc total ergo   #############
#################################################

# agent parameters
# similar to toy example, we assume 3 kinds of agents and 4 types of terrain
# lets just assume that for env type, 0: lake, 1: mountain, 2: plain, 3: city
env_type_color = ['navy', 'saddlebrown', 'forestgreen', 'ghostwhite']
n_type_agents = 3
n_num_grids = 4
agent_num = [3, 8, 5]

# Let's just assume they are drone, car and ship
agent_color = ['red', 'lime', 'lightblue']

nobs = [30, 10, 10]
weight = nobs
control_penalty = np.array(
    [
        [0.1, 0.1, 0.1, 1], # drone is good on all terrain
        [30, 3, 1, 1], # car: bad in lake, kinda bad in mountain
        [1, 30, 30, 30], # ship: bad in everywhere except for lake

    ]
)
# turn continuous alloc into discrete assignment
def get_integer(alloc):
    alloc = alloc.T
    int_alloc = np.zeros_like(alloc)
    for i in range(n_type_agents):
        remaining = agent_num[i]
        for j in range(n_num_grids):
            if j == n_num_grids - 1:
                int_alloc[j][i] = remaining
            else:
                cur_num = round(alloc[j][i]*agent_num[i])
                cur_num = np.min([remaining, cur_num])
                remaining -= cur_num
                int_alloc[j][i] = cur_num
    return int_alloc.T


def get_agent(terrain, agent_id):
    agent = xyAgent(agent_color[agent_id], erg_calc,
                    control_penalty=control_penalty[agent_id][terrain],
                    nobs=nobs[agent_id], weight=weight[agent_id])
    return agent



# the part about forming compositions
n_rand = 1
min_erg = np.inf
env = [0, 1, 2, 3]
color = []
for _ in range(n_rand):
    team_fouriers = []
    dist_fouriers = []

    assignment = np.random.uniform(-1, 1, [n_type_agents, n_num_grids])
    assignment = softmax(assignment, axis=1)
    int_assignment = get_integer(assignment)
    # int_assignment = np.array([[1., 1., 1., 0.],
    #                   [0., 2., 3., 3.],
    #                   [5., 0., 0., 0.]], dtype=int)
    print(int_assignment)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    axes = [ax1, ax2, ax3, ax4]
    for grid_id in range(n_num_grids):
        team = []
        for agent_id in range(n_type_agents):
            for _ in range(int(int_assignment[agent_id, grid_id])):
                team.append(get_agent(terrain=env[grid_id], agent_id=agent_id))
        if len(team) == 0:
            continue

        distribution = np.zeros((50, 50))
        distribution[int(25 * grid_id % 2):int(25 * grid_id % 2+25),
                     int(25 * grid_id // 2):int(25 * grid_id // 2+25)] = 1
        # distribution = np.ones((50, 50))
        distribution = distribution / np.sum(distribution)

        # start_states = [
        #     np.array([.6, .8]),
        # ]*len(team)
        erg, team_fourier, dist_fourier = optimize_team_erg(team, distribution, erg_calc) #, start_states=start_states)

        team_fouriers.append(team_fourier)
        dist_fouriers.append(dist_fourier)
        # append these results to a list for final computation


        axes[grid_id].set_facecolor(env_type_color[env[grid_id]])

        # axes[grid_id].imshow(distribution.T, origin='lower', extent=(0, 1, 0, 1), cmap='gray')
        for agent in team:
            agent.plot_self(axes[grid_id])
        print(f'erg team {grid_id}', erg)

        axes[grid_id].set_xlim([0, 1])
        axes[grid_id].set_ylim([0, 1])

    #get total erg
    dist_weights = [25, 25, 25, 25]
    erg = erg_calc.calc_erg_multi_team(
        team_fouriers, dist_fouriers, dist_weights
    )
    print('total ergodicity', erg)
    plt.show()