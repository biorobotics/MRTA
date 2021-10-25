import numpy as np
import matplotlib.pyplot as plt

from .ergodicity import ErgCalculator
from .agent import xyAgent
from .team import optimize_team_erg
from scipy.special import softmax
import copy


class MultiAgentEnv:
    def __init__(self, n_num_grids=4, n_num_agents=3, n_env_types=4, agent_num=[20, 20, 20],
                 nopt=300, erg_num=10):
        self.n_num_grids = n_num_grids
        self.n_types_agents = n_num_agents
        self.n_types_terrain = n_env_types
        self.agent_num = agent_num

        self.motion_scale = 0.3
        self.nopt = nopt
        self.erg_num = erg_num  # could be 15
        # the part where we calculate the ergodicity
        self.erg_calc = ErgCalculator(self.erg_num)

        if n_env_types == 4:
            # cars
            terrain_traversability1 = np.array([0, .02, .04, .05])*self.motion_scale
            self.agent1 = xyAgent('lime', self.erg_calc, 20, terrain_traversability1, weight=20)
            # ships
            terrain_traversability2 = np.array([.05, 0, 0, .03])*self.motion_scale
            self.agent2 = xyAgent('lightblue', self.erg_calc, 20, terrain_traversability2, weight=20)
            # drones
            terrain_traversability3 = np.array([.05, .05, .05, .03])*self.motion_scale
            self.agent3 = xyAgent('red', self.erg_calc, 20, terrain_traversability3, weight=20)
        elif n_env_types == 6:
            # add lake, mountain, plain, city, rain forrest, desert
            # cars
            terrain_traversability1 = np.array([0, .02, .04, .05, .03, .05]) * self.motion_scale
            self.agent1 = xyAgent('lime', self.erg_calc, 20, terrain_traversability1, weight=20)
            # ships
            terrain_traversability2 = np.array([.05, 0, 0, .03, .05, 0]) * self.motion_scale
            self.agent2 = xyAgent('lightblue', self.erg_calc, 20, terrain_traversability2, weight=20)
            # drones
            terrain_traversability3 = np.array([.05, .05, .05, .03, 0, .05]) * self.motion_scale
            self.agent3 = xyAgent('red', self.erg_calc, 20, terrain_traversability3, weight=20)
        else:
            exit("error in simulator.py")
    # turn continuous alloc into discrete assignment
    def get_integer(self, alloc):
        alloc = alloc.T
        int_alloc = np.zeros_like(alloc, dtype=int)
        for i in range(self.n_types_agents):
            remaining = self.agent_num[i]
            for j in range(self.n_num_grids):
                if j == self.n_num_grids - 1:
                    int_alloc[j][i] = remaining
                else:
                    cur_num = round(alloc[j][i] * self.agent_num[i])
                    cur_num = np.min([remaining, cur_num])
                    remaining -= cur_num
                    int_alloc[j][i] = cur_num
        return int_alloc.T

    def get_integer_reward(self, int_alloc, terrain, info=None, plot=False, print_info=False, save_location=False):
        if print_info:
            print(int_alloc)
        reward = 0
        teams = []
        reward_list = []
        for i in range(self.n_num_grids):
            if info is None:
                distribution = np.zeros((50, 50))
                distribution[int(25 * (i % 2)):int(25 * (i % 2) + 25),
                             int(25 * (i // 2)):int(25 * (i // 2) + 25)] = 1
                # distribution = np.ones((50, 50))
                distribution = distribution / np.sum(distribution)
                # print(distribution)
            else:
                distribution = info[i]

            cur_reward, team = self.get_grid_reward(int_alloc[:, i], distribution, terrain)
            teams.append(team)
            reward += cur_reward
            reward_list.append(cur_reward)
        if plot:
            self.draw_agent(teams, terrain)
        if save_location:
            loc_list = []
            team_idx = 0
            for team in teams:
                for agent in team:
                    loc_list.append([agent.color, agent.cur_traj, team_idx])
                team_idx += 1
            np.save("./MAETF/data_loc/loc", np.array(loc_list))
        return reward, reward_list

    # alloc: ngrid x nagent
    # env_type: ngrid x 1 vector
    # info: ngrid x 1 vector
    def get_reward(self, alloc, terrain, info=None, plot=False, print_info=False, save_location=False):
        int_alloc = self.get_integer(alloc)
        reward, reward_list = self.get_integer_reward(int_alloc, terrain, info, plot, print_info, save_location)
        return reward, reward_list

    # get reward of a single task
    def get_grid_reward(self, agents_num, distribution, terrain):
        # print(agents_num)
        car_num = agents_num[0]
        ship_num = agents_num[1]
        drone_num = agents_num[2]
        team = (
                [copy.deepcopy(self.agent1) for _ in range(car_num)] +
                [copy.deepcopy(self.agent2) for _ in range(ship_num)] +
                [copy.deepcopy(self.agent3) for _ in range(drone_num)]
        )

        # #we simply return nan
        # if len(team) == 0:
        #     return np.inf, team

        start_states = None  # [[.2, .7] for i in range(6)] + [[.7, .2] for i in range(2)]
        # team1 = [copy.deepcopy(agent2) for i in range(5)]
        # start_states = [[np.random.random(), .25] for i in range(5)]

        erg1, team_fourier1, dist_fourier1 = optimize_team_erg(team, self.n_types_terrain, terrain, distribution,
                                                               self.erg_calc, start_states=start_states,
                                                               n_opt=self.nopt)
        # print('erg team', erg1)
        return erg1, team

    def draw_agent(self, teams, terrain):
        from matplotlib import colors
        env_type_color = ['navy', 'saddlebrown', 'forestgreen', 'ghostwhite']
        cmap = colors.ListedColormap(env_type_color)
        fig, ax = plt.subplots()
        im = ax.imshow(terrain.T, origin='lower', extent=(0, 1, 0, 1), cmap=cmap)


        for team in teams:
            for agent in team:
                agent.plot_self(ax)
        # plt.colorbar(im)
        plt.show()



    def generate_random_alloc(self, num):
        assignment = np.random.uniform(-1, 1, [num, self.n_types_agents, self.n_num_grids])
        assignment = softmax(assignment, axis=-1)
        int_assignment = np.array([self.get_integer(alloc) for alloc in assignment])
        return int_assignment, assignment


if __name__ == '__main__':
    # Still a little weird

    env = MultiAgentEnv(nopt=1500, erg_num=15)
    # the part about forming compositions
    n_rand = 1
    min_erg = np.inf
    env_types = [1, 0, 2, 3]  # this is the default
    env_types = [3, 2, 2, 1]
    terrain = np.zeros((50, 50), dtype=np.int32)  # this correspond to the default second region - lake

    terrain[0:25, 0:25] = env_types[0]  # this correspond to the default first region - mountain
    terrain[25:, 0:25] = env_types[1]
    terrain[0:25, 25:] = env_types[2]  # this correspond to the default third region - plain
    terrain[25:, 25:] = env_types[3]  # this correspond to the default fourth region - city
    color = []
    for _ in range(n_rand):
        # assignment = np.random.uniform(-1, 1, [env.n_types_agents, env.n_num_grids])
        # assignment = softmax(assignment, axis=-1)
        # print(assignment)
        # # assignment = np.asarray([[0.19839782, 0.24886413, 0.44777245, 0.1049656],
        # #                          [0.19769573, 0.3426687, 0.3618094, 0.09782617],
        # #                          [0.2280346, 0.21226203, 0.1925472, 0.36715617]])
        # reward, reward_list = env.get_reward(assignment, terrain, plot=True, print_info=True)

        # four square:
        # 2 3
        # 0 1

        # four square rotated:
        # 1 0
        # 3 2
        ########## alloc multimodal  ################
        # int_alloc = np.array([[3, 0, 11, 6,],
        #                       [0, 16, 0, 4,],
        #                       [11, 3, 1, 5,]])

        # int_alloc = np.array([[11, 0, 0, 9, ],
        #                       [0, 16, 0, 4, ],
        #                       [10, 0, 10, 0, ]])

        int_alloc = np.array([[17, 0, 1, 2, ],
                              [0, 9, 0, 11, ],
                              [6, 5, 9, 0, ]])
        ########## end of alloc multimodal  ################
        int_alloc = np.array([[0, 0, 0, 20, ],
                              [2, 7, 9, 0, ],
                              [11, 5, 4, 0, ]])

        int_alloc = np.array([[0, 0, 14, 6, ],
                              [20, 0, 0, 0, ],
                              [3, 11, 0, 6, ]])

        int_alloc = np.array([[8, 2, 5, 5, ],
                              [0, 8, 2, 10, ],
                              [9, 1, 9, 1, ]])

        int_alloc = np.array([[2, 11, 7, 0, ],
                              [20, 0, 0, 0, ],
                              [0, 2, 8, 10, ]])

        reward, reward_list = env.get_integer_reward(int_alloc, terrain, plot=True, print_info=True, save_location=True)
        print(reward)
        print(reward_list)
        reward_list = np.array(reward_list)
        print(np.log(reward_list))
        print(np.sum(np.log(reward_list)))
        # mountain - lake - plain - city in network
        # lake, mountain, plain, city in the simulation