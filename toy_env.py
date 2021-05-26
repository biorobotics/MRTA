'''
Author: Jiaheng Hu

This env defines a toy MRTA problem.

# current rule: info means potential (a 1d value)
# terrain has 4 different types:
#   0, city: car + aircraft, each single 1: 0.1, combo: 1.0
#   1, mountain: aircraft:0.3, rest: 0.1
#   2, plain: ship + car:1.0, rest: 0.1
#   3, lake: ship: 0.25, rest: 0.1

# drone: 0
# car: 1
# boat: 2

'''

import numpy as np


class MultiAgentEnv:
    def __init__(self, n_num_grids=4, n_num_agents=3, n_env_types=4, agent_num=[100, 100, 100]):
        self.n_num_grids = n_num_grids
        self.n_types_agents = n_num_agents
        self.n_types_terrain = n_env_types
        self.agent_num = agent_num

    # turn continuous alloc into discrete assignment
    def get_integer(self, alloc):
        alloc = alloc.T
        int_alloc = np.zeros_like(alloc)
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

    # alloc: ngrid x nagent
    # env_type: ngrid x 1 vector
    # info: ngrid x 1 vector
    def get_reward(self, alloc, env_type, info=[1] * 4):
        int_alloc = self.get_integer(alloc)
        reward = self.get_integer_reward(int_alloc, env_type, info)
        return reward

    def get_integer_reward(self, int_alloc, env_type, info=[1] * 4):
        reward = 0
        for i in range(self.n_num_grids):
            cur_reward = self.get_grid_reward(int_alloc[:, i], info[i], env_type[i])
            reward += cur_reward
        return reward

    # def get_grid_reward(self, agents_num, info, env_type):
    #     plane_num = agents_num[0]
    #     car_num = agents_num[1]
    #     ship_num = agents_num[2]
    #     # 0, plain: car + aircraft, each single 1: 0.1, combo: 1.0
    #     # 1, mountain: aircraft:0.3, rest: 0.1
    #     # 2, river: ship + car:1.0, rest: 0.1
    #     # 3, lake: ship: 0.25, rest: 0.1
    #     if env_type == 0:
    #         combo_r = np.min([plane_num, car_num])
    #         single_r = np.abs(plane_num - car_num) * 0.1
    #         rest_r = ship_num * 0.1
    #         reward = combo_r + single_r + rest_r
    #     elif env_type == 1:
    #         reward = plane_num * 0.3 + (car_num + ship_num) * 0.1
    #     elif env_type == 2:
    #         combo_r = np.min([ship_num, car_num])
    #         single_r = np.abs(ship_num - car_num) * 0.1
    #         rest_r = plane_num * 0.1
    #         reward = combo_r + single_r + rest_r
    #     elif env_type == 3:
    #         reward = ship_num * 0.25 + (car_num + plane_num) * 0.1
    #     return reward * info

    # Version 2: this should be closer to the actual simulation
    def get_grid_reward(self, agents_num, info, env_type):
        plane_num = agents_num[0]
        car_num = agents_num[1]
        ship_num = agents_num[2]
        # 0, plain: car:5, aircraft:5, ship:0
        # 1, mountain: car:2, aircraft:5, ship:0
        # 2, city: car:5, aircraft:3, ship:2
        # 3, lake: car:0, aircraft:5, ship:5
        if env_type == 0:
            reward = car_num * 5 + plane_num * 5 + ship_num * 0
        elif env_type == 1:
            reward = car_num * 2 + plane_num * 5 + ship_num * 0
        elif env_type == 2:
            reward = car_num * 5 + plane_num * 3 + ship_num * 5
        elif env_type == 3:
            reward = car_num * 0 + plane_num * 5 + ship_num * 5
        reward = np.sqrt(reward)
        return reward

    def generate_random_dist_and_reward(self, num, env_type, constraint=False):
        if constraint:
            # currently only work for [0, 1, 2, 3]
            # #city, mountain, plain, lake
            # #drone car ship
            rand = np.zeros([num, self.n_types_agents, self.n_num_grids])

            # no ship in mountain & city
            rand_ship_non_zero = np.random.uniform(0, 1, [num, 2])
            rand_ship_non_zero /= np.sum(rand_ship_non_zero, axis=-1)[:, np.newaxis]
            rand[:, 2, 2] = rand_ship_non_zero[:, 0]
            rand[:, 2, 3] = rand_ship_non_zero[:, 1]

            # drone can be anywhere
            rand_drone = np.random.uniform(0, 1, [num, 4])
            rand_drone /= np.sum(rand_drone, axis=-1)[:, np.newaxis]
            rand[:, 0, :] = rand_drone

            # No car in lake
            rand_car_non_zero = np.random.uniform(0, 1, [num, 3])
            rand_car_non_zero /= np.sum(rand_car_non_zero, axis=-1)[:, np.newaxis]
            rand[:, 1, 0] = rand_car_non_zero[:, 0]
            rand[:, 1, 1] = rand_car_non_zero[:, 1]
            rand[:, 1, 2] = rand_car_non_zero[:, 2]
        else:
            rand = np.random.uniform(0, 1, [num, self.n_types_agents, self.n_num_grids])
        rand_res = rand / np.sum(rand, axis=-1)[:, :, np.newaxis]
        reward = np.asarray([self.get_reward(dist, env_type) for dist in rand_res])
        return rand_res, reward

    # collect training data for the reward network
    def generate_random_dist_and_reward_per_env(self, env_types):
        rand = np.random.uniform(0, 1, [env_types.shape[0], self.n_types_agents, self.n_num_grids])
        rand_res = rand / np.sum(rand, axis=-1)[:, :, np.newaxis]

        dummy = zip(rand_res, env_types)
        reward = np.asarray([self.get_reward(dist, env_type) for dist, env_type in dummy])
        return rand_res, reward

    # collect training data for the reward network
    def generate_fixed_dist_and_reward_per_env(self, env_types):
        dist = np.array([[0.0, 1.0, 0, 0], [0, 0, 1.0, 0.0], [0, 0, 1.0, 0.0]])
        reward = np.asarray([self.get_reward(dist, env_type) for env_type in env_types])
        return np.tile(dist, (env_types.shape[0], 1)), reward


    def test_dist(self, env_type):
        dist1 = np.ones([self.n_types_agents, self.n_num_grids]) / self.n_num_grids
        dist2 = np.asarray([[0.1, 0.1, 0.5, 0.3]] * self.n_types_agents)
        dist3 = np.asarray([[0.2, 0.4, 0.1, 0.3]] * self.n_types_agents)
        dists = np.asarray([dist1, dist2, dist3])
        reward = np.asarray([self.get_reward(dist, env_type) for dist in dists])
        # print(dists, reward)
        return dists, reward

if __name__ == '__main__':
    env = MultiAgentEnv()
    # print(env.test_dist(env_type=[0, 1, 2, 3]))

    robot, reward = env.generate_random_dist_and_reward(50, env_type=[0, 1, 2, 3])
    max_reward = np.max(reward)
    min_reward = np.min(reward)
    max_robot = robot[np.argmax(reward)]
    print(max_reward, min_reward, max_robot, env.get_integer(max_robot))
    print(reward.mean())
