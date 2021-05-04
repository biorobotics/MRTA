

import numpy as np



# current rule: info means potential (a 1d value)
# terrain has 4 different types:
#   0, city: car + aircraft, each single 1: 0.1, combo: 1.0
#   1, mountain: aircraft:0.3, rest: 0.1
#   2, plain: ship + car:1.0, rest: 0.1
#   3, lake: ship: 0.25, rest: 0.1

# drone: 0
# car: 1
# boat: 2

# initially assumes uniform info
# right now this can be solved greedily: look for ways to make this more complex

class MultiAgentEnv:
    def __init__(self, n_num_grids=4, n_num_agents=3, n_env_types=4, agent_num=[100, 100, 100]):
        self.n_num_grids = n_num_grids
        self.n_num_agents = n_num_agents
        self.n_env_types = n_env_types
        self.agent_num = agent_num

    # alloc: ngrid x nagent
    # turn continuous alloc into discrete assignment
    def getInteger(self, alloc):
        int_alloc = np.zeros_like(alloc)
        for i in range(self.n_num_agents):
            remaining = self.agent_num[i]
            for j in range(self.n_num_grids):
                cur_num = int(alloc[j][i]*self.agent_num[i])
                cur_num = np.min([remaining, cur_num])
                remaining -= cur_num
                int_alloc[j][i] = cur_num
        return int_alloc

    # alloc: ngrid x nagent
    # env_type: ngrid x 1 vector
    # info: ngrid x 1 vector
    def getReward(self, alloc, env_type, info=[1]*4):
        int_alloc = self.getInteger(alloc)
        reward = 0
        for i in range(self.n_num_grids):
            cur_reward = self.getGridReward(int_alloc[i], info[i], env_type[i])
            reward += cur_reward
        return reward

    def getGridReward(self, agents_num, info, env_type):
        plane_num = agents_num[0]
        car_num = agents_num[1]
        ship_num = agents_num[2]
        # 0, plain: car + aircraft, each single 1: 0.1, combo: 1.0
        # 1, mountain: aircraft:0.3, rest: 0.1
        # 2, river: ship + car:1.0, rest: 0.1
        # 3, lake: ship: 0.25, rest: 0.1
        if env_type == 0:
            combo_r = np.min([plane_num, car_num])
            single_r = np.abs(plane_num - car_num) * 0.1
            rest_r = ship_num * 0.1
            reward = combo_r + single_r + rest_r
        elif env_type == 1:
            reward = plane_num * 0.3 + (car_num + ship_num) * 0.1
        elif env_type == 2:
            combo_r = np.min([ship_num, car_num])
            single_r = np.abs(ship_num - car_num) * 0.1
            rest_r = plane_num * 0.1
            reward = combo_r + single_r + rest_r
        elif env_type == 3:
            reward = ship_num * 0.25 + (car_num + plane_num) * 0.1
        return reward * info

    def generate_random_dist_and_reward(self, num, env_type, constraint=False):
        if constraint:
            # currently only work for [0, 1, 2, 3]
            # #city, mountain, plain, lake
            # #drone car ship
            rand = np.zeros([num, self.n_num_agents, self.n_num_grids])

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
            rand = np.random.uniform(0, 1, [num, self.n_num_agents, self.n_num_grids])
        rand_res = rand / np.sum(rand, axis=-1)[:, :, np.newaxis]
        reward = np.asarray([self.getReward(dist.T, env_type) for dist in rand_res])
        return rand_res, reward

    def test_dist(self, env_type):
        dist1 = np.ones([self.n_num_agents, self.n_num_grids]) / self.n_num_grids
        dist2 = np.asarray([[0.1, 0.1, 0.5, 0.3]] * self.n_num_agents)
        dist3 = np.asarray([[0.2, 0.4, 0.1, 0.3]] * self.n_num_agents)
        dists = np.asarray([dist1, dist2, dist3])
        reward = np.asarray([self.getReward(dist.T, env_type) for dist in dists])
        # print(dists, reward)
        return dists, reward

if __name__ == '__main__':
    env = MultiAgentEnv()
    env.test_dist(env_type=[0, 1, 2, 3])

    # robot, reward = env.generate_random_dist_and_reward(50, env_type=[0, 1, 2, 3])
    # max_reward = np.max(reward)
    # min_reward = np.min(reward)
    # max_robot = robot[np.argmax(reward)]
    # print(max_reward, min_reward, max_robot, env.getInteger(max_robot.T))
    # print(reward.mean())
