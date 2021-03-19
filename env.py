import numpy as np

# another question is how do we measure frequency? (based on the actual discrete assignment? yeah)
# try to make it possible for multiple solution to occur
# the conditioning is a bit chunky though...
# RN: works on arbitrary number of agents: no need for

# current rule: info means potential (a 1d value)
# terrain has 4 different types:
#   0, plain: car + aircraft, each single 1: 0.1, combo: 1.0
#   1, mountain: aircraft:0.3, rest: 0.1
#   2, river: ship + car:1.0, rest: 0.1
#   3, lake: ship: 0.25, rest: 0.1

# plane: 0
# car: 1
# boat: 2

# initially assumes uniform info

# right now this can be solved greedily: look for ways to make this more complex

class MultiagentEnv:
    def __init__(self, ngrid=4, nagent=3, nenv=4, agent_num=[100, 100, 100]):
        self.ngrid = ngrid
        self.nagent = nagent
        self.nenv = nenv
        self.agent_num = agent_num

    # alloc: ngrid x nagent
    # turn continuous alloc into discrete assignment
    def getInteger(self, alloc):
        int_alloc = np.zeros_like(alloc)
        for i in range(self.nagent):
            remaining = self.agent_num[i]
            for j in range(self.ngrid):
                cur_num = int(alloc[j][i]*self.agent_num[i])
                cur_num = np.min([remaining, cur_num])
                remaining -= cur_num
                int_alloc[j][i] = cur_num
        return int_alloc

    # alloc: ngrid x nagent
    # env_type: ngrid x 1 vector
    # info: ngrid x 1 vector
    def getReward(self, alloc, info=[1]*4, env_type=[0, 1, 2, 3]):
        int_alloc = self.getInteger(alloc)
        reward = 0
        for i in range(self.ngrid):
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

    def generate_random_dist_and_reward(self, num):
        rand = np.random.uniform(0, 1, [num, self.nagent, self.ngrid])
        rand_res = rand / np.sum(rand, axis=-1)[:, :, np.newaxis]
        reward = np.asarray([self.getReward(dist.T) for dist in rand_res])
        return rand_res, reward

if __name__ == '__main__':
    env = MultiagentEnv()
    # print(env.getReward([[1.0, 0.5, 0], [0.0, 0.5, 1]], [1, 1], [0, 1]))
    robot, reward = env.generate_random_dist_and_reward(1000)
    max_reward = np.max(reward)
    max_robot = robot[np.argmax(reward)]
    print(max_reward, max_robot, env.getInteger(max_robot.T))
    print(reward.mean())
