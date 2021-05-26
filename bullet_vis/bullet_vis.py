import pybullet as p
import numpy as np
from time import sleep
import math

floor_scaling = 0.2
init_pos = 15 * floor_scaling
robot_pos_scaling = floor_scaling * 2 * 30
interp = 10
sleep_t = 1 / interp

physicsClient = p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
p.setAdditionalSearchPath("./robot_urdfs")


def load_terrain():
    plane1Id = p.loadURDF('plane.urdf', basePosition=[init_pos, init_pos, 0], globalScaling=floor_scaling)
    plane2Id = p.loadURDF('plane.urdf', basePosition=[init_pos, -init_pos, 0], globalScaling=floor_scaling)
    plane3Id = p.loadURDF('plane.urdf', basePosition=[-init_pos, -init_pos, 0], globalScaling=floor_scaling)
    plane4Id = p.loadURDF('plane.urdf', basePosition=[-init_pos, init_pos, 0], globalScaling=floor_scaling)

    planes = [plane1Id, plane2Id, plane3Id, plane4Id]
    sea_texture_path = "textures/ocean.jpeg"
    city_texture_path = "textures/city.jpg"
    mountain_texture_path = "textures/mountain.jpg"
    plain_texture_path = "textures/plain.jpg"

    sea_texture_ID = p.loadTexture(sea_texture_path)
    cit_texture_ID = p.loadTexture(city_texture_path)
    mount_texture_ID = p.loadTexture(mountain_texture_path)
    plain_texture_ID = p.loadTexture(plain_texture_path)

    p.changeVisualShape(plane1Id, -1, textureUniqueId=cit_texture_ID)
    p.changeVisualShape(plane2Id, -1, textureUniqueId=sea_texture_ID)
    p.changeVisualShape(plane3Id, -1, textureUniqueId=mount_texture_ID)
    p.changeVisualShape(plane4Id, -1, textureUniqueId=plain_texture_ID)

def load_robot(loc_name):
    location = np.load(loc_name)
    id_list = []
    for i in range(location.shape[0]):
        scale = 0.1
        if location[i, 0] == 'lime':
            # car
            urdf = 'jeep.urdf'
            # print("loaded car")
        elif location[i, 0] == 'lightblue':
            # ship (duck)
            urdf = 'ship.urdf'
            # scale = 3
            # print("loaded ship")
        elif location[i, 0] == 'red':
            # drone
            urdf = 'aircraft.urdf'
            # print("loaded aircraft")
        else:
            print(location[i, 0])
            # print('lime' is 'lime')
            # print(location[i, 0] == 'lime')
            exit("error with loaded data")
        id_list.append(p.loadURDF(urdf, globalScaling=scale))

    # Todo: update loc every simulation step
    return id_list, location


def update_robot(time, id_list, location):

    if time == 0:
        for i in range(len(id_list)):
            new_pos_x = location[i, 1][time, 0] * robot_pos_scaling - robot_pos_scaling / 2
            new_pos_y = location[i, 1][time, 1] * robot_pos_scaling - robot_pos_scaling / 2
            z_pos = 0

            # TODO: change quat
            quat = [0, 0, 0, 1]
            if location[i, 0] == 'lime':
                # car
                # z_pos = 0.15
                pass
            elif location[i, 0] == 'lightblue':
                # ship (duck)
                # quat = [0.7071068, 0, 0, 0.7071068]
                pass
            elif location[i, 0] == 'red':
                z_pos = 2
                #oh it's because of the topdown view
                new_pos_x*=0.8
                new_pos_y*=0.8
            else:
                exit("error line 73")
            p.resetBasePositionAndOrientation(id_list[i], [new_pos_x, new_pos_y, z_pos], quat)
        p.stepSimulation()
        sleep(sleep_t)
    else:
        
        for j in range(interp):
            for i in range(len(id_list)):
                old_pos_x = location[i, 1][time-1, 0] * robot_pos_scaling - robot_pos_scaling / 2
                old_pos_y = location[i, 1][time-1, 1] * robot_pos_scaling - robot_pos_scaling / 2

                tar_pos_x = location[i, 1][time, 0] * robot_pos_scaling - robot_pos_scaling / 2
                tar_pos_y = location[i, 1][time, 1] * robot_pos_scaling - robot_pos_scaling / 2

                angle = math.atan2(tar_pos_y-old_pos_y, tar_pos_x-old_pos_x)
                angle += np.pi / 2
                quat = p.getQuaternionFromEuler([0, 0, angle])

                alpha = (float(j)+1) / interp
                new_pos_x = tar_pos_x * alpha + old_pos_x * (1 - alpha)
                new_pos_y = tar_pos_y * alpha + old_pos_y * (1 - alpha)
                z_pos = 0

                # TODO: change quat
                # quat = [0, 0, 0, 1]
                if location[i, 0] == 'lime':
                    # car
                    # z_pos = 0.15
                    pass
                elif location[i, 0] == 'lightblue':
                    # ship (duck)
                    # quat = [0.7071068, 0, 0, 0.7071068]
                    pass
                elif location[i, 0] == 'red':
                    z_pos = 2
                    #oh it's because of the topdown view
                    new_pos_x*=0.8
                    new_pos_y*=0.8
                else:
                    exit("error line 118")
                p.resetBasePositionAndOrientation(id_list[i], [new_pos_x, new_pos_y, z_pos], quat)
            p.stepSimulation()
            sleep(sleep_t)


if __name__ == '__main__':
    p.resetSimulation()  # remove all objects from the world and reset the world to initial conditions. (not needed here but kept for example)

    load_terrain()
    loc_name = "loc.npy"
    id_list, location = load_robot(loc_name)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=physicsClient)
    p.resetDebugVisualizerCamera(5, 0, -45, [0, 0, 0], physicsClientId=physicsClient)  # I like this view
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=physicsClient)

    vid_path = 'videos/test.mp4'

    # if not os.path.exists(vid_path):
    logID = p.startStateLogging(
        p.STATE_LOGGING_VIDEO_MP4,
        fileName=vid_path)

    for i in range(20):
        update_robot(i, id_list, location)
        
