import cv2
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from scipy.stats import entropy
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

# hyperparameter
num_of_region = 4
num_of_terrain_type = 4

def get_terrain_map():
    img = cv2.imread('./Thoune.png')
    res = cv2.resize(img, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("terr",res)
    # cv2.waitKey(0)
    res = cv2.GaussianBlur(res,(5,5),cv2.BORDER_DEFAULT)
    # cv2.imshow("terr",res)
    # cv2.waitKey(0)
    img_array = np.asarray(res)
    # print(img_array.shape)
    terr_map = np.argmax(img_array, axis=-1)
    # print(terr_map)
    terr_map[terr_map==1] = 4 #dummy, temp solution
    terr_map[terr_map==2] = 3
    terr_map[terr_map==4] = 2
    terr_map[0,0] = 3 # just so there won't be error, I don't know why but it seems that the env have to contain all the terr for the visualization to be correct
    image_bw = np.sum(img_array, axis=-1)
    terr_map[image_bw>300] = 1
    # 1 (green) =>  2
    # 3 (white) => 1
    # 2 (red) => 3
    # actual: lake, mountain, plain, city
    # current: BGR
    return terr_map

# get the information map based on partition
def obtain_info_map(poly_l = None):
    # use default polygon list
    if poly_l == None:
        pl = np.array([[0, 0],[1, 0], [1, 1], [0, 1], [0.25, 1], [0.5, 0.7], [0.7, 0.38], [0.3, 0], [0, 0.2], [1, 0.38]])*50
        poly1 = pl[[3,4,5,8]]
        poly2 = pl[[4,2,9,6]]
        poly3 = pl[[6,9,1,7]]
        poly4 = pl[[8,5,6,7,0]]
        poly_l = [poly1,poly2,poly3,poly4]
    else:
        poly_l = np.array(poly_l)*50
    info_l = []
    for poly in poly_l:
        path = mpltPath.Path(poly)
        points = np.dstack(np.meshgrid(range(50), range(50))).reshape(-1, 2)
        inside2 = path.contains_points(points)
        info_l.append(inside2.reshape(50,50))
    # plt.imshow(inside2.reshape(50,50), origin='lower')
    # plt.show()
    # print(sum(inside2))
    return info_l

def calculate_region_stats(distribution, terr_map):
    a_list = terr_map[distribution != 0].flatten()
    histo_arry = np.array([(a_list == i).sum() for i in range(num_of_terrain_type)])
    plt.bar(["lake", "mountain", "plain", "city"], histo_arry)
    # plt.show()
    ent = entropy(histo_arry/histo_arry.sum())
    return a_list.shape[0], ent

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def voronoi_partition(points):
    # compute Voronoi tesselation
    vor = Voronoi(points)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor, 500000)

    min_x = 0.0
    max_x = 1.0
    min_y = 0.0
    max_y = 1.0

    # mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
    # bounded_vertices = np.max((vertices, mins), axis=0)
    # maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
    # bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

    box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    region_polygons = []
    # colorize
    for region in regions:
        polygon = vertices[region]
        # Clipping polygon
        poly = Polygon(polygon)
        poly = poly.intersection(box)
        polygon = [p for p in poly.exterior.coords]
        region_polygons.append(np.array(polygon))
    return region_polygons

def draw_voronoi(region_polygons):
    # colorize
    for region in region_polygons:
        plt.fill(*zip(*region), alpha=0.4)

    plt.plot(points[:, 0], points[:, 1], 'ko')
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

def calculate_fitness(terr_map, points):
    region_polygons = voronoi_partition(points)
    distributions = obtain_info_map(region_polygons)
    entropy_sum = 0
    area_list = []
    for i in range(num_of_region):
        area, ent = calculate_region_stats(distributions[i], terr_map)
        entropy_sum += ent
        area_list.append(area)
    return entropy_sum, np.std(area_list)

if __name__ == '__main__':
    np.set_printoptions(precision=3)

    terr_map = get_terrain_map()
    # make up data points
    for i in range(5):
        points = np.random.rand(4, 2)
        print(calculate_fitness(terr_map, points))



    exit()
    env_type_color = ['navy', 'saddlebrown', 'forestgreen', 'ghostwhite']
    cmap = colors.ListedColormap(env_type_color, N=4) # todo: figure out the color map thing
    fig, ax = plt.subplots()
    im = ax.imshow(terr_map.T, origin='lower', extent=(0, 1, 0, 1), cmap=cmap)
    # plt.show()
    x1, y1 = [0.25, 0.7], [1, 0.38]
    x2, y2 = [0.7, 1.0], [0.38, 0.38]
    x3, y3 = [0.7, .3], [0.38, 0]
    x4, y4 = [0., .5], [0.2, 0.66]
    plt.plot(x1, y1, x2, y2,x3, y3,x4, y4, color='black')
    plt.show()