import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def region_plot(obj, obj_label, ax):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(b=True, which='major', color='k', linestyle='--')
    for key in obj:
        color = 'b' if obj_label != 'region' else 'c'
        for grid in obj[key]:
            x_ = grid[0]
            y_ = grid[1]
            x = []
            y = []
            patches = []
            for point in [(x_, y_), (x_ + 1, y_), (x_ + 1, y_ + 1), (x_, y_ + 1)]:
                x.append(point[0])
                y.append(point[1])
            polygon = Polygon(np.column_stack((x, y)), True)
            patches.append(polygon)
            p = PatchCollection(patches, facecolors=color, edgecolors=color, alpha=0.4)
            ax.add_collection(p)
        ax.text(np.mean(x) - 0.2, np.mean(y) - 0.2, r'${}_{{{}}}$'.format(key[0], key[1:]), fontsize=8)


def path_plot(robot_path, regions, obs):
    """
    plot the optimal path in the 2D and 3D
    :param path: ([pre_path], [suf_path])
    :param regions: regions
    :param obs: obstacle
    :return: none
    """

    for robot, path in robot_path.items():
        # prefix path
        if len(path) == 1:
            continue
        x_pre = np.asarray([point[0] + 0.5 for point in path])
        y_pre = np.asarray([point[1] + 0.5 for point in path])
        plt.quiver(x_pre[:-1], y_pre[:-1], x_pre[1:] - x_pre[:-1], y_pre[1:] - y_pre[:-1],
                   color="#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]),
                   scale_units='xy', angles='xy', scale=1, label='prefix path')

        plt.savefig('img/path.png', bbox_inches='tight', dpi=600)
