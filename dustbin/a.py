import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb
import numpy as np
from star.ch.star import STAR
from scipy.stats import truncnorm
import numpy as np
from scipy.stats import norm
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
import numpy as np
from scipy.optimize import leastsq
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np

def plot_human_structure(skin_points, joint_points, joint_labels=None, name='human_structure'):
    joint_points=np.array(joint_points)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(skin_points[:, 0], skin_points[:, 1], skin_points[:, 2], 
               s=1, alpha=0.3, label='Skin Points',  c='lightblue')
    ax.scatter(joint_points[:, 0], joint_points[:, 1], joint_points[:, 2], 
               s=50, alpha=1.0, label='Joint Points', c='red', marker='o')
    for i, (x, y, z) in enumerate(joint_points):
        label = f'{i}' if joint_labels is None else joint_labels[i]
        ax.text(x, y, z, label, color='black', fontsize=8, weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(name)
    ax.legend()
    all_points = np.vstack((skin_points, joint_points))
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(), 
                          all_points[:, 1].max() - all_points[:, 1].min(), 
                          all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()

model = STAR(gender='female', num_betas=10)

plot_human_structure(model.v_shaped,np.load("J_trained.npy") @ np.array(model.v_shaped))