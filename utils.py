import numpy as np
import ipdb
import matplotlib.pyplot as plt
import numpy as np

def plot_human_structure(skin_points, joint_points=None, joint_labels=None, name='human_structure'):
    # import ipdb;ipdb.set_trace()
    if joint_points.any()!=None:
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
    if joint_points==None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(skin_points[:, 0], skin_points[:, 1], skin_points[:, 2], 
                s=1, alpha=0.3, label='Skin Points',  c='lightblue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        all_points = skin_points
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

def visualize_matrix(matrix,name=''):
    # visualize_matrix(J_T,'w_nnsum1')
    plt.figure(figsize=(5, 25)) 
    plt.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Value") 
    plt.title("Heatmap of 24x6890 Matrix")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(f"{name}.png") 


J_T = np.load('result\\W_nnsum1\\W.npy')
visualize_matrix(J_T,'w_nnsum1')


