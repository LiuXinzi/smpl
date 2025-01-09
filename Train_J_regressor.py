import pickle
import numpy as np
from scipy.optimize import nnls
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def save_more_groups(points_3d_skin, points_3d_joint1, points_3d_joint2, name='combined_points'):

    num_samples = points_3d_skin.shape[0]

    fig = plt.figure(figsize=(20, 15))
    cols = 4 
    rows = -(-num_samples // cols)  

    for i in range(num_samples):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

   
        ax.scatter(points_3d_skin[i, :, 0], points_3d_skin[i, :, 1], points_3d_skin[i, :, 2],
                   s=1, alpha=0.3, label='Skin Points', c='lightblue')


        ax.scatter(points_3d_joint1[i, :, 0], points_3d_joint1[i, :, 1], points_3d_joint1[i, :, 2],
                   s=20, alpha=0.8, label='True Joint', c='red') 

        ax.scatter(points_3d_joint2[i, :, 0], points_3d_joint2[i, :, 1], points_3d_joint2[i, :, 2],
                   s=20, alpha=0.8, label='J Joint', c='green')  

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Sample {i + 1}')
        ax.legend()

        all_points = np.vstack((points_3d_skin[i], points_3d_joint1[i], points_3d_joint2[i]))
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                              all_points[:, 1].max() - all_points[:, 1].min(),
                              all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


        ax.view_init(elev=90, azim=270)

    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=1000, bbox_inches='tight')
    plt.close(fig)
def save_three_groups(points_3d_skin, points_3d_joint1, points_3d_joint2, name='combined_points'):

    fig = plt.figure(figsize=(20, 15))  
    ax = fig.add_subplot(111, projection='3d')
    

    ax.scatter(points_3d_skin[:, 0], points_3d_skin[:, 1], points_3d_skin[:, 2], 
               s=1, alpha=0.3, label='Skin Points', c='lightblue')

    ax.scatter(points_3d_joint1[:, 0], points_3d_joint1[:, 1], points_3d_joint1[:, 2], 
               s=20, alpha=0.8, label='True_joint', c='red')  # 深红色

    ax.scatter(points_3d_joint2[:, 0], points_3d_joint2[:, 1], points_3d_joint2[:, 2], 
               s=20, alpha=0.8, label='J_joint', c='green')  # 深绿色
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(name)
    ax.legend()

    all_points = np.vstack((points_3d_skin, points_3d_joint1, points_3d_joint2))
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(), 
                          all_points[:, 1].max() - all_points[:, 1].min(), 
                          all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # 设置视角：俯视图，从 Z 轴向下看，并在 XY 平面逆时针旋转 90°
    ax.view_init(elev=90, azim=270)
    # plt.show()
    plt.savefig(f"{name}.png", dpi=300, bbox_inches='tight')  # 使用 dpi=300 提高分辨率
    plt.close(fig)
def nnls_with_constraints(A, b, lambda_sum=0.5, lambda_ridge=1):
    n = A.shape[1]
    
    sqrt_lambda_sum = np.sqrt(lambda_sum)
    sqrt_lambda_ridge = np.sqrt(lambda_ridge)

    A_ext = np.vstack([
        A,
        sqrt_lambda_sum * np.ones((1, n)),
        sqrt_lambda_ridge * np.identity(n)
    ])

    b_ext = np.concatenate([
        b,
        [sqrt_lambda_sum],
        np.zeros(n)
    ])

    x, rnorm = nnls(A_ext, b_ext)
    
    return x

def train_J(smpl_skin, smpl_joint, lambda_sum=1.0, lambda_ridge=0.1):
    n_samples, num_vertices, _ = smpl_skin.shape  # (n_samples, 6890, 3)
    _, num_joints, _ = smpl_joint.shape          # (n_samples, 24, 3)
    J = np.zeros((num_joints, num_vertices))     # (24, 6890)

    A = smpl_skin.transpose(0, 2, 1).reshape(n_samples * 3, num_vertices)

    for k in range(num_joints):
        b = smpl_joint[:, k, :].reshape(n_samples * 3)
        x_k = nnls_with_constraints(A, b, lambda_sum=lambda_sum, lambda_ridge=lambda_ridge)
        J[k, :] = x_k
        print(f"Joint {k}, Sum of weights: {np.sum(x_k):.4f}, Non-zero weights: {np.count_nonzero(x_k)}")
    return J

def main():
    smpl_skin = np.load("smpl_skin.npy")     # (n_samples, 6890, 3)

    ## Centers from Li-san
    # with open('joint_center.pkl', 'rb') as file:
    #     loads = pickle.load(file)
    #     for key, value in loads.items():
    #         if key=='joint_center':
    #             smpl_joint=np.array(value)

    smpl_joint = np.load("smpl_joint.npy")   # (n_samples, 24, 3)

    J = train_J(smpl_skin[:5000], smpl_joint[:5000], lambda_sum=1., lambda_ridge=0.0000000001)
    np.save("J_trained.npy", J)

    # J=np.load("J_trained.npy")

    n_samples = smpl_skin.shape[0]
    predicted_joints = np.empty_like(smpl_joint)  # (n_samples, 24, 3)
    for i in range(n_samples):
        predicted_joints[i, :, :] = J @ smpl_skin[i, :, :]
    for i in range(1):
        save_more_groups(smpl_skin[:12],smpl_joint[:12],predicted_joints[:12])
    errors = predicted_joints[5000:] - smpl_joint[5000:]
    mse_loss = np.mean(errors ** 2)
    print("Mean Squared Error Loss:", mse_loss)
 

if __name__ == "__main__":
    main()
