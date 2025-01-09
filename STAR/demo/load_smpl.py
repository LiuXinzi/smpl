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

def interactive_model_visualization(model):
    app = dash.Dash(__name__)

    # Define layout
    app.layout = html.Div([
        html.H1("Interactive 3D Model Visualization"),
        html.Div([
            html.Div([
                html.Label(f"Dimension {i+1}:"),
                dcc.Slider(
                    id=f"slider-{i}",
                    min=-10, max=10, step=0.1, value=0,
                    marks={-10: '-10', 0: '0', 10: '10'},
                    tooltip={"placement": "bottom", "always_visible": True}  # Add tooltip to display the current value
                )
            ], style={'margin': '5px', 'width': '45%'}) for i in range(10)
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),
        dcc.Graph(id='point-cloud', style={'height': '600px', 'width': '100%'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})

    # Callback to update point cloud visualization
    @app.callback(
        Output('point-cloud', 'figure'),
        [Input(f'slider-{i}', 'value') for i in range(10)]
    )
    def update_point_cloud(*sliders):
        # Update model parameters
        model.betas[:] = np.array(sliders)

        # Generate updated point cloud
        points = np.array(model.v_shaped)

        # Create 3D scatter plot
        trace = go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=2, color='lightblue', opacity=0.3)
        )
        
        # Adjusting the layout to set the camera view from slightly above the Z axis
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X', range=[-1.2, 1.2]),
                yaxis=dict(title='Y', range=[-1.2, 1.2]),
                zaxis=dict(title='Z', range=[-1.2, 1.2]),
                aspectmode='cube',  # Ensure equal scaling for all axes
                camera=dict(
                    eye=dict(x=0, y=-1, z=2)  # Adjust the camera to move slightly down along the Y-axis and maintain Z-axis view
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        return go.Figure(data=[trace], layout=layout)

    # Run the app
    app.run_server(debug=True)

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

def test_model(points, joints):
    points = np.array(points)
    joints = np.array(joints)

    joint_dict = {
        0: 'Pelvis', 1: 'Right Hip', 2: 'Left Hip', 3: 'Spine Lower', 4: 'Right Knee', 5: 'Left Knee',
        6: 'Spine Middle', 7: 'Right Ankle', 8: 'Left Ankle', 9: 'Chest', 10: 'Right Foot', 11: 'Left Foot',
        12: 'Neck', 13: 'Right Shoulder', 14: 'Left Shoulder', 15: 'Head', 16: 'Right Elbow', 17: 'Left Elbow',
        18: 'Right Wrist', 19: 'Left Wrist', 20: 'Right Hand', 21: 'Left Hand', 22: 'Right Finger Tip', 23: 'Left Finger Tip'
    }


    def calculate_length(joint_a, joint_b):
        return np.linalg.norm(joints[joint_a] - joints[joint_b])

    def calculate_circle_joint(points, joint_idx, y_tolerance=0.01):
        for tolerance in [y_tolerance, 0.02]:
            joint_position = joints[joint_idx]
            joint_points = points[np.abs(points[:, 1] - joint_position[1]) < tolerance]
            x = joint_points[:, 0]
            z = joint_points[:, 2]

            if len(x) < 3 or len(z) < 3:
                continue

            joint_points_2d = np.vstack((x, z)).T
            hull = ConvexHull(joint_points_2d)
            perimeter = 0
            for simplex in hull.simplices:
                point1 = joint_points_2d[simplex[0]]
                point2 = joint_points_2d[simplex[1]]
                perimeter += np.linalg.norm(point1 - point2)

            return perimeter
        return None


    def calculate_stature():
        return np.max(points[:, 1]) - np.min(points[:, 1])

    def calculate_acrom_radiale_length():
        return (calculate_length(16, 18) + calculate_length(17, 19)) / 2

    def calculate_cervicale_height():
        return joints[12][1] - np.min(points[:, 1])

    def calculate_crotch_height():
        return joints[2][1] - np.min(points[:, 1])

    def calculate_radiale_stylion_length():
        return calculate_length(18, 20)

    def calculate_waist_height_omphalion():
        return joints[3][1] - np.min(points[:, 1])

    def calculate_knee_height_sitting():
        return joints[5][1] - np.min(points[:, 1])

    def calculate_biacromial_breadth():
        return calculate_length(16, 17)

    def calculate_waist_circumference():
        return calculate_circle_joint(points, 3)

    def calculate_buttock_circumference():
        return calculate_circle_joint(points, 0)

    def calculate_chest_circumference():
        return calculate_circle_joint(points, 9)

    stature = calculate_stature()
    acrom_radiale_length = calculate_acrom_radiale_length()
    cervicale_height = calculate_cervicale_height()
    crotch_height = calculate_crotch_height()
    radiale_stylion_length = calculate_radiale_stylion_length()
    waist_height_omphalion = calculate_waist_height_omphalion()
    knee_height_sitting = calculate_knee_height_sitting()
    biacromial_breadth = calculate_biacromial_breadth()
    waist_circumference = calculate_waist_circumference()
    buttock_circumference = calculate_buttock_circumference()
    chest_circumference = calculate_chest_circumference()

    if waist_circumference is None or buttock_circumference is None or chest_circumference is None:
        print("Can not solve the circu")
        return False

    ansur_data = {
        'Stature': {'5th': 1528 / 1000, '95th': 1738 / 1000},
        'Acrom-Radiale Length': {'5th': 285 / 1000, '95th': 340 / 1000},
        'Cervicale Height': {'5th': 1305 / 1000, '95th': 1508 / 1000},
        'Crotch Height': {'5th': 702 / 1000, '95th': 845 / 1000},
        'Radiale-Stylion Length': {'5th': 220 / 1000, '95th': 269 / 1000},
        'Waist Height Omphalion': {'5th': 905 / 1000, '95th': 1064 / 1000},
        'Knee Height Sitting': {'5th': 474 / 1000, '95th': 561 / 1000},
        'Biacromial Breadth': {'5th': 333 / 1000, '95th': 390 / 1000},
        'Waist Circumference': {'5th': 675 / 1000, '95th': 946 / 1000},
        'Buttock Circumference': {'5th': 873 / 1000, '95th': 1180 / 1000},
        'Chest Circumference': {'5th': 814 / 1000, '95th': 1022 / 1000}
    }

    measurements = {
        'Stature': stature,
        # 'Acrom-Radiale Length': acrom_radiale_length,
        'Cervicale Height': cervicale_height,
        'Crotch Height': crotch_height,
        'Radiale-Stylion Length': radiale_stylion_length,
        'Waist Height Omphalion': waist_height_omphalion,
        # 'Knee Height Sitting': knee_height_sitting,
        'Biacromial Breadth': biacromial_breadth,
        'Waist Circumference': waist_circumference,
        'Buttock Circumference': buttock_circumference,
        'Chest Circumference': chest_circumference
    }

    for key, value in measurements.items():
        if not (ansur_data[key]['5th'] <= value <= ansur_data[key]['95th']):
            print(f"{key} : {value} is not in ({ansur_data[key]['5th']} - {ansur_data[key]['95th']})")
            return False

    return True
def optimal_index(model, max_iterations=5000, step=0.002):
    index_ranges = {i: (0.0, 0.0) for i in range(10)}

    for dim in range(10):
        lower_limit = 0.0
        upper_limit = 0.0
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            lower_limit -= step

            samples = np.zeros(10)
            samples[dim] = lower_limit

            model.betas[:] = samples
            points = np.array(model.v_shaped)
            joints = np.array(model.J)

            if test_model(points, joints):

                index_ranges[dim] = (lower_limit, index_ranges[dim][1])
            else:
                lower_limit += step
                break

        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            upper_limit += step

            samples = np.zeros(10)
            samples[dim] = upper_limit

            model.betas[:] = samples
            points = np.array(model.v_shaped)
            joints = np.array(model.J)

            if test_model(points, joints):
                index_ranges[dim] = (index_ranges[dim][0], upper_limit)
            else:
                upper_limit -= step 
                break

    index_ranges = {dim: (round(bounds[0], 3), round(bounds[1], 3)) for dim, bounds in index_ranges.items()}

    return index_ranges
def generate_and_plot_skin_points_with_colors(model, index_ranges, num_samples=200):
    """
    Generate skin points based on the input index_ranges, test them with test_model,
    and plot the results in a 10x10 grid. Failed samples are marked in red, and successful samples in green.
    """
    skin_points = []
    labels = []

    # Generate samples
    for _ in range(num_samples):
        samples = np.zeros(len(index_ranges))
        for dim, (lower, upper) in index_ranges.items():
            mu = (lower + upper) / 2
            sigma = (upper - lower) / 6
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            samples[dim] = truncnorm.rvs(a, b, loc=mu, scale=sigma)

        model.betas[:] = samples
        points = np.array(model.v_shaped)
        joints = np.array(model.J)
        skin_points.append(points)

        # Test the sample and mark if it fails
        if not test_model(points, joints):
            labels.append("Failed")
        else:
            labels.append("Passed")

    skin_points = np.array(skin_points)

    # Plot the skin points in a 10x10 grid
    fig, axes = plt.subplots(10, 20, figsize=(50, 25))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(skin_points):
            skin = skin_points[i]
            ax.scatter(skin[:, 0], skin[:, 1], c='blue', s=1, alpha=0.5)
            title_color = 'red' if labels[i] == "Failed" else 'green'
            ax.set_title(labels[i], fontsize=6, color=title_color)
        else:
            ax.axis("off")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('index/4_2.jpg')
def get_point(model, index_ranges, num_samples, save=True):
    skin_list = []
    joint_list = []

    while len(skin_list) < num_samples:
        samples = np.zeros(len(index_ranges))
        for dim, (lower, upper) in index_ranges.items():
            mu = (lower + upper) / 2
            sigma = (upper - lower) / 6
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            samples[dim] = truncnorm.rvs(a, b, loc=mu, scale=sigma)

        model.betas[:] = samples
        points = np.array(model.v_shaped)
        joints = np.array(model.J)

        skin_list.append(points)
        joint_list.append(joints)

    skin = np.array(skin_list)
    joint = np.array(joint_list)

    if save:
        np.save("smpl_skin.npy", skin)
        np.save("smpl_joint.npy", joint)

    print(f"{len(skin)}  (6890, 3)")
    return skin, joint
def get_points(model, index_ranges, num_subjects=50, num_poses=200, num_test=20,save=True):
    '''
    result = {
    "Test": {
        "W": np.ndarray,       # 真实W,形状为 (N,K)
        "P": np.ndarray,     # 真实P,形状为 (N,3,(K-1)*4+1)
        "theta": np.ndarray                # 真实theta,形状为 (num_subjects, num_poses, K * 3)
    },
    "sub0": {
        "skin_template": np.ndarray,       # 模板皮肤顶点，形状为 (N, 3)
        "joint_template": np.ndarray,     # 模板关节点坐标，形状为 (K, 3)
        "skin": np.ndarray                # 随机 pose 的皮肤顶点，形状为 (num_poses, N, 3)
        "b2":np.ndarray                     # subj的beta2, 常数
    },
    "sub1": {
        "skin_template": np.ndarray,       # 模板皮肤顶点，形状为 (N, 3)
        "joint_template": np.ndarray,     # 模板关节点坐标，形状为 (K, 3)
        "skin": np.ndarray                # 随机 pose 的皮肤顶点，形状为 (num_poses, N, 3)
        "b2":np.ndarray                     # subj的beta2, 常数
    },
    ...
    "sub49": {
        "skin_template": np.ndarray,       # 模板皮肤顶点，形状为 (N, 3)
        "joint_template": np.ndarray,     # 模板关节点坐标，形状为 (K, 3)
        "skin": np.ndarray                # 随机 pose 的皮肤顶点，形状为 (num_poses, N, 3)
        "b2":np.ndarray                     # subj的beta2, 常数
    }
    "Test_sub0": {
        "skin_template": np.ndarray,       # 模板皮肤顶点，形状为 (N, 3)
        "joint_template": np.ndarray,     # 模板关节点坐标，形状为 (K, 3)
        "skin": np.ndarray                # 随机 pose 的皮肤顶点，形状为 (num_poses, N, 3)
        "b2":np.ndarray                     # subj的beta2, 常数
    },
    "Test_sub1": {
        "skin_template": np.ndarray,       # 模板皮肤顶点，形状为 (N, 3)
        "joint_template": np.ndarray,     # 模板关节点坐标，形状为 (K, 3)
        "skin": np.ndarray                # 随机 pose 的皮肤顶点，形状为 (num_poses, N, 3)
        "b2":np.ndarray                     # subj的beta2, 常数
    },
    ...
    "Test_sub49": {
        "skin_template": np.ndarray,       # 模板皮肤顶点，形状为 (N, 3)
        "joint_template": np.ndarray,     # 模板关节点坐标，形状为 (K, 3)
        "skin": np.ndarray                # 随机 pose 的皮肤顶点，形状为 (num_poses, N, 3)
        "b2":np.ndarray                     # subj的beta2, 常数
    }
    }
    '''
    result = {}
    j_reg = np.load("J_trained.npy")  # 加载关节点回归矩阵
    assert j_reg.shape[1] == 6890, "J_trained.npy 应为 (27, 6890) 矩阵"
    theta=np.zeros((num_subjects,num_poses,24*3))
    W=np.array(model.weights)
    P=np.array(model.posedirs)
    # Train set
    for sub_id in range(num_subjects):
        # 随机生成 beta 参数
        samples = np.zeros(len(index_ranges))
        for dim, (lower, upper) in index_ranges.items():
            mu = (lower + upper) / 2
            sigma = (upper - lower) / 6
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            samples[dim] = truncnorm.rvs(a, b, loc=mu, scale=sigma)

        model.betas[:] = samples

        b2=samples[1]
        skin_template = np.array(model.v_shaped)
        joint_template=np.array(model.J)
        # joint_template = j_reg @ skin_template
        skin_list = []

        for pose_id in range(num_poses):
            pose=np.random.uniform(-0.3, 0.3, size=72)
            model.pose[:] = pose
            theta[sub_id][pose_id][:]=pose
            points = np.array(model)
            skin_list.append(points)

        # 保存该 subject 的数据
        result[f"sub{sub_id}"] = {
            "skin_template": skin_template,
            "joint_template": joint_template,
            "b2": np.array(b2),
            "skin": np.array(skin_list)  # (201, 6890, 3)
        }
    # TEST set
    for sub_id in range(int(num_subjects/10)):
        # 随机生成 beta 参数
        samples = np.zeros(len(index_ranges))
        for dim, (lower, upper) in index_ranges.items():
            mu = (lower + upper) / 2
            sigma = (upper - lower) / 6
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            samples[dim] = truncnorm.rvs(a, b, loc=mu, scale=sigma)

        model.betas[:] = samples

        b2=samples[1]
        skin_template = np.array(model.v_shaped)
        joint_template=np.array(model.J)
        # joint_template = j_reg @ skin_template
        skin_list = []

        for pose_id in range(int(num_poses/10)):
            pose=np.random.uniform(-0.3, 0.3, size=72)
            model.pose[:] = pose
            theta[sub_id][pose_id][:]=pose
            points = np.array(model)
            skin_list.append(points)

        # 保存该 subject 的数据
        result[f"Valid{sub_id}"] = {
            "skin_template": skin_template,
            "joint_template": joint_template,
            "b2": np.array(b2),
            "skin": np.array(skin_list)  # (201, 6890, 3)
        }
    result["Test"] ={
            "W": W,
            "P": P,
            "theta": theta,

        }
    # ipdb.set_trace()
    if save:
        np.save("W_train_dataset.npy", result)
    return result


if __name__ == "__main__":
    # model = STAR(gender='female', num_betas=10)
    # plot_human_structure(model.v_shaped,model.J)
    data = np.load("C:\\Users\\25983\\Documents\\Medical_LXZ\\smpl\\W_New.npy", allow_pickle=True).item()
    plot_human_structure(data['sub0']['skin_template'],data["sub0"]['joint_template'])
    # dic={0: (-4, 4),
    #     1: (-2, 2),
    #     2: (-4, 4),
    #     3: (-4, 4),
    #     4: (-4, 4),
    #     5: (-4, 4),
    #     6: (-4, 4),
    #     7: (-4, 4),
    #     8: (-4, 4),
    #     9: (-4, 4)}
    # subjects_skin = get_points(model, dic)



    # draw structure
    # plot_human_structure(model.v_shaped,np.load("J_trained.npy") @ np.array(model.v_shaped))


    # # generate model with definate range
    
    # generate_and_plot_skin_points_with_colors(model,dic)

    # # GUI
    # interactive_model_visualization(model)

    



