import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
# import ipdb
import matplotlib.pyplot as plt
import torch.nn.functional as F
def compute_A(T, J):
    N = T.shape[0]  # Number of vertices
    K = J.shape[0]  # Number of joints
    T_expanded = T.unsqueeze(0).expand(K, N, 3)  # (K, N, 3)
    J_expanded = J.unsqueeze(1).expand(K, N, 3)  # (K, N, 3)
    distances = torch.norm(T_expanded - J_expanded, dim=2) 
    A = 1.0 / (distances + 1e-8)  
    A_min = A.min(dim=1, keepdim=True)[0]
    A_max = A.max(dim=1, keepdim=True)[0]
    A_normalized = (A - A_min) / (A_max - A_min + 1e-8)  
    A_activated = torch.relu(A_normalized)

    return A_activated

def annealing_factor(epoch, init_factor, decay_rate=5e-5):
    return init_factor * np.exp(-decay_rate * epoch)

def cosine_decay_lr(epoch, max_lr, total_epochs):
    return max_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


def compute_wi(T, J, num=4):
    N, K = T.shape[0], J.shape[0]
    distances = torch.norm(T.unsqueeze(1) - J.unsqueeze(0), dim=2)  # (N, K)
    nearest_indices = torch.argsort(distances, dim=1)[:, :num]  # (N, num)
    nearest_distances = distances.gather(1, nearest_indices)  # (N, num)
    inverse_distances = 1.0 / (nearest_distances + 1e-8)  # (N, num)
    normalized_weights = inverse_distances / inverse_distances.sum(dim=1, keepdim=True)  # (N, num)
    W_i = torch.zeros((N, K), dtype=torch.float32, device=T.device)  # (N, K)
    W_i.scatter_(1, nearest_indices, normalized_weights)
    
    return W_i


def theta_to_q(theta):

    K = theta.size(0)
    angle = torch.norm(theta, dim=1, keepdim=True)  # Norm of axis-angle vector (K, 1)
    angle = torch.clamp(angle, min=1e-8)  # Avoid division by zero for small angles
    axis = theta / angle  # Normalize axis (K, 3)

    half_angle = angle / 2
    q = torch.zeros(K, 4, device=theta.device)  # Quaternion (w, x, y, z)
    q[:, 0] = torch.cos(half_angle.squeeze())  # w = cos(half_angle)
    q[:, 1:] = axis * torch.sin(half_angle)  # (x, y, z) = sin(half_angle) * normalized_axis

    return q


def save_model_params(model, save_path="WAK.npy"):

    W_clone = model.W_prime.clone().detach().cpu().numpy()
    A_clone = [a.clone().detach().cpu().numpy() for a in model.A]
    K_clone = [k.clone().detach().cpu().numpy() for k in model.K]

    param_dict = {
        "W": W_clone,
        "A": A_clone,
        "K": K_clone
    }

    np.save(save_path, param_dict)

def update_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class SMPLModel(nn.Module):
    def __init__(self, W_i, A_init, K, neighbor_list, N):
        super(SMPLModel, self).__init__()
        self.W_i = W_i.clone().detach()  # Precomputed initial blend weights (N, K)
        self.W_prime = nn.Parameter(W_i.clone().cuda())  # Blend weights (N, K)
        self.A = nn.ParameterList([nn.Parameter(A_init[i].clone().cuda()) for i in range(K-1)])  # Activation weights
        self.K = nn.ParameterList([nn.Parameter(torch.normal(0, 0.5, size=(len(neighbor_list[i]) * 4 + 1, 3 * N)).cuda()) for i in range(K-1)])
        self.neighbor_list = neighbor_list
        self.K_tree=np.array([[4294967295,          0,          0,          0,          1,
                 2,          3,          4,          5,          6,
                 7,          8,          9,          9,          9,
                12,         13,         14,         16,         17,
                18,         19,         20,         21],
       [         0,          1,          2,          3,          4,
                 5,          6,          7,          8,          9,
                10,         11,         12,         13,         14,
                15,         16,         17,         18,         19,
                20,         21,         22,         23]])
        
        
        self.debug = []#!!!!!!!!!!
        
    def rodrigues(self,pose):

        angle = torch.norm(pose, dim=1, keepdim=True)  # (K, 1)
        angle = torch.clamp(angle, min=1e-8)  # Avoid division by zero
        axis = pose / angle  # Normalize axis (K, 3)

        cos = torch.cos(angle).unsqueeze(-1)  # (K, 1, 1)
        sin = torch.sin(angle).unsqueeze(-1)  # (K, 1, 1)

        # Skew-symmetric cross-product matrix
        skew = torch.zeros(pose.shape[0], 3, 3, device=pose.device)
        skew[:, 0, 1] = -axis[:, 2]
        skew[:, 0, 2] = axis[:, 1]
        skew[:, 1, 0] = axis[:, 2]
        skew[:, 1, 2] = -axis[:, 0]
        skew[:, 2, 0] = -axis[:, 1]
        skew[:, 2, 1] = axis[:, 0]

        # Rodrigues' rotation formula
        outer = torch.bmm(axis.unsqueeze(2), axis.unsqueeze(1))  # (K, 3, 3)
        eye = torch.eye(3, device=pose.device).unsqueeze(0).repeat(pose.shape[0], 1, 1)  # (K, 3, 3)
        R = cos * eye + (1 - cos) * outer + sin * skew  # (K, 3, 3)

        return R


    # def compute_G(self,pose, J):

    #     K = J.shape[0]
    #     pose=pose.view(K, 3)
    #     R = self.rodrigues(pose)  # (K, 3, 3)

    #     # Initialize local transformation matrices
    #     G_local = torch.eye(4, device=pose.device).unsqueeze(0).repeat(K, 1, 1)  # (K, 4, 4)
    #     G_local[:, :3, :3] = R
    #     G_local[:, :3, 3] = J

    #     # Recursive computation of global transformations
    #     G_global = [G_local[0]]
    #     for i in range(1, K):
    #         parent_idx = self.K_tree[0, i]
    #         if parent_idx == 4294967295:  # Root joint (no parent)
    #             G_global.append(G_local[i])
    #         else:
    #             G_global.append(torch.matmul(G_global[parent_idx], G_local[i]))
    #     G_global = torch.stack(G_global, dim=0)  # (K, 4, 4)

    #     # Center transformations around joints
    #     J_h = torch.cat([J, torch.zeros((K, 1), device=J.device)], dim=1).unsqueeze(-1)  # (K, 4, 1)
    #     G_centered = G_global - torch.matmul(G_global, J_h)

    #     return G_centered
    def compute_G(self, pose, J):
        """
        Fixes:
          1. Use relative translations: G_local[i][:3, 3] = J[i] - J[parent_idx]
          2. Subtract transformed joint offset for each joint separately,
             rather than a single 'G_global - G_global@J_h'.
        """
        K = J.shape[0]
        pose = pose.view(K, 3)
        R = self.rodrigues(pose)  # (K, 3, 3)
    
        # ------------------------
        # 1) Construct local transforms with RELATIVE translation
        # ------------------------
        G_local = torch.eye(4, device=pose.device).unsqueeze(0).repeat(K, 1, 1)  # (K, 4, 4)
        G_local[:, :3, :3] = R
    
        # Root translation = J[0], else relative to parent
        for i in range(K):
            parent_idx = self.K_tree[0, i]
            if parent_idx == 4294967295:
                # Root joint
                G_local[i, :3, 3] = J[i]
            else:
                G_local[i, :3, 3] = J[i] - J[parent_idx]
    
        # ------------------------
        # 2) Chain up the transforms (parent -> child)
        # ------------------------
        G_global = [G_local[0]]  # root
        for i in range(1, K):
            parent_idx = self.K_tree[0, i]
            if parent_idx == 4294967295:
                # Root again (just in case)
                G_global.append(G_local[i])
            else:
                G_global.append(torch.matmul(G_global[parent_idx], G_local[i]))
        G_global = torch.stack(G_global, dim=0)  # (K, 4, 4)
    
        # ------------------------
        # 3) STAR-like “pack” step: subtract out the child’s own offset
        #    from each global transform
        # ------------------------
        # Each joint's transform is adjusted so that the joint itself
        # ends up at the origin of that joint's local coordinates.
        # The original STAR code does:
        #   results[i] - pack( results[i].dot([J[i],1]) )
        # We'll do something analogous in PyTorch:
        for i in range(K):
            # Transform J[i] by G_global[i], then subtract from the translation block
            joint_h = torch.cat([J[i], J[i].new_tensor([0.0])])  # (4,) -> homogeneous
            # or: joint_h = torch.cat([J[i], torch.tensor([1.0], device=J.device)])
            offset = torch.matmul(G_global[i], joint_h)  # (4,)
            # Now subtract offset from the last column
            G_global[i, :3, 3] -= offset[:3]
    
        return G_global

    def compute_blendP(self, T, theta, beta_2,Preg,N,K):
        # Get q - q*
        q_origin = theta_to_q(theta.view(Preg * K, 3)).view(Preg, K, 4)
        q_star = theta_to_q(torch.zeros_like(theta.view(Preg * K, 3))).view(Preg, K, 4)
        q= q_origin-q_star
        T_p = torch.zeros_like(T)  # Initialize blend contribution (Preg, N, 3)
        # reshape to Preg, K, 4*K +1
        q_ne=torch.zeros(Preg, K, 4*K +1).cuda()
        for k in range(K):
            Nk = self.neighbor_list[k]
            num_Nk = len(Nk)
            selected_vectors = q[:, Nk, :].reshape(Preg, -1)
            padded_vector = torch.cat(
                [selected_vectors, torch.full((Preg, 1), beta_2, device=selected_vectors.device)],
                    dim=1
                )
            q_ne[:, k, :4*num_Nk+1] = padded_vector
        
        # calculate the Tp of each joints
        for j in range(K-1):
            K_j = self.K[j] 
            lenK,_=K_j.shape
            corrected_contribution = torch.matmul(q_ne[:, j+1, :lenK], K_j)  # (Preg, 3N)

            A_j = self.A[j]  # Activation weights (N)

            A_j_activated = torch.relu(A_j)  # Ensure A_j >= 0
            
            A_j_expanded = A_j_activated.unsqueeze(0).expand(Preg, -1)  # Expand A_j to (Preg, N)

            weighted_contribution = corrected_contribution.view(Preg, N, 3) * A_j_expanded.unsqueeze(-1)  # (Preg, N, 3)

            T_p += weighted_contribution
            
        return T_p

    def compute_vertex_transform(self, T, W, G, T_p):

            N, K = W.shape

            T_corrected = T + T_p  # (N, 3)

            T_corrected_h = torch.cat([T_corrected, torch.ones((N, 1), device=T.device)], dim=1)  # (N, 4)
            transformed = torch.einsum('kij,nj->nki', G, T_corrected_h)[:, :, :3]  # (N, K, 3)
            transformed_vertices = torch.einsum('nk,nkj->nj', W, transformed)  # (N, 3)

            return transformed_vertices



    def forward(self, V, T, J, theta, beta_2,Tp=None,epoch=1):

        Preg, N, _ = V.shape
        K = J.shape[1]

        # Convert theta to quaternion and compute q_neighbors and q_star_neighbors
        # W_normalized = F.softmax(self.W_prime, dim=1)
        W_normalized = self.W_prime
        G_prime = torch.stack([self.compute_G(theta[p], J[p]) for p in range(Preg)])  # (Preg, K, 4, 4)

        T_p = self.compute_blendP(T, theta, beta_2,Preg,N,K)  # Pose blend correction

        transformed_vertices = torch.stack([
            self.compute_vertex_transform(T[p], W_normalized, G_prime[p], T_p[p]) for p in range(Preg)
        ])
        
        p=0
        self.debug.append( (transformed_vertices[p],T[p], W_normalized, G_prime[p], T_p[p]) )
        
        
        # Compute losses
        E_D = torch.sum((V - transformed_vertices) ** 2)
        gamma_Ed = annealing_factor(epoch,0.01)

        E_Wi = torch.norm(W_normalized - self.W_i) ** 2
        gamma_Ewi = annealing_factor(epoch,0.01)

        E_W = torch.norm(W_normalized, p=1)
        gamma_Ew = annealing_factor(epoch,1.)

        E_A = sum(torch.norm(A_j, p=1) for A_j in self.A)
        gamma_Ea = annealing_factor(epoch,1.)

        E_K = sum(torch.norm(K_j) for K_j in self.K)
        gamma_Ek = annealing_factor(epoch,1.)

        E = gamma_Ed* E_D + gamma_Ewi*E_Wi + gamma_Ew*E_W + gamma_Ea*E_A + gamma_Ek*E_K  

        return E, transformed_vertices

# load data
data = np.load("W.npy", allow_pickle=True).item()
# data_dir = 'result/thetaisright2'
# if not os.path.exists(data_dir): 
#     os.makedirs(data_dir)
all_keys = data.keys()

# load id
subject_ids = [key for key in all_keys if key.startswith("sub") and not key.startswith("Test")]
subject_id_without_shuffle= [key for key in all_keys if key.startswith("sub") and not key.startswith("Test")]
test_ids=[key for key in all_keys if key.startswith("Val")]
random.shuffle(subject_ids) 

# initial
Preg = 200  # Number of registrations per subject
Psub = len(subject_ids)  # Number of subjects
K = 24  # Number of joints
N = 6890  # Number of vertices
neighbor_list = [
    [0,1,2,3], # 0
    [0,1,4], # 1
    [0,2,5], # 2
    [0,3,6],
    [1,4,7],
    [2,5,8], # 5
    [3,6,9],
    [4,7,10], # 7
    [5,8,11],
    [6,9], # 9
    [7,10],
    [8,11], # 11
    [12,13,14,15],
    [12,13,16], # 13
    [12,14,17],
    [12,15], # 15
    [13,16,18],
    [14,17,19], # 17
    [16,18,20],
    [17,19,21], # 19
    [18,20,22],
    [19,21,23], # 21
    [20,22],
    [21,23], # 23
]
real_W = data["Test"]["W"]
real_theta = data["Test"]["theta"]
real_theta=torch.from_numpy(real_theta).cuda()

# Compute shared W_i, AB
first_subject = subject_ids[0]
T = torch.tensor(data[first_subject]['skin_template'], dtype=torch.float32).cuda()
J = torch.tensor(data[first_subject]['joint_template'], dtype=torch.float32).cuda()
W_i = compute_wi(T, J)  # Compute initial blend weights
A_init = compute_A(T, J)  # Compute A

# initial model and optimizer
# model = SMPLModel(W_i,A_init[1:], K, neighbor_list, N).cuda()
model = SMPLModel( torch.tensor(real_W, dtype=torch.float32).cuda(),A_init[1:], K, neighbor_list, N).cuda()
# all_theta = nn.Parameter((torch.randn(Psub, Preg, K * 3, dtype=torch.float32) * 0.1).cuda())
# optimizer_theta = optim.Adam([all_theta], lr=1e-3)

# max_lr_W = 1e-3  # \mathbf{W} 的最大学习率
# max_lr_AK = 1e-3  # \mathbf{A} 和 \mathbf{K} 的最大学习率

total_epochs = 100000  # 总训练 epoch 数

# optimizer_W = optim.Adam([model.W_prime], lr=1e-3)
# optimizer_AK = optim.Adam([*model.A.parameters(), *model.K.parameters()], lr=1e-3)

# save
best_loss=np.inf
best_epoch= 0

# Training loop
for epoch in range(total_epochs):
    # Randomly select a subject
    def sample_minibatch(subject_data, theta, batch_size=100):

        Preg = subject_data['skin'].shape[0]  # 每个 subject 的注册数量
        indices = torch.randperm(Preg)[:batch_size]  # 随机选取 batch_size 个索引

        V_batch = torch.tensor(subject_data['skin'][indices], dtype=torch.float32).cuda()  # (batch_size, N, 3)
        T_batch = torch.tensor(subject_data['skin_template'], dtype=torch.float32).cuda().unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, N, 3)
        J_batch = torch.tensor(subject_data['joint_template'], dtype=torch.float32).cuda().unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, K, 3)
        T_p_batch= torch.tensor(subject_data['T_p'][indices], dtype=torch.float32).cuda()  # (batch_size, N, 3)
        theta_batch = theta[indices]  # (batch_size, K, 3)

        return V_batch, T_batch, J_batch,T_p_batch,theta_batch

    subject_id = random.choice(subject_ids) 
    subject_index = subject_ids.index(subject_id)
    subject_data = data[subject_id]
    theta = real_theta[subject_index]

    V,T,J,Tp,theta=sample_minibatch(subject_data,theta)

    beta_2 = torch.tensor(subject_data['b2'], dtype=torch.float32).cuda()
    

    # lr_W = cosine_decay_lr(epoch, max_lr_W, total_epochs)
    # lr_AK = cosine_decay_lr(epoch, max_lr_AK, total_epochs)
    # update_lr(optimizer_W, lr_W)
    # update_lr(optimizer_AK, lr_AK)

    # optimizer_theta.zero_grad()
    # loss_theta , _= model(V, T.unsqueeze(0).repeat(Preg, 1, 1), J.unsqueeze(0).repeat(Preg, 1, 1), theta, beta_2)
    # loss_theta.backward()
    # optimizer_theta.step()

    # optimizer_W.zero_grad()
    # optimizer_AK.zero_grad()

    loss , _= model(V, T, J, theta, beta_2,Tp=Tp,epoch=epoch)
    loss.backward()

    # optimizer_W.step()
    # optimizer_AK.step()

    model.eval()

    if epoch % 40 == 0:
        # with torch.no_grad():
            # save_model_params(copy.deepcopy(model),save_path=f'{data_dir}/WAK.npz')
        num_plots = 2
        rows = 4 if num_plots >3 else 1
        clms = num_plots//4+1 if num_plots>3 else num_plots
        fig, axes = plt.subplots( clms, rows, figsize=(rows*4, clms*4))
        axes = axes.flatten()
        for i, test_subject_id in enumerate(subject_id_without_shuffle[:num_plots]):
            test_data = data[test_subject_id]
            T_test = torch.tensor(test_data['skin_template'], dtype=torch.float32).cuda().view(1,6890,3)
            J_test = torch.tensor(test_data['joint_template'], dtype=torch.float32).cuda().view(1,K,3)
            V_test = torch.tensor(test_data['skin'][0:1], dtype=torch.float32).cuda().view(1,6890,3)  # First register only
            beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()

            theta_test = real_theta[subject_id_without_shuffle.index(test_subject_id), 0:1]  # Corresponding theta
            with torch.no_grad():
                _,predicted_V = model(V_test, T_test, J_test, theta_test, beta_2_test)
                error = torch.mean((V_test - predicted_V) ** 2).item()
                ax = axes[i]
                ax.scatter(V_test[0, :, 0].cpu(), V_test[0, :, 1].cpu(), s=1, label="Ground Truth")
                ax.scatter(predicted_V[0, :, 0].cpu(), predicted_V[0, :, 1].cpu(), s=1, label="Prediction")
                ax.set_title(f"Subject {test_subject_id} Error: {error:.4f}")
                ax.legend()
        plt.tight_layout()
        # plt.savefig(f'{data_dir}/{epoch}.png')

    model.train()
    print(f"Epoch {epoch + 1}")
    
    
    
    

'''
T1, W1, G1, Tp1=model.debug[0][1:]
W = torch.tensor(real_W, dtype=torch.float32).cuda()
G_prime = torch.stack([model.compute_G(theta[0].reshape(72,)*0, J_test[0])])  # (Preg, K, 4, 4)
with torch.no_grad(): vert = model.compute_vertex_transform(T1, W1, G1, Tp1)

plt.figure(figsize=(10,10))
plt.scatter(T1[:, 0].cpu(), T1[:, 1].cpu(), s=1, label="Tpose template")
plt.scatter(vert[:, 0].cpu(), vert[:, 1].cpu(), s=1, label="forwarded with real W and no P blend")
plt.legend()
'''

# (transformed_vertices[p],T[p], W_normalized, G_prime[p], T_p[p]) 
# T1, W1, G1, Tp1=model.debug[0][1:]
'''
with torch.no_grad(): t_theta = copy.deepcopy(theta_test[0])
t_theta[:] = 0
t_theta[5] = 1.57
G_prime = torch.stack([model.compute_G(t_theta, J_test[0])])  # (Preg, K, 4, 4)
template = T_test[0]

W = torch.tensor(real_W, dtype=torch.float32).cuda()
# with torch.no_grad(): W = model.W_prime.detach()

G = G_prime[0]

pose_blend = torch.tensor(np.zeros((6890,3)), dtype=torch.float32).cuda()
# beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()
# pose_blend =model.compute_blendP(T_test, theta_test[0], beta_2_test,1,N,K)[0]

with torch.no_grad(): vert = model.compute_vertex_transform(template,W,G,pose_blend)
plt.figure(figsize=(10,10))
plt.scatter(template[:, 0].cpu(), template[:, 1].cpu(), s=1, label="Tpose template")
plt.scatter(vert[:, 0].cpu(), vert[:, 1].cpu(), s=1, label="forwarded with real W and no P blend")
plt.legend()

'''
    
'''


import open3d as o3d
for i, test_subject_id in enumerate(subject_id_without_shuffle[:1]):
    test_data = data[test_subject_id]
    T_test = torch.tensor(test_data['skin_template'], dtype=torch.float32).cuda().view(1,6890,3)
    J_test = torch.tensor(test_data['joint_template'], dtype=torch.float32).cuda().view(1,K,3)
    V_test = torch.tensor(test_data['skin'][0:1], dtype=torch.float32).cuda().view(1,6890,3)  # First register only
    beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()

    theta_test = real_theta[subject_ids.index(test_subject_id), 0:1]  # Corresponding theta
    with torch.no_grad():
        _,predicted_V = model(V_test, T_test, J_test, theta_test, beta_2_test)
    # 1. Convert Torch tensors to Open3D point clouds
    def torch_to_open3d(tensor):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(tensor)
        return point_cloud
    
    
    # Convert tensors to numpy
    with torch.no_grad():source_pc = predicted_V.cpu().numpy()
    target_pc = T_test.cpu().numpy()
    
    # Ensure shape is correct (n x 3)
    source_pc = source_pc.reshape(-1, 3)
    target_pc = target_pc.reshape(-1, 3)
    
    # Convert numpy arrays to Open3D point clouds
    source_pc = torch_to_open3d(source_pc)
    target_pc = torch_to_open3d(target_pc)
    
    # 2. Configure ICP parameters
    distance_threshold = 0.05  # Adjust based on application
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=2000,  # Max iterations
        relative_fitness=1e-6,  # Convergence tolerance
        relative_rmse=1e-6
    )
    
    # 3. Perform ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        source=source_pc,
        target=target_pc,
        max_correspondence_distance=distance_threshold,
        init=np.identity(4),  # Initialize with identity matrix
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=icp_criteria
    )
    
    # 4. Retrieve the transformation matrix
    transformation_matrix = icp_result.transformation
    fitness = icp_result.fitness
    rmse = icp_result.inlier_rmse
    
    print("Transformation Matrix:\n", transformation_matrix)
    print(f"ICP Fitness: {fitness}, RMSE: {rmse}")
    
    # Optional: Transform the source point cloud using the resulting transformation
    transformed_source = source_pc.transform(transformation_matrix)
    source_pc.paint_uniform_color([1, 0, 0])  # Red for source point cloud
    target_pc.paint_uniform_color([0, 1, 0])  # Green for target point cloud
    # Optional: Visualize the alignment
    o3d.visualization.draw_geometries([target_pc, transformed_source])
    
    
    
'''

    
    
    
    