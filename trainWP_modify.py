import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import ipdb
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
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

def annealing_factor(epoch, init_factor, decay_rate=8e-3):
    print(np.exp(-decay_rate * epoch))
    return init_factor * np.exp(-decay_rate * epoch)

def cosine_decay_lr(epoch, max_lr, total_epochs):
    return max_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


def compute_wi(T, J, num=10):
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

def axis2quat(p):

    # Compute the rotation angle (norm of axis-angle vector)
    angle = torch.sqrt(torch.clamp(torch.sum(p**2, dim=1), min=1e-16))  # (N,)
    
    # Normalize the axis vector
    norm_p = p / angle.unsqueeze(-1)  # (N, 3)
    
    # Compute the sine and cosine of half the angle
    cos_angle = torch.cos(angle / 2)  # (N,)
    sin_angle = torch.sin(angle / 2)  # (N,)
    
    # Compute quaternion components
    qx = norm_p[:, 0] * sin_angle  # x component
    qy = norm_p[:, 1] * sin_angle  # y component
    qz = norm_p[:, 2] * sin_angle  # z component
    qw = cos_angle - 1             # w component (with custom offset)
    
    # Combine into a single tensor
    quat = torch.stack([qx, qy, qz, qw], dim=1)  # Shape (N, 4)
    return quat
def save_model_params(model, save_path="WAK.npy"):
   
    W_clone = model.W_prime.clone().detach()
    W_relu = torch.relu(W_clone)                     # (N, K)
    row_sums = W_relu.sum(dim=1, keepdim=True)            # (N, 1)
    W_change = (W_relu / (row_sums + 1e-8)).cpu().numpy()
    A_clone = [a.clone().detach().cpu().numpy() for a in model.A]
    K_clone = [k.clone().detach().cpu().numpy() for k in model.K]

    param_dict = {
        "W": W_change,
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
        # self.W_prime = nn.Parameter((torch.rand(W_i.shape, dtype=torch.float32) * 0.5).cuda())  # Blend weights (N, K)
        self.A = nn.ParameterList([nn.Parameter(A_init[i].clone().cuda()) for i in range(K-1)])  # Activation weights
        self.K = nn.ParameterList([nn.Parameter(torch.normal(0, 0.01, size=(len(neighbor_list[i]) * 4 + 1, 3 * N)).cuda()) for i in range(K-1)])
        self.neighbor_list = neighbor_list
        self.K_tree=np.array([[4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], 
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
        
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
        # Rodigues' rotation formula
        outer = torch.bmm(axis.unsqueeze(2), axis.unsqueeze(1))  # (K, 3, 3)
        eye = torch.eye(3, device=pose.device).unsqueeze(0).repeat(pose.shape[0], 1, 1)  # (K, 3, 3)
        R = cos * eye + (1 - cos) * outer + sin * skew  # (K, 3, 3)
        return R

    def compute_G(self, pose, J):
        K = J.shape[0]
        pose = pose.view(K, 3)
        R = self.rodrigues(pose)  # (K, 3, 3)
        G_local = torch.eye(4, device=pose.device).unsqueeze(0).repeat(K, 1, 1)  # (K, 4, 4)
        G_local[:, :3, :3] = R
        for i in range(K):
            parent_idx = self.K_tree[0, i]
            if parent_idx == 4294967295:
                # Root joint
                G_local[i, :3, 3] = J[i]
            else:
                G_local[i, :3, 3] = J[i] - J[parent_idx]
        G_global = [G_local[0]]  # root
        for i in range(1, K):
            parent_idx = self.K_tree[0, i]
            if parent_idx == 4294967295:
                G_global.append(G_local[i])
            else:
                G_global.append(torch.matmul(G_global[parent_idx], G_local[i]))
        G_global = torch.stack(G_global, dim=0)  # (K, 4, 4)
        for i in range(K):
            joint_h = torch.cat([J[i], J[i].new_tensor([0.0])])  # (4,) -> homogeneous
            offset = torch.matmul(G_global[i], joint_h)  # (4,)
            G_global[i, :3, 3] -= offset[:3]
    
        return G_global


    def compute_blendP(self, T, q, beta_2,Preg,N,K):
        # Get q - q*
        
    
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

    def compute_vertex_transform(self,W, G, T_p):

            N, K = W.shape
            T_corrected_h = torch.cat([T_p, torch.ones((N, 1), device=T.device)], dim=1)  # (N, 4)
            transformed = torch.einsum('kij,nj->nki', G, T_corrected_h)[:, :, :3]  # (N, K, 3)
            transformed_vertices = torch.einsum('nk,nkj->nj', W, transformed)  # (N, 3)

            return transformed_vertices



    def forward(self, V, T, J, T_p,theta, beta_2,epoch=1):

        Preg, N, _ = V.shape
        K = J.shape[1]

        # Convert theta to quaternion and compute q_neighbors and q_star_neighbors
        # W_normalized = F.softmax(self.W_prime, dim=1)

        W_relu = torch.relu(self.W_prime)                     # (N, K)
        row_sums = W_relu.sum(dim=1, keepdim=True)            # (N, 1)
        W_change = W_relu / (row_sums + 1e-8)             # (N, K)

        G_prime = torch.stack([self.compute_G(theta[p], J[p]) for p in range(Preg)])  # (Preg, K, 4, 4)
        T_p = torch.zeros_like(T)
        q = axis2quat(theta.view(Preg * K, 3)).view(Preg, K, 4)
        T_p = self.compute_blendP(T, q, beta_2,Preg,N,K) +T # Pose blend correction


        transformed_vertices = torch.stack([
            self.compute_vertex_transform(W_change, G_prime[p], T_p[p]) for p in range(Preg)
        ])
        
        # Compute losses
        E_D = torch.sum((V - transformed_vertices) ** 2)
        gamma_Ed = 1

        E_Wi = torch.norm(W_change - self.W_i) ** 2
        gamma_Ewi = annealing_factor(epoch,0.1,decay_rate=1e-1)
        
        E_W = torch.norm(W_change , p=1)
        gamma_Ew = annealing_factor(epoch,0.002)
        # ipdb.set_trace()
        E_A = sum(torch.norm(A_j, p=1) for A_j in self.A)
        gamma_Ea = annealing_factor(epoch,0.001)

        E_K = sum(torch.norm(K_j) for K_j in self.K)
        gamma_Ek = annealing_factor(epoch,0.1)
        # ipdb.set_trace()
        # E_row_sum = torch.sum(( torch.sum(W_change, dim=1) - 1) ** 2)  # Squared deviation from 1
        # gamma_Ers = annealing_factor(epoch, 0.5)
        print(f"E_D: {gamma_Ed * E_D:.4f}, E_Wi: {gamma_Ewi *E_Wi:.4f}, E_W: { gamma_Ew *E_W:.4f}, E_A: {gamma_Ea * E_A:.4f}, E_K: { gamma_Ek *E_K:.4f}")
        # Total loss
        E = (
            gamma_Ed * E_D +
            gamma_Ewi * E_Wi +
            gamma_Ew * E_W  +
            gamma_Ea * E_A  +
            gamma_Ek * E_K 
            # + gamma_Ers * E_row_sum # Include the new loss
        )

        return E, transformed_vertices

# load data
data = np.load("W.npy", allow_pickle=True).item()
data_dir = 'result/trainWP_1'
if not os.path.exists(data_dir): 
    os.makedirs(data_dir)
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

# import ipdb;ipdb.set_trace()
# Compute shared W_i, AB
first_subject = subject_ids[0]
T = torch.tensor(data[first_subject]['skin_template'], dtype=torch.float32).cuda()
J = torch.tensor(data[first_subject]['joint_template'], dtype=torch.float32).cuda()
W_i = compute_wi(T, J)  # Compute initial blend weights
A_init = compute_A(T, J)  # Compute A

# initial model and optimizer
model = SMPLModel(W_i,A_init[1:], K, neighbor_list, N).cuda()
# all_theta = nn.Parameter((torch.randn(Psub, Preg, K * 3, dtype=torch.float32) * 0.1).cuda())
# optimizer_theta = optim.Adam([all_theta], lr=1e-3)
max_lr_W = 2e-2  # \mathbf{W} 的最大学习率
max_lr_A = 1e-2  # \mathbf{A} 和 \mathbf{K} 的最大学习率
max_lr_K = 5e-5  # \mathbf{A} 和 \mathbf{K} 的最大学习率
total_epochs = 10000  # 总训练 epoch 数

optimizer_W = optim.Adam([model.W_prime])
optimizer_A = optim.Adam([*model.A.parameters()])
optimizer_K = optim.Adam([*model.K.parameters()])

# save
best_loss=np.inf
best_epoch= 0

# Training loop
for epoch in range(total_epochs):
    model.eval()

    if epoch % 200 == 0:
        with torch.no_grad():
            save_model_params(copy.deepcopy(model),save_path=f'{data_dir}/WAK.npz')
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        for i, test_subject_id in enumerate(subject_id_without_shuffle[:20]):
            test_data = data[test_subject_id]
            T_test = torch.tensor(test_data['skin_template'], dtype=torch.float32).cuda().view(1,6890,3)
            J_test = torch.tensor(test_data['joint_template'], dtype=torch.float32).cuda().view(1,K,3)
            V_test = torch.tensor(test_data['skin'][0:1], dtype=torch.float32).cuda().view(1,6890,3)  # First register only
            Tp_test = torch.tensor(test_data['T_p'][0:1], dtype=torch.float32).cuda().view(1,6890,3)  # First register only
            beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()

            theta_test = real_theta[subject_id_without_shuffle.index(test_subject_id), 0:1]  # Corresponding theta
            with torch.no_grad():
                _,predicted_V = model(V_test, T_test, J_test,Tp_test, theta_test, beta_2_test)
                error = torch.max((V_test - predicted_V) ** 2).item()
                ax = axes[i]
                ax.scatter(V_test[0, :, 0].cpu(), V_test[0, :, 1].cpu(), s=1, label="Ground Truth")
                ax.scatter(predicted_V[0, :, 0].cpu(), predicted_V[0, :, 1].cpu(), s=1, label="Prediction")
                ax.set_title(f"Subject {test_subject_id} Error: {error:.4f}")
                ax.legend()
        plt.tight_layout()
        plt.savefig(f'{data_dir}/{epoch}.png')

    model.train()
    # Randomly select a subject
    def sample_minibatch(subject_data, theta, batch_size=150):

        Preg = subject_data['skin'].shape[0]  # 每个 subject 的注册数量
        indices = torch.randperm(Preg)[:batch_size]  # 随机选取 batch_size 个索引

        V_batch = torch.tensor(subject_data['skin'][indices], dtype=torch.float32).cuda()  # (batch_size, N, 3)
        Tp_t_batch = torch.tensor(subject_data['T_p'][indices], dtype=torch.float32).cuda()  # (batch_size, N, 3)
        T_batch = torch.tensor(subject_data['skin_template'], dtype=torch.float32).cuda().unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, N, 3)
        J_batch = torch.tensor(subject_data['joint_template'], dtype=torch.float32).cuda().unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, K, 3)
        theta_batch = theta[indices]  # (batch_size, K, 3)

        return V_batch, T_batch, J_batch, Tp_t_batch, theta_batch

    subject_id = random.choice(subject_ids) 
    
    subject_index = subject_ids.index(subject_id)
    subject_data = data[subject_id]
    theta = real_theta[int(subject_id[3:])]
    # print(subject_id,int(subject_id[3:]))
    # ipdb.set_trace()
    V,T,J,Tp,theta=sample_minibatch(subject_data,theta)

    beta_2 = torch.tensor(subject_data['b2'], dtype=torch.float32).cuda()
    

    lr_W = cosine_decay_lr(epoch, max_lr_W, total_epochs)
    lr_A = cosine_decay_lr(epoch, max_lr_A, total_epochs)
    lr_K = cosine_decay_lr(epoch, max_lr_K, total_epochs)
    update_lr(optimizer_W, lr_W)
    update_lr(optimizer_A, lr_A)
    update_lr(optimizer_K, lr_K)

    # optimizer_theta.zero_grad()
    # loss_theta , _= model(V, T.unsqueeze(0).repeat(Preg, 1, 1), J.unsqueeze(0).repeat(Preg, 1, 1), theta, beta_2)
    # loss_theta.backward()
    # optimizer_theta.step()

    optimizer_W.zero_grad()
    optimizer_A.zero_grad()
    optimizer_K.zero_grad()

    loss , _= model(V, T, J,Tp, theta, beta_2,epoch=epoch)
    loss.backward()
    clip_grad_norm_(model.K.parameters(), max_norm=10., norm_type=2)
    clip_grad_norm_(model.K.parameters(), max_norm=2., norm_type=2)

    optimizer_W.step()
    optimizer_A.step()
    optimizer_K.step()

    
    print(f"Epoch {epoch + 1}")