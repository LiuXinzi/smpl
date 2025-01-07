import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
def compute_A(T, J):

    N = T.shape[0]  # Number of vertices
    K = J.shape[0]  # Number of joints

    # Expand T and J to compute pairwise distances
    T_expanded = T.unsqueeze(0).expand(K, N, 3)  # (K, N, 3)
    J_expanded = J.unsqueeze(1).expand(K, N, 3)  # (K, N, 3)

    # Compute Euclidean distance between each vertex and each joint
    distances = torch.norm(T_expanded - J_expanded, dim=2)  # (K, N)

    # Compute initial weights as reciprocal of distances
    A = 1.0 / (distances + 1e-8)  # Avoid division by zero

    # Normalize each row of A to [0, 1]
    A_min = A.min(dim=1, keepdim=True)[0]
    A_max = A.max(dim=1, keepdim=True)[0]
    A_normalized = (A - A_min) / (A_max - A_min + 1e-8)  # Normalize to [0, 1]

    # Apply ReLU activation (ensure non-negative values)
    A_activated = torch.relu(A_normalized)

    return A_activated

def compute_wi(T, J, num=3):
    N, K = T.shape[0], J.shape[0]
    
    # 计算距离
    distances = torch.norm(T.unsqueeze(1) - J.unsqueeze(0), dim=2)  # (N, K)
    
    # 找到最近的关节点索引和对应的距离
    nearest_indices = torch.argsort(distances, dim=1)[:, :num]  # (N, num)
    nearest_distances = distances.gather(1, nearest_indices)  # (N, num)
    
    # 距离的倒数加权
    inverse_distances = 1.0 / (nearest_distances + 1e-8)  # (N, num)
    normalized_weights = inverse_distances / inverse_distances.sum(dim=1, keepdim=True)  # (N, num)
    
    # 初始化权重矩阵并分配权重
    W_i = torch.zeros((N, K), dtype=torch.float32, device=T.device)  # (N, K)
    W_i.scatter_(1, nearest_indices, normalized_weights)  # 将权重填入对应的位置 (N, K)
    
    return W_i

def theta_to_q(theta):
    # theta: Tensor of shape (K, 3)
    K = theta.size(0)
    q = torch.zeros(K, 4, device=theta.device)  # Quaternion (w, x, y, z)
    half_theta = theta / 2
    q[:, 0] = torch.cos(torch.norm(half_theta, dim=1))  # w
    sin_half_theta = torch.sin(torch.norm(half_theta, dim=1))
    q[:, 1:] = half_theta * (sin_half_theta.unsqueeze(-1) / (torch.norm(half_theta, dim=1, keepdim=True) + 1e-8))
    return q

class SMPLModel(nn.Module):
    def __init__(self, W_i, A_init, K, neighbor_list, N):
        super(SMPLModel, self).__init__()
        self.W_i = W_i.clone().detach()  # Precomputed initial blend weights (N, K)
        self.W = nn.Parameter((torch.rand(W_i.shape, dtype=torch.float32) * 0.5).cuda())  # Blend weights (N, K)
        self.A = nn.ParameterList([nn.Parameter(A_init[i].cuda()) for i in range(K-1)])  # Activation weights
        self.K = nn.ParameterList([nn.Parameter(torch.rand(len(neighbor_list[i]) * 4 + 1, 3 * N, dtype=torch.float32).cuda()) for i in range(K-1)])  # Correction matrices
        self.neighbor_list = neighbor_list

    def compute_G(self, theta, J):
        K, _ = J.shape
        omega = theta.view(K, 3)  # Reshape theta into (K, 3)
        theta_norm = torch.norm(omega, dim=1, keepdim=True)  # (K, 1)

        omega_hat = omega / (theta_norm + 1e-8)  # Normalize (K, 3)
        skew_sym_matrix = torch.zeros((K, 3, 3), device=J.device)
        skew_sym_matrix[:, 0, 1] = -omega_hat[:, 2]
        skew_sym_matrix[:, 0, 2] = omega_hat[:, 1]
        skew_sym_matrix[:, 1, 0] = omega_hat[:, 2]
        skew_sym_matrix[:, 1, 2] = -omega_hat[:, 0]
        skew_sym_matrix[:, 2, 0] = -omega_hat[:, 1]
        skew_sym_matrix[:, 2, 1] = omega_hat[:, 0]

        R = (
            torch.eye(3, device=J.device).unsqueeze(0)  # (1, 3, 3)
            + torch.sin(theta_norm).unsqueeze(-1) * skew_sym_matrix
            + (1 - torch.cos(theta_norm).unsqueeze(-1)) * torch.matmul(skew_sym_matrix, skew_sym_matrix)
        )  # (K, 3, 3)

        G = torch.eye(4, device=J.device).repeat(K, 1, 1)  # (K, 4, 4)
        G[:, :3, :3] = R
        G[:, :3, 3] = J

        return G
   
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
        T_corrected = T + T_p  # Add pose blend corrections
        T_homogeneous = torch.cat([T_corrected, torch.ones((N, 1), device=T.device)], dim=1)  # (N, 4)

        # Apply global transformations (G) to vertices (T_homogeneous)
        transformed = torch.einsum('kij,nj->nki', G, T_homogeneous)[:, :, :3]  # (N, K, 3)

        # Apply blend weights (W)
        transformed_vertices = torch.einsum('nk,nkj->nj', W, transformed)  # (N, 3)

        return transformed_vertices

    def forward(self, V, T, J, theta, beta_2):

        Preg, N, _ = V.shape
        K = J.shape[1]

        # Convert theta to quaternion and compute q_neighbors and q_star_neighbors

        G_prime = torch.stack([self.compute_G(theta[p], J[p]) for p in range(Preg)])  # (Preg, K, 4, 4)

        T_p = self.compute_blendP(T, theta, beta_2,Preg,N,K)  # Pose blend correction

        transformed_vertices = torch.stack([
            self.compute_vertex_transform(T[p], self.W, G_prime[p], T_p[p]) for p in range(Preg)
        ])

        # Compute losses
        E_D = torch.sum((V - transformed_vertices) ** 2)
        E_Wi = torch.norm(self.W - self.W_i) ** 2
        E_W = torch.norm(self.W, p=1)
        E_A = sum(torch.norm(A_j, p=1) for A_j in self.A)
        E_K = sum(torch.norm(K_j) for K_j in self.K)

        # Total error
        E = 2.5* E_D + 0.1*E_Wi + 5*E_W + 2.5*E_A + 2.5*E_K  

        return E, transformed_vertices

# load data
data = np.load("W_new.npy", allow_pickle=True).item()
data_dir = 'result/test'
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

# Compute shared W_i, A
first_subject = subject_ids[0]
T = torch.tensor(data[first_subject]['skin_template'], dtype=torch.float32).cuda()
J = torch.tensor(data[first_subject]['joint_template'], dtype=torch.float32).cuda()
W_i = compute_wi(T, J)  # Compute initial blend weights
A_init = compute_A(T, J)  # Compute A

# initial model and optimizer
model = SMPLModel(W_i,A_init[1:], K, neighbor_list, N).cuda()
all_theta = nn.Parameter((torch.randn(Psub, Preg, K * 3, dtype=torch.float32) * 0.1).cuda())
optimizer_theta = optim.Adam([all_theta], lr=1e-3)
optimizer_W = optim.Adam([model.W], lr=1e-3)
optimizer_AK = optim.Adam([*model.A.parameters(), *model.K.parameters()], lr=1e-3)

# save
best_loss=np.inf
best_epoch= 0

# Training loop
for epoch in range(1000000):
    # Randomly select a subject
    subject_id = random.choice(subject_ids) 
    subject_index = subject_ids.index(subject_id)
    subject_data = data[subject_id]

    T = torch.tensor(subject_data['skin_template'], dtype=torch.float32).cuda()
    J = torch.tensor(subject_data['joint_template'], dtype=torch.float32).cuda()
    V = torch.tensor(subject_data['skin'], dtype=torch.float32).cuda()  # (200, 6890, 3)
    beta_2 = torch.tensor(subject_data['b2'], dtype=torch.float32).cuda()
    theta = all_theta[subject_index]

    optimizer_theta.zero_grad()
    loss_theta , _= model(V, T.unsqueeze(0).repeat(Preg, 1, 1), J.unsqueeze(0).repeat(Preg, 1, 1), theta, beta_2)
    loss_theta.backward()
    optimizer_theta.step()

    optimizer_W.zero_grad()
    loss_W , _= model(V, T.unsqueeze(0).repeat(Preg, 1, 1), J.unsqueeze(0).repeat(Preg, 1, 1), theta.detach(), beta_2)
    loss_W.backward()
    optimizer_W.step()

    optimizer_AK.zero_grad()
    loss_AK ,_= model(V, T.unsqueeze(0).repeat(Preg, 1, 1), J.unsqueeze(0).repeat(Preg, 1, 1), theta.detach(), beta_2)
    loss_AK.backward()
    optimizer_AK.step()

    model.eval()
    with torch.no_grad():
        W_copy = model.W.clone().detach().cpu().numpy()
        MAE=np.mean(np.abs(W_copy- real_W))
        MAX=np.max(np.abs(W_copy- real_W))
        
        if MAE < best_loss:
            best_loss=MAE
            best_epoch=epoch
            np.save(f'{data_dir}/W.npy',W_copy)

        # print(f"W_MAE: {MAE}, W_MAX: {MAX}, Best_E: {best_epoch}")

    # train fig each 2000 epoch
    if epoch % 2000 == 0:
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        for i, test_subject_id in enumerate(subject_id_without_shuffle[:20]):
            test_data = data[test_subject_id]
            T_test = torch.tensor(test_data['skin_template'], dtype=torch.float32).cuda().view(1,6890,3)
            J_test = torch.tensor(test_data['joint_template'], dtype=torch.float32).cuda().view(1,K,3)
            V_test = torch.tensor(test_data['skin'][0:1], dtype=torch.float32).cuda().view(1,6890,3)  # First register only
            beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()

            theta_test = all_theta[subject_ids.index(test_subject_id), 0:1]  # Corresponding theta

            with torch.no_grad():
                _,predicted_V = model(V_test, T_test, J_test, theta_test, beta_2_test)
                error = torch.mean((V_test - predicted_V) ** 2).item()
                ax = axes[i]
                ax.scatter(V_test[0, :, 0].cpu(), V_test[0, :, 1].cpu(), s=1, label="Ground Truth")
                ax.scatter(predicted_V[0, :, 0].cpu(), predicted_V[0, :, 1].cpu(), s=1, label="Prediction")
                ax.set_title(f"Subject {test_subject_id} Error: {error:.4f}")
                ax.legend()
        plt.tight_layout()
        plt.savefig(f'{data_dir}/{epoch}.png')
    
    # validation
    if epoch % 10000 == 0 and epoch != 0:
        with torch.no_grad():
            copied_model = copy.deepcopy(model)
        theta_valid = nn.Parameter((torch.randn(5, 20, K * 3, dtype=torch.float32) * 0.1).cuda())
        optimizer_valid = optim.Adam([theta_valid], lr=1e-3)
        for valid_epoch in range(5000):
            print(f'Test_epoch:{valid_epoch}')
            test_id = random.choice(test_ids)  # Randomly select a subject
            test_index = test_ids.index(test_id)
            test_data = data[test_id]
            T_test = torch.tensor(test_data['skin_template'], dtype=torch.float32).cuda()
            J_test = torch.tensor(test_data['joint_template'], dtype=torch.float32).cuda()
            V_test = torch.tensor(test_data['skin'], dtype=torch.float32).cuda()  # (200, 6890, 3)
            beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()
            theta_test = theta_valid[test_index]
            optimizer_valid.zero_grad()
            loss_test , _= copied_model(V_test, T_test.unsqueeze(0).repeat(20, 1, 1), J_test.unsqueeze(0).repeat(20, 1, 1), theta_test, beta_2_test)
            loss_test.backward()
            optimizer_valid.step()
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        axes = axes.flatten()
        for i, test_subject_id in enumerate(test_ids[:5]):
            test_data = data[test_subject_id]
            T_test = torch.tensor(test_data['skin_template'], dtype=torch.float32).cuda().view(1,6890,3)
            J_test = torch.tensor(test_data['joint_template'], dtype=torch.float32).cuda().view(1,K,3)
            V_test = torch.tensor(test_data['skin'][0:1], dtype=torch.float32).cuda().view(1,6890,3)  # First register only
            beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()

            theta_test = theta_valid[test_ids.index(test_subject_id), 0:1]  # Corresponding theta

            with torch.no_grad():
                _,predicted_V = copied_model(V_test, T_test, J_test, theta_test, beta_2_test)
                error = torch.mean((V_test - predicted_V) ** 2).item()
                ax = axes[i]
                ax.scatter(V_test[0, :, 0].cpu(), V_test[0, :, 1].cpu(), s=1, label="Ground Truth")
                ax.scatter(predicted_V[0, :, 0].cpu(), predicted_V[0, :, 1].cpu(), s=1, label="Prediction")
                ax.set_title(f"Test_Subject {test_subject_id} Error: {error:.4f}")
                ax.legend()
        plt.tight_layout()
        plt.savefig(f'{data_dir}/{epoch}_test.png')
    model.train()
    print(f"Epoch {epoch + 1}")