import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import ipdb
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

def compute_wi(T, J, num=3):
    N, K = T.shape[0], J.shape[0]

    distances = torch.norm(T.unsqueeze(1) - J.unsqueeze(0), dim=2)  # (N, K)
    nearest_indices = torch.argsort(distances, dim=1)[:, :num]  # (N, num)
    nearest_distances = distances.gather(1, nearest_indices)  # (N, num)
    inverse_distances = 1.0 / (nearest_distances + 1e-8)  # (N, num)
    normalized_weights = inverse_distances / inverse_distances.sum(dim=1, keepdim=True)  # (N, num)
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


class SMPLModel(nn.Module):
    def __init__(self, W_i, A_init, K, neighbor_list, N):
        super(SMPLModel, self).__init__()
        self.W_i = W_i.clone().detach()  # Precomputed initial blend weights (N, K)
        self.W_prime = nn.Parameter((torch.rand(W_i.shape, dtype=torch.float32) * 0.5).cuda())  # Blend weights (N, K)
        self.A = nn.ParameterList([nn.Parameter(A_init[i].cuda()) for i in range(K-1)])  # Activation weights
        self.K = nn.ParameterList([nn.Parameter(torch.rand(len(neighbor_list[i]) * 4 + 1, 3 * N, dtype=torch.float32).cuda()) for i in range(K-1)])  # Correction matrices
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
    def rodrigues(self,pose):
        """
        Convert axis-angle representation to rotation matrices.
        Args:
            pose: Tensor of shape (K, 3)
        Returns:
            R: Tensor of shape (K, 3, 3)
        """
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


    def compute_G(self,pose, J):

        K = J.shape[0]
        pose=pose.view(K, 3)
        R = self.rodrigues(pose)  # (K, 3, 3)

        # Initialize local transformation matrices
        G_local = torch.eye(4, device=pose.device).unsqueeze(0).repeat(K, 1, 1)  # (K, 4, 4)
        G_local[:, :3, :3] = R
        G_local[:, :3, 3] = J

        # Recursive computation of global transformations
        G_global = [G_local[0]]
        for i in range(1, K):
            parent_idx = self.K_tree[0, i]
            if parent_idx == 4294967295:  # Root joint (no parent)
                G_global.append(G_local[i])
            else:
                G_global.append(torch.matmul(G_global[parent_idx], G_local[i]))
        G_global = torch.stack(G_global, dim=0)  # (K, 4, 4)

        # Center transformations around joints
        J_h = torch.cat([J, torch.zeros((K, 1), device=J.device)], dim=1).unsqueeze(-1)  # (K, 4, 1)
        G_centered = G_global - torch.matmul(G_global, J_h)

        return G_centered

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



    def forward(self, V, T, J, theta, beta_2):

        Preg, N, _ = V.shape
        K = J.shape[1]

        # Convert theta to quaternion and compute q_neighbors and q_star_neighbors
        W_normalized = F.softmax(self.W_prime, dim=1)
        G_prime = torch.stack([self.compute_G(theta[p], J[p]) for p in range(Preg)])  # (Preg, K, 4, 4)

        T_p = self.compute_blendP(T, theta, beta_2,Preg,N,K)  # Pose blend correction

        transformed_vertices = torch.stack([
            self.compute_vertex_transform(T[p], W_normalized, G_prime[p], T_p[p]) for p in range(Preg)
        ])

        # Compute losses
        E_D = torch.sum((V - transformed_vertices) ** 2)
        E_Wi = torch.norm(W_normalized - self.W_i) ** 2
        E_W = torch.norm(W_normalized, p=1)
        E_A = sum(torch.norm(A_j, p=1) for A_j in self.A)
        E_K = sum(torch.norm(K_j) for K_j in self.K)
        # print(E_D,E_W,E_A,E_K)
        # Total error
        E = 10* E_D + 0.05*E_Wi + 2.5*E_W + 2*E_A + 2*E_K  

        return E, transformed_vertices

# load data
data = np.load("W2.npy", allow_pickle=True).item()
data_dir = 'result/thetaisright'
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
optimizer_W = optim.Adam([model.W_prime], lr=1e-3)
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
    theta = real_theta[subject_index]

    # optimizer_theta.zero_grad()
    # loss_theta , _= model(V, T.unsqueeze(0).repeat(Preg, 1, 1), J.unsqueeze(0).repeat(Preg, 1, 1), theta, beta_2)
    # loss_theta.backward()
    # optimizer_theta.step()

    optimizer_W.zero_grad()
    loss_W , _= model(V, T.unsqueeze(0).repeat(Preg, 1, 1), J.unsqueeze(0).repeat(Preg, 1, 1), theta, beta_2)
    loss_W.backward()
    optimizer_W.step()

    optimizer_AK.zero_grad()
    loss_AK ,_= model(V, T.unsqueeze(0).repeat(Preg, 1, 1), J.unsqueeze(0).repeat(Preg, 1, 1), theta, beta_2)
    loss_AK.backward()
    optimizer_AK.step()

    model.eval()

        # W_copy = F.softmax(model.W_prime.clone(), dim=1).detach().cpu().numpy()
        # MAE=np.mean(np.abs(W_copy- real_W))
        # MAX=np.max(np.abs(W_copy- real_W))
        
        # if MAE < best_loss:
        #     best_loss=MAE
        #     best_epoch=epoch
        #     np.save(f'{data_dir}/W.npy',W_copy)

        # print(f"W_MAE: {MAE}, W_MAX: {MAX}, Best_E: {best_epoch}")

    # train fig each 2000 epoch
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
            beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()

            theta_test = real_theta[subject_ids.index(test_subject_id), 0:1]  # Corresponding theta
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
    
    # # validation
    # if epoch % 10000 == 0 and epoch != 0:
    #     with torch.no_grad():
    #         copied_model = copy.deepcopy(model)
    #     theta_valid = nn.Parameter((torch.randn(5, 20, K * 3, dtype=torch.float32) * 0.1).cuda())
    #     optimizer_valid = optim.Adam([theta_valid], lr=1e-3)
    #     for valid_epoch in range(5000):
    #         print(f'Test_epoch:{valid_epoch}')
    #         test_id = random.choice(test_ids)  # Randomly select a subject
    #         test_index = test_ids.index(test_id)
    #         test_data = data[test_id]
    #         T_test = torch.tensor(test_data['skin_template'], dtype=torch.float32).cuda()
    #         J_test = torch.tensor(test_data['joint_template'], dtype=torch.float32).cuda()
    #         V_test = torch.tensor(test_data['skin'], dtype=torch.float32).cuda()  # (200, 6890, 3)
    #         beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()
    #         theta_test = theta_valid[test_index]
    #         optimizer_valid.zero_grad()
    #         loss_test , _= copied_model(V_test, T_test.unsqueeze(0).repeat(20, 1, 1), J_test.unsqueeze(0).repeat(20, 1, 1), theta_test, beta_2_test)
    #         loss_test.backward()
    #         optimizer_valid.step()
    #     fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    #     axes = axes.flatten()
    #     for i, test_subject_id in enumerate(test_ids[:5]):
    #         test_data = data[test_subject_id]
    #         T_test = torch.tensor(test_data['skin_template'], dtype=torch.float32).cuda().view(1,6890,3)
    #         J_test = torch.tensor(test_data['joint_template'], dtype=torch.float32).cuda().view(1,K,3)
    #         V_test = torch.tensor(test_data['skin'][0:1], dtype=torch.float32).cuda().view(1,6890,3)  # First register only
    #         beta_2_test = torch.tensor(test_data['b2'], dtype=torch.float32).cuda()

    #         theta_test = theta_valid[test_ids.index(test_subject_id), 0:1]  # Corresponding theta

    #         with torch.no_grad():
    #             _,predicted_V = copied_model(V_test, T_test, J_test, theta_test, beta_2_test)
    #             error = torch.mean((V_test - predicted_V) ** 2).item()
    #             ax = axes[i]
    #             ax.scatter(V_test[0, :, 0].cpu(), V_test[0, :, 1].cpu(), s=1, label="Ground Truth")
    #             ax.scatter(predicted_V[0, :, 0].cpu(), predicted_V[0, :, 1].cpu(), s=1, label="Prediction")
    #             ax.set_title(f"Test_Subject {test_subject_id} Error: {error:.4f}")
    #             ax.legend()
    #     plt.tight_layout()
    #     plt.savefig(f'{data_dir}/{epoch}_test.png')
    model.train()
    print(f"Epoch {epoch + 1}")