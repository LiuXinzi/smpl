import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
def compute_A(T, J):
    """
    初始化权重矩阵 A (K x N)。
    Args:
        T: Tensor of shape (N, 3), skin vertices coordinates.
        J: Tensor of shape (K, 3), joint coordinates.
    Returns:
        A: Tensor of shape (K, N), initialized weight matrix.
    """
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
def compute_wi(T, J):
    N = T.shape[0]
    K = J.shape[0]
    
    # Expand and compute distances
    T_expanded = T.unsqueeze(1).expand(N, K, 3)  # (N, K, 3)
    J_expanded = J.unsqueeze(0).expand(N, K, 3)  # (N, K, 3)
    distances = torch.norm(J_expanded - T_expanded, dim=2)  # Compute distances (N, K)
    
    # Find 4 nearest joints for each vertex
    nearest_indices = torch.argsort(distances, dim=1)[:, :1]  # (N, 1)
    
    # Initialize blend weights (N, K)
    W_i = torch.zeros((N, K), dtype=torch.float32, device=T.device)
    
    # Assign equal weights to the 1 nearest joints
    for i in range(1):
        W_i[torch.arange(N), nearest_indices[:, i]] += 0.99  # (N, K)

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
    def compute_blendP(self, T, q_neighbors, q_star_neighbors, beta_2):
        Preg, N, _ = T.shape  # Extract dimensions from T
        _, K, y, _ = q_neighbors.shape  # Extract dimensions from q_neighbors

        T_p = torch.zeros_like(T)  # Initialize blend contribution (Preg, N, 3)

        # Compute quaternion differences (Preg, K, y, 4)
        q_diff = q_neighbors - q_star_neighbors

        # Concatenate q_diff and beta_2 to form (Preg, K, y, 5)
        beta_2_expanded = beta_2.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(Preg, K, y, 1)
        q_combined = torch.cat([q_diff, beta_2_expanded], dim=-1)  # (Preg, K, y, 5)

        # Reshape q_combined to (Preg, K, 5*y)
        q_combined_flattened = q_combined.view(Preg, K, -1)  # (Preg, K, 5*y)

        for j in range(K-1):
            K_j = self.K[j]  # Correction matrix (5*y, 3N)

            # Matrix multiplication with K_j
            corrected_contribution = torch.matmul(q_combined_flattened[:, j+1, :], K_j)  # (Preg, 3N)

            A_j = self.A[j]  # Activation weights (N)
            
            # Apply ReLU activation to A_j
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
        q = theta_to_q(theta.view(Preg * K, 3)).view(Preg, K, 4)
        q_star = theta_to_q(torch.zeros_like(theta.view(Preg * K, 3))).view(Preg, K, 4)

        q_neighbors = torch.stack([q[:, self.neighbor_list[j], :] for j in range(K)], dim=1).to(q.device)
        q_star_neighbors = torch.stack([q_star[:, self.neighbor_list[j], :] for j in range(K)], dim=1).to(q_star.device)
        # import ipdb;ipdb.set_trace()
        G_prime = torch.stack([self.compute_G(theta[p], J[p]) for p in range(Preg)])  # (Preg, K, 4, 4)

        T_p = self.compute_blendP(T, q_neighbors, q_star_neighbors, beta_2)  # Pose blend correction

        transformed_vertices = torch.stack([
            self.compute_vertex_transform(T[p], self.W, G_prime[p], T_p[p]) for p in range(Preg)
        ])

        # Compute losses
        E_D = torch.sum((V - transformed_vertices) ** 2)
        E_Wi = torch.norm(self.W - self.W_i) ** 2
        E_W = torch.norm(self.W, p=1)
        # import ipdb;ipdb.set_trace()
        E_A = sum(torch.norm(A_j, p=1) for A_j in self.A)
        E_K = sum(torch.norm(K_j) for K_j in self.K)

        # Total error
        E = 5* E_D + E_Wi + 50*E_W + 40*E_A + 30*E_K
        # print(E_D.item(), E_W.item(), E_A.item(), E_K.item())
        return E, transformed_vertices

data = np.load("W_new.npy", allow_pickle=True).item()
data_dir = 'result/1_10'
subject_ids = [key for key in data.keys() if key != "Test"]
subject_id_without_shuffle= [key for key in data.keys() if key != "Test"]
random.shuffle(subject_ids) 
Preg = 200  # Number of registrations per subject
Psub = len(subject_ids)  # Number of subjects
K = 24  # Number of joints
N = 6890  # Number of vertices

# Compute shared W_i
first_subject = subject_ids[0]
T = torch.tensor(data[first_subject]['skin_template'], dtype=torch.float32).cuda()
J = torch.tensor(data[first_subject]['joint_template'], dtype=torch.float32).cuda()
W_i = compute_wi(T, J)  # Compute initial blend weights
A_init = compute_A(T, J)  # Compute A

real_W = data["Test"]["W"]

neighbor_list = [[i] for i in range(K)]
# neighbor_list[2]=[2,3]

model = SMPLModel(W_i,A_init[1:], K, neighbor_list, N).cuda()

# Pre-generate theta for all subjects and registrations
all_theta = nn.Parameter((torch.randn(Psub, Preg, K * 3, dtype=torch.float32) * 0.1).cuda())
optimizer_theta = optim.Adam([all_theta], lr=1e-3)

# Separate optimizers for W and A + K
optimizer_W = optim.Adam([model.W], lr=1e-3)
optimizer_AK = optim.Adam([*model.A.parameters(), *model.K.parameters()], lr=1e-3)
best_loss=np.inf
best_epoch= 0
# Training loop
for epoch in range(1000000):

    subject_id = random.choice(subject_ids)  # Randomly select a subject
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
    with torch.no_grad():
        W_copy = model.W.clone().detach().cpu().numpy()
        MAE=np.mean(np.abs(W_copy- real_W))
        MAX=np.max(np.abs(W_copy- real_W))
        
        if MAE < best_loss:
            best_loss=MAE
            best_epoch=epoch
            # np.save(f'{data_dir}/W.npy',W_copy)

        print(f"W_MAE: {MAE}, W_MAX: {MAX}, Best_E: {best_epoch}")

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

    print(f"Epoch {epoch + 1}")