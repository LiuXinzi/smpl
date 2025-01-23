import numpy as np
import torch

# 定义 axis2quat 函数
def axis2quat(p):
    angle = np.sqrt(np.clip(np.sum(np.square(p), axis=1), 1e-16, 1e16))
    norm_p = p / angle[:, np.newaxis]
    cos_angle = np.cos(angle / 2)
    sin_angle = np.sin(angle / 2)
    qx = norm_p[:, 0] * sin_angle
    qy = norm_p[:, 1] * sin_angle
    qz = norm_p[:, 2] * sin_angle
    qw = cos_angle - 1  # 注意这里的 -1 偏移
    return np.concatenate([qx[:, np.newaxis], qy[:, np.newaxis], qz[:, np.newaxis], qw[:, np.newaxis]], axis=1)

# 定义 theta_to_q 函数
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

# 测试函数
def test_quaternion_difference():
    # 输入相同的轴角表示 (numpy for axis2quat, torch for theta_to_q)
    axis_angle_np = np.array([[0.0, 0.0, np.pi / 2], [0.0, np.pi / 4, 0.0], [np.pi / 3, 0.0, 0.0]])  # 3个测试数据
    axis_angle_torch = torch.tensor(axis_angle_np, dtype=torch.float32)

    # 计算四元数
    quat_axis2quat = axis2quat(axis_angle_np)  # 使用 axis2quat
    quat_theta_to_q = theta_to_q(axis_angle_torch).cpu().numpy()  # 使用 theta_to_q，并转为 numpy
    q_star = theta_to_q(torch.zeros_like(axis_angle_torch)).cpu().numpy()
    q= quat_theta_to_q-q_star

    # 打印比较结果
    print("Axis-Angle Input (axis_angle_np):")
    print(axis_angle_np)
    print("\nQuaternion from axis2quat (qx, qy, qz, qw):")
    print(quat_axis2quat)
    print("\nQuaternion from theta_to_q (qw, qx, qy, qz):")
    print(q)
    print(q_star)

    # 比较差异
    print("\nDifference (axis2quat - theta_to_q):")
    quat_axis2quat_reordered = np.concatenate([quat_axis2quat[:, 3:4], quat_axis2quat[:, 0:3]], axis=1)  # 重排 axis2quat 格式
    print(quat_axis2quat_reordered - quat_theta_to_q)

# 运行测试
test_quaternion_difference()
