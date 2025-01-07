import numpy as np
import ipdb
import matplotlib.pyplot as plt
import numpy as np

def visualize_matrix(matrix,name=''):
    """
    可视化一个24*6890的矩阵为热力图，限制颜色条范围为 [0, 1]。
    
    参数:
        matrix (numpy.ndarray): 输入的矩阵，大小为 (24, 6890)
    """
    # 检查矩阵大小是否正确（根据需要取消注释）
    # if matrix.shape != (24, 6890):
    #     raise ValueError("矩阵尺寸必须为24x6890！")
    
    plt.figure(figsize=(5, 25))  # 根据矩阵大小调整图的比例
    plt.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Value")  # 添加颜色条
    plt.title("Heatmap of 24x6890 Matrix")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(f"{name}.png")  # 保存图片为文件
    print("图像已保存为 J_T.png")
def visualize_3d_matrix(matrix_3d):
    """
    可视化一个24x6890x3的矩阵，将第三个维度的三个矩阵绘制为热力图。
    
    参数:
        matrix_3d (numpy.ndarray): 输入的矩阵，大小为 (24, 6890, 3)
    """
    if matrix_3d.shape != (24, 6890, 3):
        raise ValueError("矩阵尺寸必须为24x6890x3！")
    
    # 创建一个包含3个子图的图形
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    titles = ['Dimension 1', 'Dimension 2', 'Dimension 3']
    
    for i in range(3):
        ax = axes[i]
        ax.imshow(matrix_3d[:, :, i], aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(titles[i])
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_xticks([])  # 可选：隐藏列标签以简化
        ax.set_yticks([])  # 可选：隐藏行标签以简化
        fig.colorbar(ax.imshow(matrix_3d[:, :, i], aspect='auto', cmap='viridis', interpolation='nearest'), 
                     ax=ax, orientation='vertical')
    
    plt.suptitle("Heatmaps of 3 Dimensions from a 24x6890x3 Matrix", fontsize=16)
    plt.savefig("J.png")

# # 示例数据
# matrix = np.random.random((24, 6890))  # 生成随机矩阵
# visualize_matrix(matrix)

# J =np.load("W.npy", allow_pickle=True).item()
# visualize_matrix(J["Test"]["W"],'W')
J_T = np.load('result\\W_nnsum1\\W.npy')
visualize_matrix(J_T,'w_nnsum1')

