import pickle
import numpy as np

smpl=np.load("smpl_skin.npy") 
num, total_points, dims = smpl.shape
marker_index=np.array([3099, 3156, 6522, 6573, 4655, 4386, 4515, 4396, 4507, 4336, 4514,
       4926, 4495, 4634, 4615, 4605, 4845, 6599, 6727, 4580, 4664, 4560,
       6721, 6866, 6786, 6703, 6749, 6750, 6623, 6618, 1168,  962, 1172,
        909, 1021,  848, 1028,  935, 1010, 1148, 1131, 1121, 1372, 3199,
       3328, 1083, 1178, 1076, 3321, 3466, 3387, 3303, 3348, 3350, 3223,
       3216, 3023, 3502, 1485, 6289, 1306, 3169,  589, 4079, 5295, 4219,
       4724, 6471, 5090, 4266, 4862, 4115, 6282, 5112, 5155, 5568, 5573,
       5675, 5905, 5545, 5714, 1834, 2886, 1239, 3013, 1621,  780, 1389,
        628, 1505, 1657, 1688, 2108, 2112, 2214, 2445, 2082, 2251, 3051,
        395,  136,  410, 3897, 3648,  414])
marker_points = np.zeros((num, len(marker_index), dims)) 
for i, index in enumerate(marker_index):
    marker_points[:, i, :] = smpl[:, index, :] 

print("marker_points 的形状:", marker_points.shape)
with open('marker_points.pkl', 'wb') as f:
    pickle.dump(marker_points, f)