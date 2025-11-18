import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree

def umeyama_alignment(src, dst, with_scaling=True):
    """
    使用Umeyama算法计算相似变换（旋转、平移、缩放）
    参数:
        src: 源点云 [n, 3]
        dst: 目标点云 [n, 3]
        with_scaling: 是否包含缩放
    返回:
        4x4变换矩阵
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    
    n, dim = src.shape
    src_centroid = np.mean(src, axis=0)
    dst_centroid = np.mean(dst, axis=0)
    
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    
    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    # 计算缩放因子
    scale = 1.0
    if with_scaling:
        src_var = np.var(src_centered, axis=0).sum()
        scale = np.trace(S) / src_var if src_var > 1e-8 else 1.0
    
    # 计算平移
    t = dst_centroid - scale * (R @ src_centroid)
    
    # 构造4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    
    return T

def transform_points(points, T):
    """
    应用4x4变换矩阵到点云
    参数:
        points: 输入点云 [n, 3]
        T: 4x4变换矩阵
    返回:
        变换后的点云 [n, 3]
    """
    homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = (homogeneous @ T.T)[:, :3]
    return transformed

def ransac_similarity_transform(src, dst, n_iters=1000, threshold=0.1, min_inliers=10):
    """
    使用RANSAC估计相似变换
    参数:
        src: 源点云 [n, 3]
        dst: 目标点云 [m, 3]
        n_iters: RANSAC迭代次数
        threshold: 内点距离阈值
        min_inliers: 最小内点数
    返回:
        最佳变换矩阵
    """
    best_inliers = []
    best_transform = np.eye(4)
    
    # 建立最近邻对应关系
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(dst)
    
    for _ in range(n_iters):
        # 随机选择3个点
        indices = np.random.choice(len(src), 3, replace=False)
        src_sample = src[indices]
        
        # 查找最近点
        _, dst_indices = nn.kneighbors(src_sample, return_distance=True)
        dst_sample = dst[dst_indices.flatten()]
        
        # 计算变换矩阵
        T = umeyama_alignment(src_sample, dst_sample, with_scaling=True)
        
        # 变换所有点
        transformed = transform_points(src, T)
        
        # 计算距离
        distances = np.linalg.norm(transformed - dst[nn.kneighbors(transformed)[1].flatten()], axis=1)
        
        # 统计内点
        inliers = np.where(distances < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transform = T
            
    # 使用所有内点重新计算变换
    if len(best_inliers) >= min_inliers:
        src_inliers = src[best_inliers]
        dst_inliers = dst[nn.kneighbors(transform_points(src_inliers, best_transform))[1].flatten()]
        best_transform = umeyama_alignment(src_inliers, dst_inliers, with_scaling=True)
    
    return best_transform

def icp_similarity(source, target, init_pose=np.eye(4), max_iter=50, tolerance=1e-5, distance_threshold=0.1):
    """
    带缩放因子的ICP算法
    参数:
        source: 源点云 [n, 3]
        target: 目标点云 [m, 3]
        init_pose: 初始变换矩阵
        max_iter: 最大迭代次数
        tolerance: 收敛阈值
        distance_threshold: 距离过滤阈值
    返回:
        最终变换矩阵
    """
    T = init_pose.copy()
    prev_error = 0
    kdtree = KDTree(target)
    
    for i in range(max_iter):
        # 变换源点云
        transformed = transform_points(source, T)
        
        # 查找最近邻
        distances, indices = kdtree.query(transformed)
        
        # 过滤距离过大的点
        valid = distances < distance_threshold
        if np.sum(valid) < 3:
            break
            
        src_valid = source[valid]
        tgt_valid = target[indices[valid]]
        
        # 计算增量变换
        T_inc = umeyama_alignment(src_valid, tgt_valid, with_scaling=True)
        
        # 更新变换
        T = T_inc @ T
        
        # 计算误差
        mean_error = np.mean(distances[valid])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    return T

def align_point_clouds(source, target, voxel_size=0.05, ransac_iters=500, icp_iters=50):
    """
    点云对齐主函数
    参数:
        source: 源点云 [n, 3] (稀疏点云)
        target: 目标点云 [m, 3] (全局点云)
        voxel_size: 下采样体素大小
        ransac_iters: RANSAC迭代次数
        icp_iters: ICP迭代次数
    返回:
        4x4变换矩阵
    """
    # 1. 下采样目标点云
    if voxel_size > 0:
        target = voxel_downsample(target, voxel_size)
    
    # 2. RANSAC初始配准
    T_ransac = ransac_similarity_transform(source, target, n_iters=ransac_iters)
    
    # 3. ICP精细配准
    T_final = icp_similarity(source, target, init_pose=T_ransac, max_iter=icp_iters)
    
    return T_final

def voxel_downsample(points, voxel_size):
    """
    体素网格下采样
    参数:
        points: 输入点云 [n, 3]
        voxel_size: 体素大小
    返回:
        下采样后的点云
    """
    if voxel_size <= 0:
        return points
        
    # 计算体素网格索引
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # 使用唯一索引获取每个体素的第一个点
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    return points[unique_indices]
