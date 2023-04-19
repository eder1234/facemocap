import open3d as o3d
import numpy as np
from scipy.interpolate import interp1d

def estimate_nan_values(vector):
    nans = np.isnan(vector)
    if not np.any(nans):
        return vector

    non_nan_indices = np.nonzero(~nans)[0]
    estimated_vector = np.interp(np.arange(len(vector)), non_nan_indices, vector[non_nan_indices])

    return estimated_vector

def interpolate_1d_vector(vector, target_length):
    vector = estimate_nan_values(vector)
    original_length = len(vector)
    if original_length == target_length:
        return vector

    original_indices = np.linspace(0, original_length - 1, num=original_length)
    target_indices = np.linspace(0, original_length - 1, num=target_length)

    # Create an interpolation function using the original data
    # You can replace linear by cubic or quadratic if you want
    interpolation_func = interp1d(original_indices, vector, kind="linear")

    # Apply the interpolation function to the target indices
    interpolated_vector = interpolation_func(target_indices)

    return interpolated_vector

def scale_spc(spc_array, ds_removed=True):
    if ds_removed:
        spc_array[:, :3] = np.nan
    spc_array = spc_array - np.nanmin(np.nanmin(spc_array, axis=0), axis=0)
    spc_array = spc_array / np.nanmax(np.nanmax(spc_array, axis=0), axis=0)
    return spc_array

def interpolate_nd_vector(vector, target_length):
    n_cols = vector.shape[1]
    new_vector = np.zeros((target_length, n_cols))
    vector = vector.reshape(-1,3*108)
    for i in range(n_cols):
        if np.all(np.isnan(vector[:, i])):
            new_vector[:, i] = np.nan
        else:
            new_vector[:, i] = interpolate_1d_vector(vector[:, i], target_length)
    return new_vector

def interpolate_spc(spc, target_length=100): # Missing values are estimated
    spc = spc.reshape(-1, 108*3)
    interpolated_spc = interpolate_nd_vector(spc, target_length)
    interpolated_spc = interpolated_spc.reshape(-1, 108, 3)
    return interpolated_spc

def array_to_spc(spc_array):
    spc = []
    #print(type(spc_array))
    for pc_array in spc_array:
        pc_array = pc_array[3:] # not sure
        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        pcd.points = v3d(pc_array)
        spc.append(pcd)
    return spc

def dental_support_frame(spc_array):
    pass

def displacement(spc_array): # reserved for future use
    pass

def get_spc_from_df(facemocap_df, file_name, scaled=False, interpolated=False, dental_support_frame=False, target_length=100):
    spc_array = facemocap_df.loc[facemocap_df['File name'] == file_name, "Original SPC"]
    spc_array = spc_array.values[0].reshape(-1, 108, 3)
    ds_removed = True    
    if scaled:
        spc_array = scale_spc(spc_array, ds_removed)
    if interpolated:
        spc_array = interpolate_spc(spc_array, target_length)
    if dental_support_frame:
        spc_array = dental_support_frame(spc_array)
    
    spc = array_to_spc(spc_array)
    if ds_removed:
        print("Dental support was removed")
    return spc, spc_array