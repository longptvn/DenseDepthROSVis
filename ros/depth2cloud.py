import numpy as np

def meshgrid_abs(height, width):
  """Meshgrid in the absolute coordinates."""
  x_t = np.matmul(
      np.ones(shape=np.stack([height, 1])),
      np.transpose(np.expand_dims(np.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = np.matmul(
      np.expand_dims(np.linspace(-1.0, 1.0, height), 1),
      np.ones(shape=np.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * np.cast[np.float32](width - 1)
  y_t = (y_t + 1.0) * 0.5 * np.cast[np.float32](height - 1)
  x_t_flat = np.reshape(x_t, (1, -1))
  y_t_flat = np.reshape(y_t, (1, -1))
  ones = np.ones_like(x_t_flat)
  grid = np.concatenate([x_t_flat, y_t_flat, ones], axis=0)
  return grid

def pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
  """Transform coordinates in the pixel frame to the camera frame."""
  cam_coords = np.matmul(intrinsic_mat_inv, pixel_coords) * depth
  return cam_coords

def get_cloud(depth, intrinsics_inv):  # pylint: disable=unused-argument
  """Convert depth map to 3D point cloud."""
  batch_size, img_height, img_width = depth.shape[0], depth.shape[1], depth.shape[2]
  depth = np.reshape(depth, [batch_size, 1, img_height * img_width])
  grid = meshgrid_abs(img_height, img_width)
  grid = np.tile(np.expand_dims(grid, 0), [batch_size, 1, 1])
  cam_coords = pixel2cam(depth, grid, intrinsics_inv)
  cam_coords = np.transpose(cam_coords, [0, 2, 1])
  cam_coords = np.reshape(cam_coords, [batch_size, img_height, img_width, 3])
  return cam_coords

