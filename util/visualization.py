import copy
import open3d as o3d
import numpy as np

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def get_color_map(x):
  colours = plt.cm.Spectral(x)
  return colours[:, :3]


def mesh_sphere(pcd, voxel_size, sphere_size=0.6):
  # Create a mesh sphere
  spheres = o3d.geometry.TriangleMesh()
  s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
  s.compute_vertex_normals()

  for i, p in enumerate(pcd.points):
    si = copy.deepcopy(s)
    trans = np.identity(4)
    trans[:3, 3] = p
    si.transform(trans)
    si.paint_uniform_color(pcd.colors[i])
    spheres += si
  return spheres


def get_colored_point_cloud_feature(pcd, feature, voxel_size):
  tsne_results = embed_tsne(feature)

  color = get_color_map(tsne_results)
  pcd.colors = o3d.utility.Vector3dVector(color)
  spheres = mesh_sphere(pcd, voxel_size)

  return spheres


def embed_tsne(data):
  """
  N x D np.array data
  """
  tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
  tsne_results = tsne.fit_transform(data)
  tsne_results = np.squeeze(tsne_results)
  tsne_min = np.min(tsne_results)
  tsne_max = np.max(tsne_results)
  return (tsne_results - tsne_min) / (tsne_max - tsne_min)
