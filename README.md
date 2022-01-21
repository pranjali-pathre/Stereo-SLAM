# Stereo-SLAM

### PART 1: Stereo dense reconstruction
3-D point clouds are very useful in robotics for several tasks such as object detection, motion estimation (3D-3D matching or 3D-2D matching), SLAM, and other forms of scene understanding. Stereo cameras provide us with a convenient way to generate dense point clouds. Dense here, in contrast to sparse,means all the image points are used for the reconstruction.

<!-- Procedure:
* Generate a disparity map for each stereo pair.
* Use the camera parameters and baseline information generate colored point clouds from each disparity map.
* Register (or transform) all the generated point clouds into the world frame by using the providedground truth poses.
* Visualize the registered point cloud data, in color using open3d. -->

### PART 2: Motion estimation using iterative PnP

Using the generated reconstruction from the previous part, synthesize a new image taken by a virtualmonocular camera fixed at any arbitrary position and orientation. Your task in this part is to recover this pose using an iterative Perspective-from-n-Points (PnP) algorithm.
