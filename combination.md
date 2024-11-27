# combination of ZED SDK Depth and depth-Anything depth

Here shows the reason for disparity based RANSAC fitting.


![full_depth.png](figures/full_depth.png)
An example of combination.
left top: by ZED SDK has a lot of missing point in this condition.
right up: by depth anything.
right down: You can fill missing point by depth anything.


![depth_cmp_log.png](figures/depth_cmp_log.png)

In log scale Depth and disparity has slope -1.

---
In good condition
![full_depth_2.png](figures/full_depth_2.png)
Both of ZED SDK and depth anything return good value.

![depth_cmp_log_2.png](figures/depth_cmp_log_2.png)


