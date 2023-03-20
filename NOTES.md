# Probabilistic SLAM


<u>**Map Types** and their respective **observation models:**</u> 
 
1. [x] **Occupancy grid map**: 
	- Bayesian map update, fast look-up/evaluation, different versions for map creation/updates (e.g. bayesian update; from lines to cell values)
	- [x] **Grid distribution**
		- Fast, Evaluation is a simple look-up in the map.
		- Uncertainty is handled on the map level through updating the map. This means it is "baked-in". 

2. [x] **Polygon / Line map**:
	- ...
	- [x] **Deterministic Raycaster + Noise**
		- Performance engineering needed
		- Different versions possible, e.g. could translate to point cloud
	- [x] **Segment distributions** (almost like point cloud): 	

3. [ ] **Point cloud map**:
	- ...
	- [x] **Deterministic Raycaster + Noise**
		- Performance engineering needed (e.g. via Binning -- faster)
		- Different versions possible, e.g. could translate to point cloud
	- [ ] **Gaussian Mixtures**
		- Different versions possible... 3d3p vs full mixture  	 	
