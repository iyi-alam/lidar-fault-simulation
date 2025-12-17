## References used in simulation

- [Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection in Adverse Weather](https://arxiv.org/abs/2108.05249): Used to siumlate the dust as a homogenous medium just like the fog. The only change is that the alpha and beta values (extinction  and backscattering co-efficients) have been computed using scattering theory as per [LISA](https://arxiv.org/abs/2107.07004)

- [LIDAR Point Cloud Augmentation for Dusty Weather Based on a Physical Simulation](https://www.mdpi.com/2227-7390/12/1/141): Tried simulating the method discussed in this paper. However, the computational requirement is prohibitively high. The paper simulates actual dust particles in the 3-D space around lidar and number of such particles may shoot upto trillions. Then it computes extinction and scattering from each of them which is not practical for large scale dataset simulation.

- [Dust Particle Size Distribution](https://onlinelibrary.wiley.com/doi/pdf/10.1155/2016/6940502): Reference for getting the data related to the distribution of dust particle sizes under varying scenarios.

- [Reflectivity of Dust Particles](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JD030629#:~:text=Complex%20refractive%20index%20of%20Asian%20dust%20is,in%20dust%20samples%20reported%20in%20previous%20literature.&text=Meanwhile%2C%20the%20imaginary%20refractive%20indices%20of%20dust,free%20troposphere%20determined%20by%20Arimoto%20et%20al.) 