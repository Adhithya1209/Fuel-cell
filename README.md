# Fuel-cell
The project consists of the following files:
1) fuel_cell.py : This is the main file which consists of the pysical models that use semiemprical equations and state based equations to simulate a fuel cells (Nedstack PS6) under different operating conditions. The model framework support can be extended to other fuel cell stacks if their nominal parameters are known.
2) grid_search.py : This script finds the valid regions of the fuel cell pysical models
3) lstm_test_noise.py : Tests the lstm's performance when noise is present in the simulated/ synthetic data set
4) ode_solver : Consists of the solver used to solve the state space model which in turn consists of ordinary partial differential equations
5) pca.py : The input to the surrogate model is preprocessed to its principal components to reduce the number of computations done in the dense networks. The inverse pca is performed on the output from the dense networks to get the polarisation curves
6) test_script.py tests and shows the demo for all the critical components of this project in a systematic/sequential order for the nedstack ps6 pemfc and others
