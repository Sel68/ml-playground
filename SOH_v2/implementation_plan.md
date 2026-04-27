Act as an expert in Physics-Informed Machine Learning and Battery Electrochemical Modeling. 

### OBJECTIVE
Implement the B2 hybrid PINN architecture for Battery State of Health (SOH) estimation as described in Wang et al. (Nature Communications, 2024), enhanced with an Arrhenius-based temporal scaling innovation.

### DATA SOURCE & ENVIRONMENT
- Location: The folder "Dataset_1_NCA_battery" is located in the current root directory.
- Format: Each CSV file in this folder represents one cell's cycling data.
- CSV Columns: The files contain 9 columns, including 'time/s', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', and 'cycle number'.
- Task: Iterate through all CSV files in "Dataset_1_NCA_battery" to build the training and testing sets.

### CORE ARCHITECTURE
1. Solution Network F(t_arr, x; Φ): Maps Arrhenius-scaled time (t_arr) and 16 statistical features (x) to estimated SOH (û).
2. Dynamics Network G(t_arr, x, u, u_t, u_x; Θ): A learnable surrogate for degradation dynamics.
3. Feature Set: Extract 16 CC-CV statistical features (Voltage mean/std/kurtosis/skewness, etc.) and the 3 relaxation features (VAR, SKE, MAX) as described in the research proposal.

### THE INNOVATION: ARRHENIUS SCALING
Transform the linear cycle count (t) into "Equivalent Aging Time" (t_arr) using:
t_arr = sum(γ(T_i) * Δt_i)
where γ(T) = exp[(Ea / R) * (1/T_ref - 1/T)]. 
- Cell Temperature (T_i): Extract this from the filename (e.g., 'X' in CYX-Y_Z-#N).
- Reference Temperature (T_ref): 298.15 K.
- Activation Energy (Ea): Set as a trainable parameter, initialized at 52 kJ/mol.

### LOSS FUNCTION
Implement the multi-task loss: L = L_data + α*L_PDE + β*L_mono.
- L_PDE: Enforces |∂F/∂t_arr - G| = 0 via automatic differentiation.
- L_mono: Ensures SOH is non-increasing (û_{k+1} <= û_k).

### VALIDATION STRATEGY & ANTI-LEAKAGE
- Split the dataset by CELL ID (e.g., 40 cells for training, 13 for validation, 13 for testing).
- Perform Min-Max normalization on features to the range [-1, 1].
- Strictly ensure no data from test Cell IDs is used during the training or validation of the PINN.

### OUTPUT REQUIREMENTS
1. Python/PyTorch code for the PINN class and custom Loss functions.
2. Training loop with early stopping based on validation loss.
3. Visualization script (Matplotlib) showing:
   - SOH Predicted vs. True trajectories for test cells.
   - A plot of "Thermal Age" (t_arr) vs. Cycle Number for different temperature batches.