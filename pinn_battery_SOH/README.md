# PINN Battery SOH Estimation & Climate Simulation

A Physics-Informed Neural Network (PINN) project for battery State of Health (SOH) estimation based on the **Nature Communications (2024)** paper: *"Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis"*.

## Model Architecture

The model is a 4-layer Multilayer Perceptron (MLP) specifically designed to ingest time-series features and enforce physical constraints.

- **Inputs (17 total)**: 
    - 16 features extracted from the charging phase (CC-CV).
    - 1 normalized cycle index.
- **Hidden Layers**: 3 hidden layers with 128 neurons each, using **Tanh** activation functions for smooth gradient flow.
- **Output Layer**: Single neuron with **Sigmoid** activation, predicting SOH in the range [0, 1].

### Physics-Informed Component: Monotonicity Loss
The "Physics" in this PINN is implemented via a custom loss term $L_{mono}$ that enforces the non-regenerative nature of battery capacity:
$$L_{mono} = \frac{1}{N-1} \sum_{i=0}^{N-2} \text{ReLU}(\hat{\text{SOH}}_{i+1} - \hat{\text{SOH}}_i)$$
This ensures the model learns that SOH cannot increase over time (excluding small measurement noise), significantly improving the physical consistency of long-term projections.

## Feature Extraction (Nature 2024)

Features are extracted from the **Charging Phase** where the battery behavior is most stable. For each cycle, we analyze the:
1. **Voltage Range**: $[V_{end}-0.2, V_{end}]$
2. **Current Range**: During Constant Voltage (CV) phase $[0.1, 0.5]$ A.

From these segments, we compute 8 statistics for both Voltage and Current (16 total):
- Mean, Standard Deviation, Kurtosis, Skewness
- Segment Duration (Time)
- Accumulated Charge (Ah)
- Linear Curve Slope
- Information Entropy (Distribution-based)

## Climate-Aware Simulation

The project simulates how different ambient temperatures ($T_{amb}$) affect battery life using an **Arrhenius-based thermal aging model**.

### Simulation Logic:
1. **Temperature Profiles**: We use monthly average temperatures for **Anchorage (Cold)** and **Kuwait City (Hot)**.
2. **Acceleration Factor ($k$)**: Based on the Arrhenius equation:
   $$k = \exp\left(-\frac{E_a}{R} \cdot \left(\frac{1}{T_{amb}} - \frac{1}{T_{ref}}\right)\right)$$
   Where $E_a = 30,000$ J/mol and $T_{ref} = 25^\circ$C.
3. **Effective Cycle ($n_{eff}$)**: For each real cycle $n$, the model calculates an equivalent "physics cycle" $n_{eff} = n \cdot k$. This $n_{eff}$ is then used as the cycle input to the PINN, allowing the model trained on lab data to project SOH under real-world thermal stress.