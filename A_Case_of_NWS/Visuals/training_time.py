import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Data from the provided image ---
# Nf values (Col A)
Nf_values = [
    1000, 2000, 3000, 4000, 5000, 
    6000, 7000, 8000, 9000, 10000, 
    11000, 12000, 13000, 14000, 15000
]

# Wall-clock time (s) (Col B)
time_s = [
    3.863395, 3.706938, 3.673115, 4.616278, 4.448315, 
    4.723784, 4.998482, 5.427092, 5.869342, 6.054514, 
    6.341763, 6.726422, 7.162635, 7.552636, 7.982853
]

# Final loss (Col C)
final_loss = [
    0.000395, 3.43E-05, 5.25E-05, 0.000691, 1.64E-05, 
    0.000161, 4.41E-05, 3.37E-05, 9.98E-06, 1.82E-05, 
    1.78E-05, 2.32E-05, 0.000147, 0.000399, 0.00047
]

# Create a DataFrame to hold the results
df = pd.DataFrame({
    "Nf": Nf_values, 
    "time_s": time_s, 
    "final_loss": final_loss
})

# Assuming the epochs_timing was 1000 for this run, as in the PINN code
epochs_timing = 1000 

# --- Plotting Code (Mirrors the style in Cell 39 of your notebook) ---

## Plotting Timing vs. Nf
print("\nPlotting Wall-clock Time vs. Collocation Points (Nf)...")
plt.figure(figsize=(7, 4))
plt.plot(df['Nf'], df['time_s'], marker='o', label='Measured Time')

# Polynomial fit (linear fit, degree=1)
coeffs = np.polyfit(df['Nf'], df['time_s'], 1)
fit_line = np.poly1d(coeffs)(df['Nf'])
plt.plot(df['Nf'], fit_line, linestyle='--', color='red', 
         label=f"Fit: {coeffs[0]:.4e} s/pt * Nf + {coeffs[1]:.2f}")

plt.xlabel("Number of collocation points (Nf)")
plt.ylabel("Time (s)")
plt.title(f"PINN Training Time Scaling (Epochs={epochs_timing})")
plt.legend()
plt.grid(True)
plt.show()

# You can save the plot to a file if needed:
# plt.savefig(\"pinn_timing_vs_nf_data.png\")


## Plotting Final Loss vs. Nf
print("\nPlotting Final Loss vs. Collocation Points (Nf)...")
plt.figure(figsize=(7, 4))
# Use semilogy due to the small values of final loss
plt.semilogy(df['Nf'], df['final_loss'], marker='o', label='Final Total Loss') 

plt.xlabel("Number of collocation points (Nf)")
plt.ylabel("Final Total Loss")
plt.title(f"PINN Final Loss vs. Nf (Epochs={epochs_timing})")
plt.legend()
plt.grid(True, which="both")
plt.show()

# You can save the plot to a file if needed:
# plt.savefig(\"pinn_final_loss_vs_nf_data.png\")