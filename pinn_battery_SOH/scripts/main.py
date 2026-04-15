import os
import argparse
from nature_preprocess import load_nature_dataset
from train import train_nature_pinn
from simulate import simulate_climates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    # 1. Preprocess
    print("Pre-processing...")
    if not os.path.exists("nature_processed.csv"):
        load_nature_dataset(
            "/mnt/disk4/GitHub/Battery-degration-pinn/cleaned_dataset/metadata.csv",
            "/mnt/disk4/GitHub/Battery-degration-pinn/cleaned_dataset/data"
        )
    
    # 2. Train
    print("Training...")
    train_nature_pinn("nature_processed.csv", epochs=args.epochs)
    
    # 3. Simulate
    print("Simulating...")
    simulate_climates(
        'checkpoints/nature_best_model.pth',
        'checkpoints/nature_stats.json',
        'climate_profiles.csv'
    )
    
    print("Done. Results in sim_results.png")

if __name__ == "__main__":
    main()
