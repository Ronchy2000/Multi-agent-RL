import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse

def plot_rewards(file_path=None, data_dir=None, show=True, save=True):
    """
    Plot training reward curves
    
    Parameters:
    file_path: Specified CSV file path. If None, use the latest CSV file
    data_dir: Directory containing CSV files
    show: Whether to display the chart
    save: Whether to save the chart as PNG file
    """
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If data directory not specified, use default data subdirectory
    if data_dir is None:
        data_dir = os.path.join(current_dir)
    elif not os.path.isabs(data_dir):
        # If a relative path is provided, convert to absolute path
        data_dir = os.path.join(current_dir, data_dir)
    
    # Confirm directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # If file not specified, find the latest CSV file
    if file_path is None:
        csv_files = glob.glob(os.path.join(data_dir, "mappo_rewards_simple_spread_v3_n2_s23_2025-09-23_03-19.csv"))
        if not csv_files:
            print(f"Error: No CSV files found in directory {data_dir}")
            return
        file_path = max(csv_files, key=os.path.getctime)  # Select the latest file
    elif not os.path.isabs(file_path):
        # If a relative path is provided, convert to absolute path
        file_path = os.path.join(current_dir, file_path)
    
    print(f"Loading data file: {file_path}")
    
    # Read CSV data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return
    
    # Extract filename information
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    if len(parts) >= 5:  # Ensure there are enough parts to extract information
        env_name = parts[2]
        agents = parts[3][1:]  # Remove 'n'
        seed = parts[4][1:]    # Remove 's'
    else:
        env_name = "unknown"
        agents = "?"
        seed = "?"
    
    # Create chart
    plt.figure(figsize=(10, 6))
    plt.plot(df['Steps'], df['Reward'], marker='o', linestyle='-', markersize=4)
    plt.xlabel('Training Steps')
    plt.ylabel('Evaluation Reward')
    plt.title(f'MAPPO Training Progress - Env:{env_name} Agents:{agents} Seed:{seed}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add average line and trend line
    if len(df) > 1:
        plt.axhline(y=df['Reward'].mean(), color='r', linestyle='--', alpha=0.5, 
                   label=f'Average: {df["Reward"].mean():.2f}')
        
        # Add moving average line (if enough points)
        if len(df) >= 3:
            df['MA'] = df['Reward'].rolling(window=3, min_periods=1).mean()
            plt.plot(df['Steps'], df['MA'], color='green', linestyle='-', 
                    label='Moving Average (3 points)')
    
    plt.legend()
    
    # Save chart
    if save:
        plots_dir = os.path.join(current_dir)
        os.makedirs(plots_dir, exist_ok=True)
        plt_filename = os.path.join(plots_dir, f"mappo_learning_curve_{env_name}_n{agents}_s{seed}.png")
        plt.savefig(plt_filename)
        print(f"Chart saved to: {plt_filename}")
    
    # Display chart
    if show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MAPPO training reward curves")
    parser.add_argument("--file", type=str, default=None, 
                       help="CSV file path to plot. If not specified, use the latest CSV file")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing CSV data files")
    parser.add_argument("--no-show", action="store_false", dest="show",
                       help="Do not display the chart")
    parser.add_argument("--no-save", action="store_false", dest="save",
                       help="Do not save the chart")
    
    args = parser.parse_args()
    plot_rewards(args.file, args.data_dir, args.show, args.save)