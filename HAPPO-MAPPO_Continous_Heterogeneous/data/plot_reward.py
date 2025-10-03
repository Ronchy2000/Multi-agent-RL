import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
import re

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

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
        # 修改：支持自动查找最新的happo奖励文件
        csv_files = glob.glob(os.path.join(data_dir, "happo_rewards_simple_tag_v3_n1_s23_2025-09-24_20-12.csv"))
        if not csv_files:
            print(f"Error: No HAPPO reward CSV files found in directory {data_dir}")
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
    
    algorithm = "HAPPO"  # 默认算法名
    env_name = "simple_tag_v3"
    # 从文件名中提取种子值
    seed_match = re.search(r"_s(\d+)_", filename)
    if seed_match:
        seed = seed_match.group(1)
    
    # Create chart with better styling
    plt.figure(figsize=(12, 8))
    plt.plot(df['Steps'], df['Reward'], marker='o', linestyle='-', markersize=3, linewidth=2, alpha=0.8)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Evaluation Reward', fontsize=12)
    
    # 修改：使用英文标题，更清晰的格式
    plt.title(f'{algorithm} Learning Curve | Env: {env_name} | Seed: {seed}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add statistics and trend lines
    if len(df) > 1:
        # Average line
        avg_reward = df['Reward'].mean()
        plt.axhline(y=avg_reward, color='red', linestyle='--', alpha=0.6, linewidth=2,
                   label=f'Average: {avg_reward:.2f}')
        
        # Add moving average line (if enough points)
        if len(df) >= 5:
            window_size = min(max(len(df) // 10, 3), 10)  # 动态窗口大小
            df['MA'] = df['Reward'].rolling(window=window_size, min_periods=1).mean()
            plt.plot(df['Steps'], df['MA'], color='green', linestyle='-', linewidth=2,
                    alpha=0.7, label=f'Moving Average ({window_size} points)')
        
        # Add final performance indicator
        final_rewards = df['Reward'].tail(5).mean()  # 最后5个点的平均值
        plt.axhline(y=final_rewards, color='purple', linestyle=':', alpha=0.6, linewidth=2,
                   label=f'Final Performance: {final_rewards:.2f}')
    
    # Add performance statistics text box
    if len(df) > 1:
        max_reward = df['Reward'].max()
        min_reward = df['Reward'].min()
        std_reward = df['Reward'].std()
        
        stats_text = f'Max: {max_reward:.2f}\nMin: {min_reward:.2f}\nStd: {std_reward:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=10)
    
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    # Save chart
    if save:
        plots_dir = os.path.join(current_dir)
        os.makedirs(plots_dir, exist_ok=True)
        # 修改：更清晰的文件名
        plt_filename = os.path.join(plots_dir, f"{algorithm.lower()}_learning_curve_{env_name}_s{seed}.png")
        plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {plt_filename}")
    
    # Display chart
    if show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HAPPO training reward curves")
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