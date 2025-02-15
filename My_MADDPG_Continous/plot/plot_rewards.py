import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
'''
注意：
作者用pands==2.2.3出错了。
pip install pandas==2.2.1 没问题。
'''

def moving_average(data, window_size=50):
    """简单移动平均"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def exponential_moving_average(data, alpha=0.1):
    """指数移动平均"""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

# def plot_rewards(csv_file, window_size=50, alpha=0.1):
#     # 读取CSV文件，不指定数据类型
#     df = pd.read_csv(csv_file)
#     # 设置中文字体（如果需要显示中文）
#     plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
#     plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    
#         # 计算平滑后的数据
#     adv_ma = moving_average(df['Adversary Average Reward'].values)
#     adv_ema = exponential_moving_average(df['Adversary Average Reward'].values)
#     sum_ma = moving_average(df['Sum Reward of All Agents'].values)
#     sum_ema = exponential_moving_average(df['Sum Reward of All Agents'].values)
    
#     # 创建图形
#     plt.figure(figsize=(15, 10))
    
#     # 绘制追捕者平均奖励
#     plt.subplot(2, 1, 1)
#     plt.plot(df['Episode'], df['Adversary Average Reward'], 'lightgray', alpha=0.3, label='原始数据')
#     plt.plot(df['Episode'][window_size-1:], adv_ma, 'b-', linewidth=2, label='移动平均')
#     plt.plot(df['Episode'], adv_ema, 'r-', linewidth=2, label='指数移动平均')
#     plt.title('追捕者平均奖励随回合数变化')
#     plt.xlabel('回合数')
#     plt.ylabel('平均奖励')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
    
#     # 绘制所有智能体总奖励
#     plt.subplot(2, 1, 2)
#     plt.plot(df['Episode'], df['Sum Reward of All Agents'], 'lightgray', alpha=0.3, label='原始数据')
#     plt.plot(df['Episode'][window_size-1:], sum_ma, 'b-', linewidth=2, label='移动平均')
#     plt.plot(df['Episode'], sum_ema, 'r-', linewidth=2, label='指数移动平均')
#     plt.title('所有智能体总奖励随回合数变化')
#     plt.xlabel('回合数')
#     plt.ylabel('总奖励')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
    
#     # 调整子图之间的间距
#     plt.tight_layout()
    
#     # 保存图片
#     save_path = os.path.join(os.path.dirname(csv_file), f'rewards_plot.png')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"图片已保存至 {save_path}")
    
#     # 显示图形
#     plt.show()

def different_plot_rewards(csv_file, window_size=50, alpha=0.1):
    df = pd.read_csv(csv_file)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算平滑后的数据
    adv_ma = moving_average(df['Adversary Average Reward'].values, window_size)
    adv_ema = exponential_moving_average(df['Adversary Average Reward'].values, alpha)
    sum_ma = moving_average(df['Sum Reward of All Agents'].values, window_size)
    sum_ema = exponential_moving_average(df['Sum Reward of All Agents'].values, alpha)
    
    # 创建两个图形
    # 1. 移动平均对比图
    plt.figure(figsize=(15, 10))
    # 追捕者奖励
    plt.subplot(2, 1, 1)
    plt.plot(df['Episode'], df['Adversary Average Reward'], 'lightgray', alpha=0.3, label='原始数据')
    plt.plot(df['Episode'][window_size-1:], adv_ma, 'b-', linewidth=2, label='移动平均')
    plt.title('追捕者平均奖励 - 移动平均对比')
    plt.xlabel('回合数')
    plt.ylabel('平均奖励')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 总奖励
    plt.subplot(2, 1, 2)
    plt.plot(df['Episode'], df['Sum Reward of All Agents'], 'lightgray', alpha=0.3, label='原始数据')
    plt.plot(df['Episode'][window_size-1:], sum_ma, 'b-', linewidth=2, label='移动平均')
    plt.title('所有智能体总奖励 - 移动平均对比')
    plt.xlabel('回合数')
    plt.ylabel('总奖励')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存移动平均对比图
    save_path_ma = os.path.join(os.path.dirname(csv_file), f'rewards_plot_ma.png')
    plt.savefig(save_path_ma, dpi=300, bbox_inches='tight')
    
    # 2. 指数移动平均对比图
    plt.figure(figsize=(15, 10))
    # 追捕者奖励
    plt.subplot(2, 1, 1)
    plt.plot(df['Episode'], df['Adversary Average Reward'], 'lightgray', alpha=0.3, label='原始数据')
    plt.plot(df['Episode'], adv_ema, 'r-', linewidth=2, label='指数移动平均')
    plt.title('追捕者平均奖励 - 指数移动平均对比')
    plt.xlabel('回合数')
    plt.ylabel('平均奖励')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 总奖励
    plt.subplot(2, 1, 2)
    plt.plot(df['Episode'], df['Sum Reward of All Agents'], 'lightgray', alpha=0.3, label='原始数据')
    plt.plot(df['Episode'], sum_ema, 'r-', linewidth=2, label='指数移动平均')
    plt.title('所有智能体总奖励 - 指数移动平均对比')
    plt.xlabel('回合数')
    plt.ylabel('总奖励')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存指数移动平均对比图
    save_path_ema = os.path.join(os.path.dirname(csv_file), f'rewards_plot_ema.png')
    plt.savefig(save_path_ema, dpi=300, bbox_inches='tight')
    
    print(f"移动平均对比图已保存至 {save_path_ma}")
    print(f"指数移动平均对比图已保存至 {save_path_ema}")
    
    plt.show()

if __name__ == "__main__":
    # CSV文件路径（相对于当前脚本的路径）
    csv_file = os.path.join(os.path.dirname(__file__), 'data', 'data_rewards.csv')
    print("csv_file name:",csv_file)

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # print(df.head())
        # plot_rewards(csv_file)
        different_plot_rewards(csv_file)
    else:
        print(f"错误：未找到CSV文件：{csv_file}")