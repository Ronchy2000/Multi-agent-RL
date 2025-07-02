import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import platform
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

def set_font_for_plot():
    """根据平台动态设置字体"""
    system_platform = platform.system()
    print("system_platform:", system_platform)
    if system_platform == "Darwin":  # MacOS
        font = 'Arial Unicode MS'
    elif system_platform == "Windows":  # Windows
        font = 'SimHei'
    else:  # Linux
        # 中文字体需要手动安装
        # 参考：https://blog.csdn.net/takedachia/article/details/131017286  https://blog.csdn.net/weixin_45707277/article/details/118631442
        font = 'SimHei'
    
    plt.rcParams['font.sans-serif'] = [font]
    plt.rcParams['axes.unicode_minus'] = False

def plot_all_rewards(csv_file, window_size=50):
    """在一张图上绘制所有智能体的奖励曲线（包括追捕者和逃避者）"""
    df = pd.read_csv(csv_file)
    set_font_for_plot()
    
    # 打印CSV文件的列名，帮助调试
    print(f"CSV文件列名: {df.columns.tolist()}")
    
    # 获取数据点数量，动态调整窗口大小
    data_points = len(df)
    print(f"数据点数量: {data_points}")
    
    # 如果数据点数量小于窗口大小，则调整窗口大小为数据点数量的一半
    if data_points < window_size:
        window_size = max(2, data_points // 2)  # 确保窗口大小至少为2
        print(f"数据点不足，调整窗口大小为: {window_size}")
    
    # 从CSV文件名中提取时间戳
    base_name = os.path.basename(csv_file)
    if 'rewards_' in base_name and '.csv' in base_name:
        timestamp = base_name.replace('rewards_', '').replace('.csv', '')
    else:
        timestamp = ''
    
    # 创建一个包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个子图：所有智能体的奖励曲线
    # 修改：适配CSV文件列名
    adversary_col = 'Adversary_Mean' if 'Adversary_Mean' in df.columns else 'Adversary_Mean_Reward'
    agent_columns = [col for col in df.columns if col not in ['Episode', adversary_col]]
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_columns)))
    
    # 绘制每个智能体的奖励曲线
    for agent, color in zip(agent_columns, colors):
        # 原始数据（半透明）
        ax1.plot(df['Episode'], df[agent], color=color, alpha=0.2, label=f'{agent} (原始)')
        # 移动平均
        ma_data = moving_average(df[agent].values, window_size)
        # 确保x轴和y轴数据长度匹配
        x_data = df['Episode'][window_size-1:window_size-1+len(ma_data)]
        ax1.plot(x_data, ma_data, 
                color=color, linewidth=2, label=f'{agent} (移动平均)')
    
    ax1.set_title('所有智能体奖励曲线')
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('奖励')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 第二个子图：追捕者平均奖励
    # 修改：适配CSV文件列名
    # 原始数据（半透明）
    ax2.plot(df['Episode'], df[adversary_col], 
            'gray', alpha=0.2, label='原始数据')
    # 移动平均
    adv_ma = moving_average(df[adversary_col].values, window_size)
    # 确保x轴和y轴数据长度匹配
    x_data = df['Episode'][window_size-1:window_size-1+len(adv_ma)]
    ax2.plot(x_data, adv_ma, 
            'r-', linewidth=2, label='移动平均')
    
    ax2.set_title('追捕者平均奖励趋势')
    ax2.set_xlabel('回合数')
    ax2.set_ylabel('平均奖励')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图片
    if timestamp:
        save_path = os.path.join(os.path.dirname(csv_file), f'training_rewards_{timestamp}.png')
    else:
        save_path = os.path.join(os.path.dirname(csv_file), 'training_rewards.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练奖励图像已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 修改：指定具体的CSV文件名
    csv_file = os.path.join(os.path.dirname(__file__), 'xxxx.csv') # 替换为你的CSV文件名
    print("csv_file name:", csv_file)

    if os.path.exists(csv_file):
        plot_all_rewards(csv_file)
    else:
        print(f"错误：未找到CSV文件：{csv_file},请检查路径及文件名是否正确!")