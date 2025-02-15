import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
'''
注意：
作者用pands==2.2.3出错了。
pip install pandas==2.2.1 没问题。
'''
def plot_rewards(csv_file):
    # 读取CSV文件，不指定数据类型
    df = pd.read_csv(csv_file)
    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制追捕者平均奖励
    plt.subplot(2, 1, 1)
    plt.plot(df['Episode'], df['Adversary Average Reward'], 'b-', linewidth=1, label='追捕者平均奖励')
    plt.title('追捕者平均奖励随回合数变化')
    plt.xlabel('回合数')
    plt.ylabel('平均奖励')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 绘制所有智能体总奖励
    plt.subplot(2, 1, 2)
    plt.plot(df['Episode'], df['Sum Reward of All Agents'], 'r-', linewidth=1, label='总体奖励')
    plt.title('所有智能体总奖励随回合数变化')
    plt.xlabel('回合数')
    plt.ylabel('总奖励')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_path = os.path.join(os.path.dirname(csv_file), f'rewards_plot_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至 {save_path}")
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    # CSV文件路径（相对于当前脚本的路径）
    csv_file = os.path.join(os.path.dirname(__file__), 'data', 'data_rewards.csv')
    print("csv_file:",csv_file)
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(df.head())
        plot_rewards(csv_file)
    else:
        print(f"错误：未找到CSV文件：{csv_file}")