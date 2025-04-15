import os
import json
import torch
from datetime import datetime
import uuid

"""
1. 在logs/下保存了一个 training_log.json 文件，它包含了训练的所有参数和日志信息。
2. 保存的 plot_data_{current_time.replace(':', '-')}.pkl 是一个 PyTorch 保存的文件，它并不包含模型本身，而是 训练过程中的奖励数据。

"""
class TrainingLogger:
    def __init__(self, log_dir = None, module_name = None):
        # 始终基于文件位置构建绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # utils目录
        project_root = os.path.dirname(current_dir)  # 项目根目录

        # 如果指定了模块名，则在logs下创建对应子目录
        if module_name:
            default_log_dir = os.path.join('logs', module_name)
        else:
            default_log_dir = 'logs'

        if log_dir is None:
            # 默认在项目根目录下创建logs目录
            self.log_dir = os.path.join(project_root, default_log_dir)
        elif os.path.isabs(log_dir):
            # 如果提供了绝对路径，直接使用
            self.log_dir = log_dir
        else:
            # 如果提供了相对路径，则相对于项目根目录解析
            self.log_dir = os.path.join(project_root, log_dir)

        # 确保目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        print(f"日志将保存到: {self.log_dir}")
        
    def save_training_log(self, args, device, start_time, end_time, training_duration, runner):
        """
        保存训练日志，为每次运行创建单独的日志文件
        """
        # 生成唯一的运行ID
        run_id = self.generate_run_id()
        
        # 获取当前时间作为日志记录时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 将 NumPy 数组转换为列表
        episode_rewards_dict = {}
        for agent_id, rewards in runner.episode_rewards.items():
            episode_rewards_dict[agent_id] = rewards.tolist()

        # 准备训练日志信息
        log_info = {
            "run_id": run_id,  # 添加运行ID
            "训练开始时间": start_time,
            "训练结束时间": end_time,
            "训练用时": training_duration,
            "训练设备": str(device),
            "环境名称": args.env_name,
            "渲染模式": args.render_mode,
            "随机种子": args.seed,
            "使用可变随机种子": args.use_variable_seeds if hasattr(args, 'use_variable_seeds') else False,
            "总回合数": args.episode_num,
            "每回合步数": args.episode_length,
            "评估回合数": args.evaluate_episode_num if hasattr(args, 'evaluate_episode_num') else 100,
            "学习间隔": args.learn_interval,
            "随机步数": args.random_steps,
            "tau": args.tau,
            "gamma": args.gamma,
            "buffer容量": args.buffer_capacity,
            "batch_size": args.batch_size,
            "actor学习率": args.actor_lr,
            "critic学习率": args.critic_lr,
            "comm学习率": args.comm_lr if hasattr(args, 'comm_lr') else None,
            "通信网络参数-输出维度": args.message_dim if hasattr(args, 'message_dim') else None,
            "是否使用visdom": args.visdom,
            "visdom窗口大小": args.size_win,
        }

        # 1. 保存此次运行的单独配置文件
        run_config_dir = os.path.join(self.log_dir, "run_configs")
        os.makedirs(run_config_dir, exist_ok=True)
        run_config_file = os.path.join(run_config_dir, f"config_{run_id}.json")
        
        with open(run_config_file, 'w', encoding='utf-8') as f:
            json.dump(log_info, f, ensure_ascii=False, indent=4)
        
        print(f"本次运行配置已保存到: {run_config_file}")
        
        # 2. 同时更新主日志文件
        log_file = os.path.join(self.log_dir, "training_log.json")
        
        # 读取现有的日志文件，如果存在的话
        existing_logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        existing_logs = json.loads(content)
            except Exception as e:
                print(f"Warning: Error reading log file: {str(e)}. Creating new log file.")
                existing_logs = []

        # 确保 existing_logs 是列表类型
        if not isinstance(existing_logs, list):
            existing_logs = []

        # 添加本次运行信息及运行ID (便于关联到详细配置)
        summary_info = {
            "run_id": run_id,
            "训练开始时间": start_time,
            "训练结束时间": end_time,
            "训练用时": training_duration,
            "环境名称": args.env_name,
            "渲染模式": args.render_mode,
            "随机种子": args.seed,
            "使用可变随机种子": args.use_variable_seeds if hasattr(args, 'use_variable_seeds') else True,
            "总回合数": args.episode_num,
            "config_file": run_config_file  # 添加配置文件路径
        }
        existing_logs.append(summary_info)
        
        # 保存更新后的日志文件
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving log file: {str(e)}")
            # 尝试备份旧文件并创建新文件
            backup_file = log_file + '.backup'
            if os.path.exists(log_file):
                os.rename(log_file, backup_file)
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump([summary_info], f, ensure_ascii=False, indent=4)

        # 3. 保存训练曲线数据，使用运行ID关联
        plot_data = {
            "run_id": run_id,
            "episode_rewards": episode_rewards_dict,
            "adversary_mean_rewards": runner.all_adversary_mean_rewards,
            "timestamps": end_time
        }
        
        plot_file = os.path.join(self.log_dir, f"plot_data_{run_id}.pkl")
        torch.save(plot_data, plot_file)
        print(f"训练曲线数据已保存到: {plot_file}")
        
        # 4. 添加配置导出功能 - 将当前config导出为可运行的Python文件
        self.export_config_as_python(args, run_id)
        
        return run_id  # 返回运行ID供其他地方使用

    def export_config_as_python(self, args, run_id):
        """
        导出当前配置为可直接使用的Python文件，格式与原始config.py一致
        """
        config_dir = os.path.join(self.log_dir, "run_configs")
        os.makedirs(config_dir, exist_ok=True)  # 确保目录存在
        
        config_file = os.path.join(config_dir, f"config_{run_id}.py")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("import argparse\n\n")
            f.write("def parse_args():\n")
            f.write("    \"\"\"解析命令行参数，如果未指定则使用默认值\"\"\"\n")
            f.write("    parser = argparse.ArgumentParser(\"MADDPG + 通信\")\n")
            f.write("    \n")
            f.write("    # 环境参数\n")
            f.write("    parser.add_argument(\"--env_name\", type=str, default=\"{}\", \n".format(args.env_name))
            f.write("                       choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3', 'simple_tag_env'])\n")
            f.write("    parser.add_argument(\"--render_mode\", type=str, default=\"{}\", help=\"None | human | rgb_array\")\n".format(args.render_mode))
            f.write("    parser.add_argument(\"--episode_num\", type=int, default={}, help=\"训练轮数\")\n".format(args.episode_num))
            f.write("    parser.add_argument(\"--episode_length\", type=int, default={}, help=\"每轮最大步数\")\n".format(args.episode_length))
            f.write("    parser.add_argument(\"--evaluate_episode_num\", type=int, default={}, help=\"评估轮数\")\n".format(
                args.evaluate_episode_num if hasattr(args, 'evaluate_episode_num') else 100))
            
            # 处理种子 - 确保-1表示不使用固定种子
            seed_value = -1 if args.seed is None else args.seed
            f.write("    parser.add_argument(\"--seed\", type=int, default={}, help=\"随机种子 (使用-1表示不使用固定种子)\")\n".format(seed_value))
            
            # 添加可变随机种子参数
            use_variable_seeds = getattr(args, 'use_variable_seeds', False)  # 获取属性，如果不存在则默认为False
            f.write("    parser.add_argument(\"--use_variable_seeds\", type=bool, default={}, help=\"使用可变随机种子\")\n".format(use_variable_seeds))

            f.write("    \n")
            f.write("    # MADDPG参数\n")
            f.write("    parser.add_argument(\"--gamma\", type=float, default={}, help=\"折扣因子\")\n".format(args.gamma))
            f.write("    parser.add_argument(\"--tau\", type=float, default={}, help=\"软更新参数\")\n".format(args.tau))
            f.write("    parser.add_argument(\"--buffer_capacity\", type=int, default={}, help=\"经验回放缓冲区容量\")\n".format(args.buffer_capacity))
            f.write("    parser.add_argument(\"--batch_size\", type=int, default={}, help=\"批次大小\")\n".format(args.batch_size))
            f.write("    parser.add_argument(\"--actor_lr\", type=float, default={}, help=\"Actor学习率\")\n".format(args.actor_lr))
            f.write("    parser.add_argument(\"--critic_lr\", type=float, default={}, help=\"Critic学习率\")\n".format(args.critic_lr))
            
            # 添加comm_lr参数，如果存在
            comm_lr = getattr(args, 'comm_lr', 0.00001)
            f.write("    parser.add_argument(\"--comm_lr\", type=float, default={}, help=\"Comm学习率\")\n".format(comm_lr))
            
            f.write("    parser.add_argument(\"--learn_interval\", type=int, default={}, help=\"学习间隔步数\")\n".format(args.learn_interval))
            f.write("    parser.add_argument(\"--random_steps\", type=int, default={}, help=\"初始随机探索步数\")\n".format(args.random_steps))
            
            # 通信网络参数
            message_dim = getattr(args, 'message_dim', 3)
            f.write("    # 通信网络参数\n")
            f.write("    parser.add_argument(\"--message_dim\", type=int, default={}, help=\"通信消息维度\")\n".format(message_dim))
            f.write("    \n")
            
            # 可视化参数 - 使用action="store_true"
            f.write("    # 可视化参数\n")
            if args.visdom:
                f.write("    parser.add_argument(\"--visdom\", action=\"store_true\", help=\"是否使用visdom可视化\")\n")
            else:
                f.write("    parser.add_argument(\"--visdom\", action=\"store_true\", help=\"是否使用visdom可视化\")\n")
                
            f.write("    parser.add_argument(\"--size_win\", type=int, default={}, help=\"平滑窗口大小\")\n".format(args.size_win))
            f.write("    \n")  
            # 训练设备
            device = getattr(args, 'device', 'cpu')
            f.write("    # 训练设备\n")
            f.write("    parser.add_argument(\"--device\", type=str, default='{}', help=\"训练设备\")\n".format(device))
            f.write("    \n")
            f.write("    # 解析参数\n")
            f.write("    args = parser.parse_args([])\n")
            f.write("    \n")
            f.write("    # 如果seed为-1，则设置为None\n")
            f.write("    if args.seed == -1:\n")
            f.write("        args.seed = None\n\n")
            f.write("    return args\n\n")
            f.write("def get_config():\n")
            f.write("    \"\"\"获取配置，优先使用命令行参数，其次使用默认配置\"\"\"\n")
            f.write("    return parse_args()\n")
        
        print(f"配置文件已导出为标准格式: {config_file}")


    def generate_run_id(self):
        """生成唯一运行ID：时间戳+随机字符"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        random_part = str(uuid.uuid4())[:4]
        return f"{timestamp}_{random_part}"

    def compare_configs(self, run_id1, run_id2):
        """比较两次运行的配置差异"""
        config_dir = os.path.join(self.log_dir, "run_configs")
        # 加载两个配置
        config1_file = os.path.join(config_dir, f"config_{run_id1}.json")
        config2_file = os.path.join(config_dir, f"config_{run_id2}.json")

        if not os.path.exists(config1_file) or not os.path.exists(config2_file):
            print("指定的配置文件不存在")
            return

        with open(config1_file, 'r', encoding='utf-8') as f:
            config1 = json.load(f)

        with open(config2_file, 'r', encoding='utf-8') as f:
            config2 = json.load(f)

        # 找出差异
        print(f"比较运行 {run_id1} 和 {run_id2} 的配置差异:")
        print("=" * 50)
        diff_found = False

        # 所有键的并集
        all_keys = set(config1.keys()) | set(config2.keys())

        for key in sorted(all_keys):
            # 检查键是否存在于两个配置中
            if key not in config1:
                print(f"{key}: 仅存在于第二个配置 - {config2[key]}")
                diff_found = True
            elif key not in config2:
                print(f"{key}: 仅存在于第一个配置 - {config1[key]}")
                diff_found = True
            # 比较值是否相同
            elif config1[key] != config2[key]:
                print(f"{key}: {config1[key]} -> {config2[key]}")
                diff_found = True

        if not diff_found:
            print("两个配置完全相同")
        print("=" * 50)