#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def convert_gif_to_loop(input_path, output_path=None, backup=True):
    """
    将GIF转换为循环播放的GIF
    
    参数:
        input_path: 输入GIF文件路径或包含GIF文件的目录
        output_path: 输出GIF文件路径或目录，默认为在原文件名后添加"_loop"
        backup: 是否备份原始文件
    """
    # 检查是否安装了PIL
    if not PIL_AVAILABLE:
        print("错误: 未安装PIL/Pillow库。请使用以下命令安装:")
        print("pip install Pillow")
        return
        
    input_path = Path(input_path)
    
    # 检查输入路径是文件还是目录
    if input_path.is_file() and input_path.suffix.lower() == '.gif':
        gif_files = [input_path]
    elif input_path.is_dir():
        gif_files = list(input_path.glob('*.gif'))
    else:
        print(f"错误: {input_path} 不是有效的GIF文件或目录")
        return
    
    if not gif_files:
        print(f"在 {input_path} 中没有找到GIF文件")
        return
    
    for gif_file in gif_files:
        # 确定输出文件路径
        if output_path is None:
            output_file = gif_file.parent / f"{gif_file.stem}_loop{gif_file.suffix}"
        elif Path(output_path).is_dir():
            output_file = Path(output_path) / f"{gif_file.stem}_loop{gif_file.suffix}"
        else:
            output_file = Path(output_path)
        
        # 备份原始文件
        if backup and gif_file.exists():
            backup_file = gif_file.parent / f"{gif_file.stem}_original{gif_file.suffix}"
            if not backup_file.exists():
                print(f"备份 {gif_file} 到 {backup_file}")
                subprocess.run(['cp', str(gif_file), str(backup_file)])
        
        # 使用PIL/Pillow转换GIF为循环播放
        print(f"转换 {gif_file} 为循环播放GIF: {output_file}")
        
        try:
            # 打开GIF文件
            img = Image.open(gif_file)
            
            # 提取所有帧
            frames = []
            durations = []
            
            try:
                while True:
                    # 记录当前帧的持续时间
                    durations.append(img.info.get('duration', 100))  # 默认100ms
                    # 复制当前帧
                    frames.append(img.copy())
                    # 尝试移动到下一帧
                    img.seek(img.tell() + 1)
            except EOFError:
                pass  # 到达文件末尾
            
            # 保存为循环播放的GIF
            if frames:
                frames[0].save(
                    str(output_file),
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=durations,
                    loop=0  # 0表示无限循环
                )
                print(f"成功创建循环播放GIF: {output_file}")
            else:
                print(f"警告: {gif_file} 似乎不是有效的GIF动画")
                
        except Exception as e:
            print(f"处理 {gif_file} 时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将GIF转换为循环播放的GIF')
    parser.add_argument('input', help='输入GIF文件路径或包含GIF文件的目录')
    parser.add_argument('-o', '--output', help='输出GIF文件路径或目录，默认为在原文件名后添加"_loop"')
    parser.add_argument('--no-backup', action='store_false', dest='backup', 
                        help='不备份原始文件')
    
    args = parser.parse_args()
    convert_gif_to_loop(args.input, args.output, args.backup)