import os
from datetime import datetime

# 定义文件头信息
file_header = """# -*- coding: utf-8 -*-
# @Time : {time}
# @Author : Ronchy Lu
# @Email : rongqi1949@gmail.com
# @File : {filename}
# @Software: PyCharm
# @Description: None
"""

# 获取当前目录下所有的 .py 文件
current_directory = os.getcwd()
python_files = [f for f in os.listdir(current_directory) if f.endswith('.py')]

for file in python_files:
    # 读取原始内容
    with open(file, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # 创建新的文件头
    file_header_formatted = file_header.format(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename=file
    )

    # 写入新的内容
    with open(file, 'w', encoding='utf-8') as f:
        f.write(file_header_formatted + original_content)

print("文件头信息已添加到所有 .py 文件")
