# 使用 sys.executable 获取当前虚拟环境的 pip，这样它会始终使用当前虚拟环境的 pip 安装包，而不是系统环境的 pip
import pkg_resources
import sys
import platform
import os
from subprocess import call

def check_and_install_pettingzoo():
    # 打印当前虚拟环境的相关信息
    print("================================")
    print(f"Current Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current virtual environment: {sys.prefix}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print("================================")

    try:
        # 检查 pettingzoo 是否已经安装
        pkg_resources.get_distribution("pettingzoo")
        print("================================")
        print("pettingzoo is already installed.")
        print("================================")
    except pkg_resources.DistributionNotFound:
        # 如果 pettingzoo 没有安装，执行安装操作
        print("================================")
        print("pettingzoo is not installed. Installing pettingzoo...")
        print("================================")
        
        # 获取当前虚拟环境的 Python 解释器路径
        python_executable = sys.executable
        
        # 根据操作系统确定 pip 路径
        if platform.system() == "Windows":
            # Windows 系统下，pip 通常在 Scripts 目录下
            pip_executable = os.path.join(os.path.dirname(python_executable), "Scripts", "pip.exe")
        else:
            # macOS/Linux 系统下
            pip_dir = os.path.dirname(python_executable)
            pip_executable = os.path.join(pip_dir, "pip")
            if not os.path.exists(pip_executable):
                pip_executable = python_executable.replace("python", "pip")
        
        print(f"Using pip executable: {pip_executable}")

        # 尝试安装 pettingzoo==1.24.4
        try:
            print("Attempting to install pettingzoo==1.24.4...")
            result = call([pip_executable, "install", "pettingzoo==1.24.4"])
            if result == 0:
                print("================================")
                print("Successfully installed pettingzoo==1.24.4")
                print("================================")
            else:
                print("Installation of pettingzoo==1.24.4 failed. Trying GitHub installation...")
                # 如果安装失败，尝试从 GitHub 安装
                try:
                    # 根据操作系统调整命令格式
                    if platform.system() == "Windows":
                        # Windows 下不使用引号
                        result = call([pip_executable, "install", "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"])
                    else:
                        # macOS/Linux 下使用引号
                        result = call([pip_executable, "install", "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"])
                    
                    if result == 0:
                        print("================================")
                        print("Successfully installed pettingzoo from GitHub.")
                        print("================================")
                    else:
                        print("GitHub installation failed. Please check the error above.")
                except Exception as e:
                    print(f"Failed to install pettingzoo from GitHub: {e}")
                    print("================================")
                    print("Please manually install pettingzoo or check the error above.")
        except Exception as e:
            print(f"Failed to install pettingzoo==1.24.4: {e}")
            print("Attempting to install pettingzoo from GitHub...")

if __name__ == "__main__":
    check_and_install_pettingzoo()