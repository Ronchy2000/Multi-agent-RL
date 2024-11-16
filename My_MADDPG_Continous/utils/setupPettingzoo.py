import pkg_resources
from subprocess import call

def check_and_install_pettingzoo():
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
        
        # 尝试安装 pettingzoo==1.24.4
        try:
            print("Attempting to install pettingzoo==1.24.4...")
            result = call(["pip", "install", "pettingzoo==1.24.4"])
            if result == 0:
                print("================================")
                print("Successfully installed pettingzoo==1.24.4")
                print("================================")
            else:
                print("Installation of pettingzoo==1.24.4 failed. Trying GitHub installation...")
                # 如果安装失败，尝试从 GitHub 安装
                try:
                    result = call(["pip", "install", "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"])
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