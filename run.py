# run.py
# 项目启动脚本
# 从项目根目录运行此脚本来执行SA-MOO攻击
# 默认使用同目录下的config.yaml参数
# 现在使用合并的单文件版本

import sys
from pathlib import Path

# 导入并运行主函数
from samoo_merged import main

if __name__ == "__main__":
    main()