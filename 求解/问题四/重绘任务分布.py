# 本绘图程序参考 GPT-5.6-Luna 辅助完成。
# 辅助工具：GPT-5.6-Luna。
# 仅读取既有结果，不重新拟合或修改结果数据。
"""兼容原命令入口，统一使用问题四的子任务分布图绘制逻辑。"""
from 重绘图表 import hashes, style, subtasks


def main():
    before = hashes()
    style()
    subtasks(audit=None)
    assert hashes() == before


if __name__ == '__main__':
    main()
