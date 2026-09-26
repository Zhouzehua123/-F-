# 本绘图程序在人工智能工具 Codex 辅助下完成。
# 开发机构：OpenAI；模型：GPT-5.6-Luna（2026-04-21）。
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
