import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", "-o", type=str)
    parser.add_argument("file", type=str)
    return parser.parse_args()


def plot_heatmap(q_table):
    """
    Qテーブルを可視化するヒートマップを作成
    各状態 (行) における行動 (列) のQ値を表示
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(q_table, cmap="coolwarm", annot=False, fmt=".2f")
    plt.title("Q-Table Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States")
    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()


args = parse_args()
Q_table = np.load(args.file)
plot_heatmap(Q_table)
