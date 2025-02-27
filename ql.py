# 必要なライブラリをインポート
import numpy as np
import pandas as pd
import random
import argparse

from baseball_env import BaseballEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", "-n", type=int, default=1000)
    parser.add_argument("--alpha", "-a", type=float, default=0.1)
    parser.add_argument("--gamma", "-g", type=float, default=0.99)
    parser.add_argument("--epsilon", "-e", type=float, default=0.1)
    parser.add_argument("--input_file_path", "-i", type=str, default="classified_data.csv")
    parser.add_argument("--output_q_table", "-o", type=str, default="q_table.npy")
    return parser.parse_args()


# ---------------------------
# ここからメインの学習コード
# ---------------------------
args = parse_args()

# データの読み込み
data = pd.read_csv(args.input_file_path)

# 必要な列（状態 + 報酬計算用）を抽出
columns_for_state = [
    "runner_status1_numeric",
    "BallCountBefore_Corrected",
    "StrikeCountBefore_Corrected",
    "OutCountBefore_Corrected",
    "score_diff_class",
    "inning_class",
]
columns_for_action = ["swing_flag"]
columns_for_reward = [
    "score_reward",
    "home_reward",
    "away_reward",
    "home_or_away_attacking",
]

# 結合して学習用データを用意
data_state = data[columns_for_state]
data_action = data[columns_for_action]
data_reward = data[columns_for_reward]
data_combined = pd.concat([data_state, data_action, data_reward], axis=1)

# 環境の初期化
env = BaseballEnv(data_combined, columns_for_state, columns_for_action)

# Qテーブルの初期化
# 「n_states」はユニークな状態数（最大8640程度）
n_states = env.n_states
n_actions = env.action_space.n
Q_table = np.zeros((n_states, n_actions))

# パラメータ設定
alpha = args.alpha  # 学習率
gamma = args.gamma  # 割引率
epsilon = args.epsilon  # ε-greedy の探索率
n_episodes = args.num_episodes  # エピソード数

# Q学習アルゴリズム
for episode in range(n_episodes):
    # 環境リセット
    state_idx = env.reset()
    total_reward = 0
    done = False

    while not done:
        # ε-greedy 方策による行動選択
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # ランダム
        else:
            action = int(np.argmax(Q_table[state_idx]))  # 最大Q値を持つ行動

        # 環境との相互作用: step(action) → (次状態, 報酬, 終了フラグ, info)
        next_state_idx, reward, done, _ = env.step(action)

        # Q値の更新
        Q_table[state_idx, action] = Q_table[state_idx, action] + alpha * (
            reward + gamma * np.max(Q_table[next_state_idx]) - Q_table[state_idx, action]
        )

        # 報酬を累積
        total_reward += reward

        # 状態を更新
        state_idx = next_state_idx

    print(f"Episode {episode + 1}/{n_episodes} : Steps {env.current_step} : Total Reward {total_reward}")

# Qテーブルの保存
np.save(args.output_q_table, Q_table)
print(f"Q-learning completed. Q-table saved to {args.output_q_table}")
