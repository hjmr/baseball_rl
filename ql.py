# 必要なライブラリをインポート
import gym
from gym import spaces
import numpy as np
import pandas as pd
import random


class BaseballEnv(gym.Env):
    def __init__(self, data):
        super(BaseballEnv, self).__init__()

        # ----------------------------------------------------------
        # 1) 空データかどうかをチェック
        # ----------------------------------------------------------
        if data.empty:
            raise ValueError("The input data is empty. Please provide a valid dataset.")

        # ----------------------------------------------------------
        # 2) データの保持
        #    - full_data: 学習に使う24万行すべて
        # ----------------------------------------------------------
        self.full_data = data.reset_index(drop=True)

        # ----------------------------------------------------------
        # 3) ユニーク状態の抽出 (drop_duplicates)
        #    状態空間 8変数だけを抽出 → ユニークな組合せを取得
        # ----------------------------------------------------------
        state_cols = [
            "runner_status1_numeric",
            "BallCountBefore_Corrected",
            "StrikeCountBefore_Corrected",
            "OutCountBefore_Corrected",
            "score_diff_class",
            "inning_class",
        ]
        unique_state_df = self.full_data[state_cols].drop_duplicates()

        # ----------------------------------------------------------
        # 4) 状態 -> インデックス のマッピングを作成
        #    例: (0, 1, 0, 2, 0, 1, 3, 0) -> 1234 (状態ID)
        # ----------------------------------------------------------
        self.state_to_index = {}
        for idx, row_values in enumerate(unique_state_df.values):
            # row_values は [batting_average_Class, ERA_Class, ..., inning_class] のndarray
            # タプルに変換して dictキーにする
            row_tuple = tuple(row_values)
            self.state_to_index[row_tuple] = idx

        # 状態数
        self.n_states = len(self.state_to_index)

        # ----------------------------------------------------------
        # 5) Gymのobservation_space と action_space を定義
        #    - 観測空間は「状態ID」の離散値（0 ~ n_states-1）
        #    - 行動空間は2離散 (Swing / Not Swing)
        # ----------------------------------------------------------
        self.observation_space = spaces.Discrete(self.n_states)  # 0 ~ n_states-1
        self.action_space = spaces.Discrete(2)  # 0: Not Swing, 1: Swing

        # ----------------------------------------------------------
        # 6) 学習の進行用変数
        # ----------------------------------------------------------
        self.current_step = 0
        self.previous_runner_status = 0
        self.current_state_index = None  # 離散状態ID

    def reset(self):
        """環境のリセット"""
        self.current_step = 0
        self.previous_runner_status = 0

        # 最初の行から状態IDを取得
        self.current_state_index = self._get_state_index(self.full_data.iloc[self.current_step])
        return self.current_state_index

    def collect_states(self, current_state):
        pass

    def step(self, action):
        """次のステップに進む"""
        if self.current_step >= len(self.full_data):
            raise IndexError("Step index out-of-bounds. Check data length and step logic.")

        # 現在の行データを取得
        row = self.full_data.iloc[self.current_step]
        reward = 0

        # ----------------------------------------------------------
        # 7) 行動に対する報酬 (例: Swingで得点があれば加算)
        #    ※数値は例示で変更している
        # ----------------------------------------------------------
        if action == 1:  # Swing
            reward += row["score_reward"] * 30  # 得点 x 30 の報酬

        # ----------------------------------------------------------
        # 8) ランナー増加時の報酬
        # ----------------------------------------------------------
        if row["runner_status1_numeric"] > self.previous_runner_status:
            reward += 9  # ランナーが増えた場合の報酬
        self.previous_runner_status = row["runner_status1_numeric"]

        # ----------------------------------------------------------
        # 9) 試合終了時の追加報酬 (最終行で勝敗判定)
        # ----------------------------------------------------------
        done = False
        if self.current_step == len(self.full_data) - 1:
            done = True
            if row["home_or_away_attacking"] == 1:  # ホームチームが攻撃
                reward += 90 if row["home_reward"] > 0 else -45
            else:  # アウェイチームが攻撃
                reward += 90 if row["away_reward"] > 0 else -45

        # ----------------------------------------------------------
        # 10) 次のステップへ
        # ----------------------------------------------------------
        self.current_step += 1
        if self.current_step < len(self.full_data):
            next_row = self.full_data.iloc[self.current_step]
            next_state_index = self._get_state_index(next_row)
        else:
            # データの終端に達した場合
            next_state_index = self.current_state_index

        # 状態を更新
        self.current_state_index = next_state_index

        return next_state_index, reward, done, {}

    def _get_state_index(self, row):
        """
        特定の行(row)から状態を示すタプルを生成し,
        そのタプルに対応する離散状態IDを返す
        """
        state_tuple = (
            row["runner_status1_numeric"],
            row["BallCountBefore_Corrected"],
            row["StrikeCountBefore_Corrected"],
            row["OutCountBefore_Corrected"],
            row["score_diff_class"],
            row["inning_class"],
        )
        return self.state_to_index[state_tuple]


# ---------------------------
# ここからメインの学習コード
# ---------------------------
# データの読み込み
input_file_path = "classified_data.csv"  # 実際のファイルパスに合わせる
data = pd.read_csv(input_file_path)

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
env = BaseballEnv(data_combined)

# Qテーブルの初期化
# 「n_states」はユニークな状態数（最大8640程度）
n_states = env.n_states
n_actions = env.action_space.n
Q_table = np.zeros((n_states, n_actions))

# パラメータ設定
alpha = 0.1  # 学習率
gamma = 0.99  # 割引率
epsilon = 0.1  # ε-greedy の探索率
n_episodes = 300  # テスト的に少数 → 実際は150などに増やす

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
            action = np.argmax(Q_table[state_idx])  # 最大Q値を持つ行動

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

    print(f"Episode {episode + 1}/{n_episodes}: Total Reward: {total_reward}")

# Qテーブルの保存
output_q_table = "/content/q_table_updated3_essential.npy"
np.save(output_q_table, Q_table)
print(f"Q-learning completed. Q-table saved to {output_q_table}")

# ダウンロードリンクの作成 (Colab環境用)
files.download(output_q_table)
