import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import sys
from tqdm import tqdm


class BaseballEnv(gym.Env):
    def __init__(self, data, state_cols, action_cols):
        super(BaseballEnv, self).__init__()

        # ----------------------------------------------------------
        # 空データかどうかをチェック
        # ----------------------------------------------------------
        if data.empty:
            raise ValueError("The input data is empty. Please provide a valid dataset.")

        # ----------------------------------------------------------
        # データの保持
        #    - full_data: 学習に使う24万行すべて
        # ----------------------------------------------------------
        self.full_data = data.reset_index(drop=False)

        # ----------------------------------------------------------
        # ユニーク状態の抽出 (drop_duplicates)
        #    状態空間 8変数だけを抽出 → ユニークな組合せを取得
        # ----------------------------------------------------------
        self.state_cols = state_cols
        self.state_set = self.full_data[self.state_cols].drop_duplicates()

        # ----------------------------------------------------------
        # ユニークアクションの抽出 (drop_duplicates)
        # ----------------------------------------------------------
        self.action_set = self.full_data[action_cols].drop_duplicates()
        self.n_actions = len(self.action_set)

        # ----------------------------------------------------------
        # 状態 -> インデックス のマッピングを作成
        #    例: (0, 1, 0, 2, 0, 1, 3, 0) -> 1234 (状態ID)
        # ----------------------------------------------------------
        self.state_to_index = {}
        for idx, row_values in enumerate(self.state_set.values):
            # row_values は [batting_average_Class, ERA_Class, ..., inning_class] のndarray
            # タプルに変換して dictキーにする
            row_tuple = tuple(row_values)
            self.state_to_index[row_tuple] = idx

        # 状態数
        self.n_states = len(self.state_to_index)

        # ----------------------------------------------------------
        # 高速化のため：各データの状態indexを予め計算
        # ----------------------------------------------------------
        self.full_data["state_index"] = self.full_data.apply(self._get_state_index, axis=1)

        # ----------------------------------------------------------
        # Gymのobservation_space と action_space を定義
        #    - 観測空間は「状態ID」の離散値（0 ~ n_states-1）
        #    - 行動空間は2離散 (Swing / Not Swing)
        # ----------------------------------------------------------
        self.observation_space = spaces.Discrete(self.n_states)  # 0 ~ n_states-1
        self.action_space = spaces.Discrete(self.n_actions)  # 0: Not Swing, 1: Swing

        # ----------------------------------------------------------
        # 学習の進行用変数
        # ----------------------------------------------------------
        self.current_step = 0
        self.previous_runner_status = 0
        self.current_state_index = None  # 離散状態ID
        self.reset()

    def reset(self):
        """環境のリセット"""
        self.current_step = 0
        self.previous_runner_status = 0

        # 初期状態をランダムに決定
        self.current_state_index = random.randint(0, self.n_states - 1)
        return self.current_state_index

    def step(self, action):
        # 現在の状態を取得
        reward = 0

        # データから現在の状態＋行動に対応する行を収集
        next_state_candidates = self.full_data[
            (self.full_data["state_index"] == self.current_state_index) & (self.full_data["swing_flag"] == action)
        ]["index"].values
        if len(next_state_candidates) == 0:
            print(
                f"Warning: No next state candidates found for state: {self.current_state_index} and action: {action}. Skip to the next episode.",
                file=sys.stderr,
            )
            return self.current_state_index, -1, True, {}  # penalty

        # ランダムに1つを選択
        next_row_index = random.choice(next_state_candidates)
        next_row = self.full_data.iloc[next_row_index]

        # 状態IDが変更になるまで次の行を取得
        while next_row["state_index"] == self.current_state_index:
            next_row_index += 1
            try:
                next_row = self.full_data.iloc[next_row_index]
            except IndexError:
                print(
                    f"Warning: No more rows found for state: {self.current_state_index} and action: {action}. Skip to the next episode.",
                    file=sys.stderr,
                )
                return self.current_state_index, -1, True, {}  # penalty

        # ----------------------------------------------------------
        # 行動に対する報酬 (例: Swingで得点があれば加算)
        #    ※数値は例示で変更している
        # ----------------------------------------------------------
        reward += next_row["score_reward"]  # 得点の報酬

        # ----------------------------------------------------------
        # ランナー増加時の報酬
        # ----------------------------------------------------------
        # if next_row["runner_status1_numeric"] > self.previous_runner_status:
        #    reward += 1  # ランナーが増えた場合の報酬
        # self.previous_runner_status = next_row["runner_status1_numeric"]

        # ----------------------------------------------------------
        # ゲームの終了判定
        # ----------------------------------------------------------
        done = False
        # ゲームの終了が分からないので，適当な回数のSTEPで終了させる。
        if self.current_step >= random.randint(1000, 3000):
            done = True

        # ----------------------------------------------------------
        # 次のステップへ
        # ----------------------------------------------------------
        self.current_step += 1
        self.current_state_index = next_row["state_index"]

        return self.current_state_index, reward, done, {}

    def _get_state_index(self, row):
        """
        特定の行(row)から状態を示すタプルを生成し,
        そのタプルに対応する離散状態IDを返す
        """
        state_tuple = tuple(row[self.state_cols].values)
        return self.state_to_index[state_tuple]
