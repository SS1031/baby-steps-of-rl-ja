import random
import numpy as np


class CoinToss:

    def __init__(self, head_probs, max_episode_steps=30):
        self.head_probs = head_probs  # 表が出る確率
        self.max_episode_steps = max_episode_steps  # コイントスを行う回数
        self.toss_count = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        """選択したコインでの試行、表が出た場合は報酬が1
        """
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception("The step count exceeded maximum. \
                            Please reset env.")
        else:
            done = True if self.toss_count == final else False

        if action >= len(self.head_probs):
            raise Exception("The No.{} coin doesn't exist.".format(action))
        else:
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done


class EpsilonGreedyAgent:
    """Epsilon-Greedy法に基づき行動するエージェント
    確率epsilonで探索、それ以外のときは活用
    - 探索 = ランダムにコインを投げる
    - 活用 = これまでの試行の期待値に基づいて期待値最大のコインを投げる
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.V = []

    def policy(self):
        coins = range(len(self.V))
        if random.random() < self.epsilon:
            return random.choice(coins)  # 探索
        else:
            return np.argmax(self.V)  # 活用

    def play(self, env):
        """コイントスをプレイする処理
        """
        # Initialize estimation.
        N = [0] * len(env)  # 各コインを投げた回数を記録
        self.V = [0] * len(env)  # 各コインの期待値（ステップごとに更新

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1  # 回数の更新
            self.V[selected_coin] = new_average  # 期待値の更新

        return rewards


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt


    def main():
        env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
        epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
        game_steps = list(range(10, 310, 10))
        result = {}
        for e in epsilons:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = s
                rewards = agent.play(env)
                means.append(np.mean(rewards))
            result["epsilon={}".format(e)] = means
        result["coin toss count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("coin toss count", drop=True, inplace=True)
        result.plot.line(figsize=(10, 5))
        plt.show()


    main()
