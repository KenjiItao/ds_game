import numpy as np
import pandas as pd
import random
import sys
import os
import math
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt
# from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from scipy.spatial.distance import cdist

plt.switch_backend('agg')
sns.set(style='whitegrid')
current_palette = sns.color_palette("colorblind", 7)
if True:
    # "#0072b2", "#f0e442", "#009e73", "#d55e00", "#cc79a7"
    current_palette[0] = (255 / 255, 0 / 255, 0 / 255)
    current_palette[1] = (0 / 255, 48 / 255, 163 / 255)
    current_palette[2] = (5 / 255, 163 / 255, 0 / 255)
    current_palette[3] = (213 / 255, 94 / 255, 0 / 255)
    current_palette[4] = (204 / 255, 121 / 255, 167 / 255)
    current_palette[5] = (75 / 255, 0 / 255, 146 / 255)
    current_palette[6] = (0 / 255, 114 / 255, 178 / 255)

import warnings
warnings.filterwarnings('ignore')

class Player:
    def __init__(self, c, d):
        self.state = np.random.normal(1, 0.1)
        self.b = 1
        self.c = c
        self.d = d
        self.e = 0
        self.fitness = 0

def payoff(resource, action1, action2):
    resource = 2 * resource - resource ** 2
    next_resource = resource / 3 ** (action1 + action2)
    if action1 + action2 > 0:
        payoff = (resource - next_resource) / (action1 + action2)
    else:
        payoff = 0
    return next_resource, payoff * action1, payoff * action2

def game_step(player1, player2):
    resource = .1
    action1_ls, action2_ls, state1_ls, state2_ls, resource_ls = [], [], [], [], []
    for i in range(num_steps):
        state1 = player1.state
        state2 = player2.state
        action1 = (resource + player1.c * state1 + player1.d * state2 > 0) * 1
        action2 = (resource + player2.c * state2 + player2.d * state1 > 0) * 1
        resource, payoff1, payoff2 = payoff(resource, action1, action2)
        player1.state = 0.8 * player1.state + payoff1
        player2.state = 0.8 * player2.state + payoff2
        action1_ls.append(action1)
        action2_ls.append(action2)
        state1_ls.append(state1)
        state2_ls.append(state2)
        resource_ls.append(resource)
    # return action1_ls[100:], action2_ls[100:], state1_ls[100:], state2_ls[100:], resource_ls[100:]
    return action1_ls, action2_ls, state1_ls, state2_ls, resource_ls

def phase_check(action_freq, fitness1, fitness2):
    if action_freq > 0.3:
       return 0
    elif 0.24 < fitness1 < 0.35 and 0.24 < fitness2 < 0.35:
        return 1
    elif 0.35 < fitness1 < 0.44 and 0.35 < fitness2 < 0.44:
        return 2
    elif 0.45 < fitness1 < 0.5 and 0.45 < fitness2 < 0.5:
        return 3
    elif fitness1 > 0.5 and fitness2 > 0.5:
        return 4
    elif action_freq < 0.1:
        return -1
    else:
        return -2

def plot_gamesteps(action1_ls, action2_ls, resource_ls, filepath, window_size=30):
    # プロットする範囲を設定
    x_values = list(range(num_steps))[-window_size:]

    # プレイヤー1のデータ
    action1 = np.array(action1_ls[-window_size:])
    y_values_player1 = np.full(len(x_values), 1.0)

    # プレイヤー2のデータ
    action2 = np.array(action2_ls[-window_size:])
    y_values_player2 = np.full(len(x_values), 0.95)

    # plt.figure(figsize=(10, 6))
    plt.figure()

    # プレイヤー1のアクションをプロット
    mask_filled = action1 == 1
    mask_unfilled = action1 == 0
    plt.scatter(np.array(x_values)[mask_filled], y_values_player1[mask_filled],
                marker='s', s=100, color='blue', label="Action 1", edgecolors='blue')
    plt.scatter(np.array(x_values)[mask_unfilled], y_values_player1[mask_unfilled],
                marker='s', s=100, facecolors='none', edgecolors='blue')

    # プレイヤー2のアクションをプロット
    mask_filled = action2 == 1
    mask_unfilled = action2 == 0
    plt.scatter(np.array(x_values)[mask_filled], y_values_player2[mask_filled],
                marker='s', s=100, color='red', label="Action 2", edgecolors='red')
    plt.scatter(np.array(x_values)[mask_unfilled], y_values_player2[mask_unfilled],
                marker='s', s=100, facecolors='none', edgecolors='red')

    # リソースの時間変化をプロット
    plt.plot(x_values, resource_ls[-window_size:], color='green', label="Resource", linewidth=2)

    # 軸の設定
    # plt.xlabel('Time Steps', fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 凡例の設定
    # plt.legend(loc='upper right', fontsize=12)

    # プロットの範囲を調整
    plt.ylim(-0.05, 1.05)  # アクションのY軸範囲
    plt.xlim(x_values[0], x_values[-1])

    # プロットを保存
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def plot_first_gamesteps(action1_ls, action2_ls, resource_ls, filepath, window_size=40):
    # プロットする範囲を設定
    x_values = list(range(num_steps))[:window_size]

    # プレイヤー1のデータ
    action1 = np.array(action1_ls[:window_size])
    y_values_player1 = np.full(len(x_values), 1.0)

    # プレイヤー2のデータ
    action2 = np.array(action2_ls[:window_size])
    y_values_player2 = np.full(len(x_values), 0.95)

    # plt.figure(figsize=(10, 6))
    plt.figure()

    # プレイヤー1のアクションをプロット
    mask_filled = action1 == 1
    mask_unfilled = action1 == 0
    plt.scatter(np.array(x_values)[mask_filled], y_values_player1[mask_filled],
                marker='s', s=70, color='blue', label="Action 1", edgecolors='blue')
    plt.scatter(np.array(x_values)[mask_unfilled], y_values_player1[mask_unfilled],
                marker='s', s=70, facecolors='none', edgecolors='blue')

    # プレイヤー2のアクションをプロット
    mask_filled = action2 == 1
    mask_unfilled = action2 == 0
    plt.scatter(np.array(x_values)[mask_filled], y_values_player2[mask_filled],
                marker='s', s=70, color='red', label="Action 2", edgecolors='red')
    plt.scatter(np.array(x_values)[mask_unfilled], y_values_player2[mask_unfilled],
                marker='s', s=70, facecolors='none', edgecolors='red')

    # リソースの時間変化をプロット
    plt.plot(x_values, resource_ls[:window_size], color='green', label="Resource", linewidth=2)

    # 軸の設定
    # plt.xlabel('Time Steps', fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 凡例の設定
    # plt.legend(loc='upper right', fontsize=12)

    # プロットの範囲を調整
    plt.ylim(-0.05, 1.05)  # アクションのY軸範囲
    plt.xlim(x_values[0], x_values[-1])

    # プロットを保存
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def generation(players, iteration):
    random.shuffle(players)
    action_ls = []
    resource_ls_ls = []
    ave_action_ls =[]
    ave_resource_ls = []
    phase_ls = []
    phase_cur = [0, 0, 0, 0, 0, 0]
    for i in range(0, len(players) - 1, 2):
        action1_ls, action2_ls, state1_ls, state2_ls, resource_ls = game_step(players[i], players[i + 1])
        fitness1 = np.mean(state1_ls)
        fitness2 = np.mean(state2_ls)
        players[i].fitness = fitness1 + 0.001
        players[i + 1].fitness = fitness2 + 0.001
        action_ls.extend(action1_ls)
        action_ls.extend(action2_ls)
        resource_ls_ls.extend(resource_ls)
        action_freq = (sum(action1_ls) + sum(action2_ls)) / len(action1_ls) / 2
        phase = phase_check(action_freq, fitness1, fitness2)
        if not phase == -2:
            phase_ls.append(phase)
            phase_cur[phase + 1] += 1

        ave_action_ls.append(np.mean(action1_ls))
        ave_action_ls.append(np.mean(action2_ls))
        ave_resource_ls.append(np.mean(resource_ls))

        # if i == 0 and iteration % 100 == 0:
        if i == 0 and iteration % 10 == 0:
            plot_gamesteps(action1_ls, action2_ls, resource_ls, f"figs/timeseries/{path}_step{iteration}_{round(np.mean(state1_ls) * 100)}pc_{round(np.mean(state2_ls) * 100)}pc.pdf")
            plot_first_gamesteps(action1_ls, action2_ls, resource_ls, f"figs/timeseries/{path}_step{iteration}_first_{round(np.mean(state1_ls) * 100)}pc_{round(np.mean(state2_ls) * 100)}pc.pdf")
            #
            # Histplot of players' decision-making parameters
            plt.figure()
            # plt.hist([player.b for player in players], bins=20, color='blue', alpha=0.5, label=r'$b$')
            plt.hist([player.c for player in players], bins=20, color=color1, alpha=0.5, label=r'$c$')
            plt.hist([player.d for player in players], bins=20, color=color2, alpha=0.5, label=r'$d$')
            # plt.hist([player.e for player in players], bins=20, color='purple', alpha=0.5, label=r'$e$')
            # plt.legend()
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(f"figs/timeseries/{path}_step{iteration}_hist.pdf", bbox_inches='tight')
            plt.close()
    fitness_ls = [player.fitness for player in players]
    normalizer = sum(fitness_ls) / num_players
    next_players = []
    for player in players:
        num_offspring = np.random.poisson(player.fitness / normalizer)
        for i in range(num_offspring):
            # b = player.b + np.random.normal(0, mutation)
            c = player.c + np.random.normal(0, mutation)
            d = player.d + np.random.normal(0, mutation)
            # e = player.e + np.random.normal(0, mutation)
            next_players.append(Player(c, d))
        # mean can be median
        ave_acts = np.mean(ave_action_ls)
        ave_resource = np.mean(ave_resource_ls)
        ave_fitness = np.mean(fitness_ls)
        mean_b = np.mean([player.b for player in players])
        mean_c = np.mean([player.c for player in players])
        mean_d = np.mean([player.d for player in players])
        mean_e = np.mean([player.e for player in players])
        if len(phase_ls) > 0:
            phase = scipy.stats.mode(phase_ls)[0]
        else:
            phase = -1
    if len(next_players) == 0:
        next_players = players
    if len(next_players) == 1:
        next_players.append(next_players[0])
    return next_players, ave_acts, ave_resource, ave_fitness, mean_b, mean_c, mean_d, mean_e, phase, phase_cur


def main():
    players = [Player(np.random.normal(0, 0), np.random.normal(0, 0)) for _ in range(num_players)]
    act_ls, resource_ls, fitness_ls = [], [], []
    b_ls, c_ls, d_ls, e_ls = [], [], [], []
    phase_ls = []
    for iteration in range(num_generations):
        players, ave_acts, ave_resource, ave_fitness, mean_b, mean_c, mean_d, mean_e, phase, phase_cur = generation(players, iteration)
        act_ls.append(ave_acts)
        resource_ls.append(ave_resource)
        fitness_ls.append(ave_fitness)
        b_ls.append(mean_b)
        c_ls.append(mean_c)
        d_ls.append(mean_d)
        e_ls.append(mean_e)
        # phase = scipy.stats.mode(phase_ls)[0]
        phase_ls.append(phase)
        # if iteration % 100 == 0:
        #     print(ave_acts, ave_resource, ave_fitness, ave_phase_shift, ave_action_period, ave_resource_period, phase_cur)

    plt.figure()
    plt.plot(act_ls, color='blue', label="action")
    plt.plot(resource_ls, color='red', label="resource")
    plt.plot(fitness_ls, color='green', label="fitness")
    # plt.legend()
    plt.savefig(f"figs/{path}_tot.pdf")
    plt.close()

    plt.figure()
    # plt.plot(b_ls, color='blue', alpha=0.8, label=r'$b$')
    plt.plot(c_ls, color=color1, alpha=0.8, label=r'$S$')
    plt.plot(d_ls, color=color2, alpha=0.8, label=r'$O$')
    # plt.plot(e_ls, color='purple', alpha=0.8, label=r'$e$')
    # plt.legend()
    plt.savefig(f"figs/{path}_tot_strategies.pdf")
    plt.close()

    df = pd.DataFrame(
        {"action": act_ls, "resource": resource_ls, "fitness": fitness_ls, "phase": phase_ls,
         "b": b_ls, "c": c_ls, "d": d_ls, "e": e_ls})
    df.to_csv(f"res/{path}.csv")

num_steps = 1000
num_players = 300
num_generations = 3000
trial = 0
mutation = 0.03
color1 = current_palette[3]
color2 = current_palette[5]
game_mode = "original"

if not os.path.exists(f"res"):
    os.mkdir(f"res")
    os.mkdir(f"figs")
    os.mkdir(f"figs/timeseries")


# trial = int(sys.argv[2])
# mutation = [0.01, 0.03, 0.1, 0.3][int(sys.argv[1]) // 3]
# for num_players in [[10, 30, 100, 300], [1000], [3000]][int(sys.argv[1]) % 3]:
#     path = f"original_N{num_players}_{num_steps}steps_{num_generations}generations_mu{round(mutation * 1000)}pm_{trial}"
#     main()

for trial in range(5):
    path = f"original_N{num_players}_{num_steps}steps_{num_generations}generations_mu{round(mutation * 1000)}pm_{trial}"
    main()
