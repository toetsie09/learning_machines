import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from scipy.signal import savgol_filter

def list_to_numpy(data):
    new_data = [np.array(i) for i in data]
    return np.array(new_data)

def count_food(data):
    new_data = [np.count_nonzero(np.array(i) == 100) for i in data]
    return np.array(new_data)

def count_survived(data, food_mask, max_food, n_steps):
    new_data = []
    for i in data:
        if food_mask == max_food:
            new_data.append(n_steps)
        else:
            new_data.append(len(i))

    # new_data = [len(i) if np.count_nonzero(np.array(i) == 100) == max_food else n_steps for i in data]
    return np.array(new_data)

def training_line_plot(rewards, food, title, xlabel, ylabels):
    fig, ax = plt.subplots(2, sharex=True)

    x = np.arange(300)
    sum_reward = [np.sum(row) for row in rewards]
    # mean_reward = [np.mean(row) for row in rewards]
    ax[0] = sns.lineplot(x, savgol_filter(sum_reward, 51, 3), ax=ax[0], color='blue')
    ax[0] = sns.lineplot(x, sum_reward, ax=ax[0], alpha=0.5, color='darkblue')
    # ax[0] = sns.lineplot(x, mean_reward, ax=ax[0])
    ax[1] = sns.lineplot(x, savgol_filter(food, 51, 3), ax=ax[1], color='green')
    ax[1] = sns.barplot(x, food, saturation=0.75, ax=ax[1], color='darkgreen', alpha=0.5)

    fig.suptitle(title)
    # ax.set_xlabel(xlabel)
    ax[0].set_ylabel(ylabels[0])
    ax[1].set_ylabel(ylabels[1])
    ax[1].set_xlabel(xlabel)
    fig.align_ylabels(ax[:])
    plt.xticks([0, 50, 100, 150, 200, 250, 300])
    plt.show()

def validation_box_plot(DDPG, DQN, title):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    ax1 = sns.boxplot(data=[DQN[0], DDPG[0]], ax=ax1)
    ax1.set_xticklabels(['DQN', 'DDPG'])
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Survival rate')

    sum_reward_dqn = [np.sum(row) for row in DQN[1]]
    sum_reward_ddpg = [np.sum(row) for row in DDPG[1]]

    ax2 = sns.boxplot(data=[sum_reward_dqn, sum_reward_ddpg], ax=ax2)
    ax2.set_xticklabels(['DQN', 'DDPG'])
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Reward')

    ax3 = sns.boxplot(data=[DQN[2], DDPG[2]], ax=ax3)
    ax3.set_xticklabels(['DQN', 'DDPG'])
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Collected Food')
    fig.suptitle(title)
    plt.show()
    print()


# def validation_line_plot(DQN, DDPG, title, xlabel, ylabel):
#     fig, ax = plt.subplots()

#     x = np.arange(50)
#     sum_reward_dqn = [np.sum(row) for row in DQN]
#     sum_reward_ddpg = [np.sum(row) for row in DDPG]
#     # mean_reward = [np.mean(row) for row in rewards]
#     ax = sns.lineplot(x, sum_reward_dqn, ax=ax, alpha=0.8, color='blue')
#     ax = sns.lineplot(x, sum_reward_ddpg, ax=ax, alpha=0.8, color='green')

#     fig.suptitle(title)
#     # ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_xlabel(xlabel)
#     # plt.xticks([0, 50, 100, 150, 200, 250, 300])
#     plt.tight_layout()
#     plt.show()


def main():
    path_dqn = './src/_task2/DQN/results/'
    path_ddpg = './src/_task2/DDPG/results/'

    # with open(path_dqn + 'DQN_training_rewards_v1.pkl', 'rb') as file:
    #     rewards_training_DQN = list_to_numpy(pickle.load(file))
    #     padding = np.zeros(29)
    #     rewards_training_DQN = np.append(rewards_training_DQN, padding)
    #     foods_training_DQN = count_food(rewards_training_DQN)

    # with open(path_ddpg + 'DDPG_training_rewards_final.pkl', 'rb') as file:
    #     rewards_training_DDPG = list_to_numpy(pickle.load(file))

    # with open(path_ddpg + 'DDPG_training_collected_foods.pkl', 'rb') as file:
    #     foods_training_DDPG = list_to_numpy(pickle.load(file))

    # # line_plot_advanced([food_training_DDPG, rewards_training_DDPG])
    # training_line_plot(rewards_training_DDPG, foods_training_DDPG, title='Training DDPG', xlabel='#Episodes', ylabels=['Reward', 'Food Collected'])

    # training_line_plot(rewards_training_DQN, foods_training_DQN, title='Training DQN', xlabel='#Episodes', ylabels=['Reward', 'Food Collected'])

    with open(path_dqn + 'DQN_test_collected_foods_v1.pkl', 'rb') as file:
        foods_test_DQN = pickle.load(file)

    with open(path_ddpg + 'DDPG_evaluation_collected_foods.pkl', 'rb') as file:
        foods_test_DDPG = pickle.load(file)

    with open(path_dqn + 'DQN_test_rewards_v1.pkl', 'rb') as file:
        rewards_test_DQN = pickle.load(file)
    survival_rate_DQN = count_survived(rewards_test_DQN, foods_test_DQN, 7, 300)

    with open(path_ddpg + 'DDPG_evaluation_rewards.pkl', 'rb') as file:
        rewards_test_DDPG = pickle.load(file)
    survival_rate_DDPG = count_survived(rewards_test_DDPG, foods_test_DDPG, 8, 300)

    validation_box_plot([survival_rate_DDPG, rewards_test_DDPG, foods_test_DDPG], [survival_rate_DQN, rewards_test_DQN, foods_test_DQN], title='')
    # validation_line_plot(rewards_test_DQN, rewards_test_DDPG, title='', xlabel='#Episodes', ylabel='Reward')
    print()

if __name__ == "__main__":
    main()
