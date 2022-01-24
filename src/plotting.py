import numpy as np
import matplotlib.pyplot as plt

def bar_plot(data, title='', xlabel='', ylabel='', legend_labels=['DDPG', 'DQN'], colors=['r','b'], 
                save_fig=False, file_name='', file_location='./src/figures/'):
    """
        This function is used to create a bar plot for plotting the completion rate
        Inputs
            data: list of lists of lists
                ->  [
                        algorithm 1 [value1, value2, ..., valueN],
                        algorithm 2 [value1, value2, ..., valueN]
                    ]
            title (string): name of the figure
            xlabel (string): label of x-axis
            ylabel (string): label of y-axis
            legend_labels (list): the names of the algorithms for the legend(make sure to use the right order)
            colors (list): the colors of the respective algorithms
            save_fig (boolean): saves the figure
            file_name (string): name of the file
            file_location (string): target location of the file             
    """
    fig, ax = plt.subplots()

    new_data = [x.sum()/x.shape[0] for x in data]

    ax.bar(legend_labels, new_data)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_fig:
        plt.savefig(file_location + file_name + '.png', bbox_inches='tight')

    plt.legend()
    plt.show()

def line_plot_simple(data, title='', xlabel='', ylabel='', legend_labels=['DDPG', 'DQN'], colors=['r','b'], 
                save_fig=False, file_name='', file_location='./src/figures/'):
    """
        This function is used to create a line plot, for example for plotting the collected food and reward per episode.
        Inputs
            data: list of lists
                ->  [
                        algorithm 1 [value1, value2, ..., valueN],
                        algorithm 2 [value1, value2, ..., valueN]
                    ]
            title (string): name of the figure
            xlabel (string): label of x-axis
            ylabel (string): label of y-axis
            legend_labels (list): the names of the algorithms for the legend(make sure to use the right order)
            colors (list): the colors of the respective algorithms
            save_fig (boolean): saves the figure
            file_name (string): name of the file
            file_location (string): target location of the file             
    """
    fig, ax = plt.subplots()

    for i, line in enumerate(data):
        plt.plot(line, colors[i], label=legend_labels[i])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_fig:
        plt.savefig(file_location + file_name + '.png', bbox_inches='tight')

    plt.legend()
    plt.show()

def line_plot_advanced(data, title='', xlabel='', ylabel='', legend_labels=['DDPG', 'DQN'], colors=['r','b'], 
                save_fig=False, file_name='', file_location='./src/figures/'):
    """
        This function is used to create a line plot with the standard deviation, 
        for example for plotting the collected food and reward per episode.
        Inputs
            data: list of list
                ->  [
                        algorithm 1 [
                            Trial 1 [value1, value2, ..., valueN],
                            Trial 2 [value1, value2, ..., valueN],
                            Trial N [value1, value2, ..., valueN],
                            ]

                        algorithm 2 [
                            Trial 1 [value1, value2, ..., valueN],
                            Trial 2 [value1, value2, ..., valueN],
                            Trial N [value1, value2, ..., valueN],
                            ]
                    ]
            title (string): name of the figure
            xlabel (string): label of x-axis
            ylabel (string): label of y-axis
            legend_labels (list): the names of the algorithms for the legend(make sure to use the right order)
            colors (list): the colors of the respective algorithms
            save_fig (boolean): saves the figure
            file_name (string): name of the file
            file_location (string): target location of the file             
    """
    fig, ax = plt.subplots()

    for i, line in enumerate(data):
        x = np.arange(len(line[0]))
        mean = np.mean(line, axis=0)
        std = np.std(line, axis=0)
        plt.plot(x, mean, colors[i], label=legend_labels[i])
        plt.fill_between(x, mean-std, mean+std, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_fig:
        plt.savefig(file_location + file_name + '.png', bbox_inches='tight')

    plt.legend()
    plt.show()

def main():
    x = np.random.choice(a=[False, True], size=100, p=[0.6, 0.4])
    y = np.random.choice(a=[False, True], size=100, p=[0.7, 0.3])    
    bar_plot([x,y])

    x = [7, 14, 21, 28, 35, 42, 49]
    y = [5, 12, 19, 21, 31, 27, 35]
    line_plot_simple([x,y])

    E = np.random.normal((3, 5, 4), (1.25, 1.00, 0.25), (100, 3))
    D = np.random.normal((6, 8, 2), (2, 3, 1), (100, 3))
    line_plot_advanced([E, D])

if __name__ == "__main__":
    main()
