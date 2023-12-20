# for plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from env import Action
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def draw_animation(agent, animation_name='training_animation.gif', agent_name=""):

    def update(frame, fig):
        print("frame", frame)
        # Train the agent
        agent.train(num_episodes=1000, canshow=False)
        dealer_space, player_space, action_space = agent.q_values.shape

        # Clear the previous content
        fig.clear()

        # Create subplots on the existing figure
        ax1 = fig.add_subplot(131, projection='3d')
        x1 = range(dealer_space)
        y1 = range(player_space)
        x1, y1 = np.meshgrid(x1, y1)
        z1 = agent.q_values[:, :, Action.STICK.value]
        ax1.set_title(agent_name + 'Value of Action.STICK ' + str(frame))
        ax1.set_xlabel('Dealer Initial Card Value')
        ax1.set_ylabel('Player Card Value')
        ax1.plot_surface(x1, y1, z1.T, cmap='viridis')

        ax2 = fig.add_subplot(132, projection='3d')
        x2 = range(dealer_space)
        y2 = range(player_space)
        x2, y2 = np.meshgrid(x2, y2)
        z2 = agent.q_values[:, :, Action.HIT.value]
        ax2.set_title(agent_name + 'Value of Action.HIT ' + str(frame))
        ax2.set_xlabel('Dealer Initial Card Value')
        ax2.set_ylabel('Player Card Value')
        ax2.plot_surface(x2, y2, z2.T, cmap='viridis')

        ax3 = fig.add_subplot(133, projection='3d')
        x3 = range(dealer_space)
        y3 = range(player_space)
        x3, y3 = np.meshgrid(x3, y3)
        z3 = np.mean(agent.q_values, axis=2)
        ax3.set_title(agent_name + 'Average Value of Actions ' + str(frame))
        ax3.set_xlabel('Dealer Initial Card Value')
        ax3.set_ylabel('Player Card Value')
        ax3.plot_surface(x3, y3, z3.T, cmap='viridis')

    # Create the initial figure
    fig = plt.figure(figsize=(15, 5))

    # Create animation
    animation = FuncAnimation(
        fig, update, fargs=(fig,), frames=100, repeat=False)

    # Save as GIF
    animation.save(animation_name, writer='pillow', fps=5)
    print('Gif will be saved as %s' % animation_name)


def plot_q_values(q_values, canshow=True, agent_name=""):

    dealer_space, player_space, action_space = q_values.shape

    fig = plt.figure(figsize=(15, 5))

    # 绘制 Action.STICK 的价值
    ax1 = fig.add_subplot(131, projection='3d')
    x1 = range(dealer_space)
    y1 = range(player_space)
    x1, y1 = np.meshgrid(x1, y1)
    z1 = q_values[:, :, Action.STICK.value]
    ax1.set_title(agent_name + ' Value of Action.STICK')
    ax1.set_xlabel('Dealer Initial Card Value')
    ax1.set_ylabel('Player Card Value')
    ax1.plot_surface(x1, y1, z1.T, cmap='viridis')

    # 绘制 Action.HIT 的价值
    ax2 = fig.add_subplot(132, projection='3d')
    x2 = range(dealer_space)

    y2 = range(player_space)
    x2, y2 = np.meshgrid(x2, y2)
    z2 = q_values[:, :, Action.HIT.value]
    ax2.set_title(agent_name + ' Value of Action.HIT')
    ax2.set_xlabel('Dealer Initial Card Value')
    ax2.set_ylabel('Player Card Value')
    ax2.plot_surface(x2, y2, z2.T, cmap='viridis')

    # 绘制两个动作的平均价值
    ax3 = fig.add_subplot(133, projection='3d')
    x3 = range(dealer_space)
    y3 = range(player_space)
    x3, y3 = np.meshgrid(x3, y3)
    z3 = np.mean(q_values, axis=2)
    ax3.set_title(agent_name + ' Average Value of Actions')
    ax3.set_xlabel('Dealer Initial Card Value')
    ax3.set_ylabel('Player Card Value')
    ax3.plot_surface(x3, y3, z3.T, cmap='viridis')

    if canshow:
        plt.show()


# def plot_q_values(agent, title='Q Values Surface Plot', generate_gif=False, train_steps=None):
#     """
#     Plots Q values as a surface plot.

#     Args:
#         agent: A MonteCarloAgent.
#         title (string): Plot title.
#         generate_gif (boolean): If want to save plots as a gif.
#         train_steps: If is not None and generate_gif = True, then will use this
#                      value as the number of steps to train the model at each frame.
#     """
#     # you can change this values to change the size of the graph
#     fig = plt.figure(title, figsize=(10, 5))

#     # explanation about this line: https://goo.gl/LH5E7i
#     ax = fig.add_subplot(111, projection='3d')

#     q_values = agent.q_values

#     if generate_gif:
#         print('Gif will be saved as %s' % title)

#     def plot_frame(ax, agent, action_index=0):
#         # min value allowed accordingly with the documentation is 1
#         # we're getting the max value from V dimensions
#         min_x = 1
#         max_x = agent.q_values.shape[0]
#         min_y = 1
#         max_y = agent.q_values.shape[1]

#         # creates a sequence from min to max
#         x_range = np.arange(min_x, max_x)
#         y_range = np.arange(min_y, max_y)

#         # creates a grid representation of x_range and y_range
#         X, Y = np.meshgrid(x_range, y_range)

#         # get value function for X and Y values
#         def get_stat_val(x, y):
#             return agent.q_values[x, y, action_index]

#         Z = get_stat_val(X, Y)

#         # creates a surface to be plotted
#         ax.set_xlabel('Dealer Showing')
#         ax.set_ylabel('Player Sum')
#         ax.set_zlabel('Value')
#         return ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

#     # 在调用 plot_frame 时指定 action_index 参数，例如：
#     # plot_frame(ax, mc_agent, action_index=0)  # 绘制第一个动作的 Q 值
#     # plot_frame(ax, mc_agent, action_index=1)  # 绘制第二个动作的 Q 值


def plot_value_function(agent, title='Value Function', generate_gif=False, train_steps=None):
    """
    Plots a value function as a surface plot, like in: https://goo.gl/aF2doj

    Args:
        agent: An agent.
        title (string): Plot title.
        generate_gif (boolean): If want to save plots as a gif.
        train_steps: If is not None and generate_gif = True, then will use this
                     value as the number of steps to train the model at each frame.
    """
    # you can change this values to change the size of the graph
    fig = plt.figure(title, figsize=(10, 5))

    # explanation about this line: https://goo.gl/LH5E7i
    ax = fig.add_subplot(111, projection='3d')

    V = agent.get_value_function()

    if generate_gif:
        print('gif will be saved as %s' % title)

    def plot_frame(ax):
        # min value allowed accordingly with the documentation is 1
        # we're getting the max value from V dimensions
        min_x = 1
        max_x = V.shape[0]
        min_y = 1
        max_y = V.shape[1]

        print(V.shape)
        # creates a sequence from min to max
        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)

        # creates a grid representation of x_range and y_range
        X, Y = np.meshgrid(x_range, y_range)

        # get value function for X and Y values

        def get_stat_val(x, y):
            return V[x, y, 0]
        Z = get_stat_val(X, Y)

        # creates a surface to be ploted
        # check documentation for details: https://goo.gl/etEhPP
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_zlabel('Value')
        return ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

    def animate(frame):
        # clear the plot and create a new surface
        ax.clear()
        surf = plot_frame(ax)
        if generate_gif:
            i = agent.iterations
            # cool math to increase number of steps as we go
            if train_steps is None:
                step_size = int(min(max(1, agent.iterations), 2 ** 16))
            else:
                step_size = train_steps

            agent.train(step_size)
            plt.title('%s MC score: %s frame: %s' %
                      (title, float(agent.wins)/agent.iterations*100, frame))
        else:
            plt.title(title)

        fig.canvas.draw()
        return surf

    ani = animation.FuncAnimation(fig, animate, 32, repeat=False)

    # requires gif writer
    if generate_gif:
        ani.save(title + '.gif', writer='imagemagick', fps=3)
    else:
        plt.show()


def plot_error_vs_episode(sqrt_error, lambdas, train_steps=1000000, eval_steps=1000,
                          title='SQRT error VS episode number', save_as_file=False):
    """
    Given the sqrt error between sarsa(lambda) for multiple lambdas and
    an already trained MC control model this function plots a
    graph: sqrt error VS episode number.

    Args:
        sqrt_error (tensor): Multi dimension tensor.
        lambdas (tensor): 1D tensor.
        train_steps (int): The total steps used to train the models.
        eval_steps (int): Train_steps/eval_steps is the number of time the
                          errors were calculated while training.
        save_as_file (boolean).
    """
    # avoid zero division
    assert eval_steps != 0
    x_range = np.arange(0, train_steps, eval_steps)

    # assert that the inputs are correct
    assert len(sqrt_error) == len(lambdas)
    for e in sqrt_error:
        assert len(list(x_range)) == len(e)

    # create plot
    fig = plt.figure(title, figsize=(12, 6))
    plt.title(title)
    ax = fig.add_subplot(111)

    for i in range(len(sqrt_error)-1, -1, -1):
        ax.plot(x_range, sqrt_error[i], label='lambda %.2f' % lambdas[i])

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save_as_file:
        plt.savefig(title)
    plt.show()


def plot_error_vs_lambda(sqrt_error, lambdas, title='SQRT error vs lambda', save_as_file=False):
    """
        Given the sqrt error between sarsa(lambda) for multiple lambdas and
        an already trainedMC Control ths function plots a graph:
        sqrt error VS lambda.

        Args:
            sqrt_error (tensor): multiD tensor.
            lambdas (tensor): 1D tensor.
            title (string): Plot title.
            save_as_file (boolean).

        The srt_error 1D length must be equal to the lambdas length.
    """

    # assert input is correct
    assert len(sqrt_error) == len(lambdas)

    # create plot
    fig = plt.figure(title, figsize=(12, 6))
    plt.title(title)
    ax = fig.add_subplot(111)

    # Y are the last values found at sqrt_error
    y = [s[-1] for s in sqrt_error]
    ax.plot(lambdas, y)

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save_as_file:
        plt.savefig(title)
    plt.show()
