#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:02:07 2021

@author: brandon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys
#np.set_printoptions(threshold=sys.maxsize)

BASE_DIR='output/'
NUM_TRIALS=50
NANOS_IN_MS=1000000
problem_title=''
FIG_WIDTH = 6
FIG_HEIGHT = 3.5

PLOT_OPTIONS = {
   'marker': 'o',
   'markersize': 4,
}

def save_plot(title):
    is_by_state = False
    if 'tate' in plt.gcf().get_axes()[0].get_xlabel():
        is_by_state = True
    _, legend_labels = plt.gca().get_legend_handles_labels()
    for label in legend_labels:
        if 'tate' in label:
            is_by_state = True
    if is_by_state:
        plt.suptitle(problem_title.split()[0], fontsize='medium', va='top')
    else:
        plt.suptitle(problem_title, fontsize='medium', va='top')
    plt.gcf().set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    plt.savefig(data_dir + title + '.png', bbox_inches='tight')
    plt.close()

def plot_bellman(dfv, dfp, x_field):
    plt.title("Bellman Equation Calls required for Convergence")
    ax = plt.gca()
    ax.set_ylabel('Bellman Equation Invocations')
    ax.plot(dfv[x_field], dfv['Bellman Invocations'], label='Value Iteration', **PLOT_OPTIONS)
    ax.plot(dfp[x_field], dfp['Bellman Invocations'], label='Policy Iteration', **PLOT_OPTIONS)
    ax.legend()
    plt.xlabel(x_field)
    save_plot(x_field + ' - Bellman Invocations')

def plot_iterations(dfv, dfp, x_field):
    plt.title("Iterations required for Convergence")
    plt.xlabel(x_field)
    ax = plt.gca()
    color = 'tab:red'
    ax.set_ylabel('Value Iterations', color=color)
    line1 = ax.plot(dfv[x_field], dfv['Iterations'], label='Value Iteration',
                    color=color, **PLOT_OPTIONS)
    ax.tick_params(axis='y', labelcolor=color)
    twin = ax.twinx()
    color = 'tab:blue'
    twin.set_ylabel('Policy Iterations', color=color)
    line2 = twin.plot(dfp[x_field], dfp['Iterations'], label='Policy Iteration',
                      color=color, **PLOT_OPTIONS)
    twin.tick_params(axis='y', labelcolor=color)
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    #ax.set_yscale('log')
    plt.legend(lines, labels)
    save_plot(x_field + ' - Iterations')

def plot_value_iterations(dfv, dfp, x_field):
    plt.title("Value Iterations required for Convergence")
    ax = plt.gca()
    ax.set_ylabel('Value Iterations')
    ax.plot(dfv[x_field], dfv['Iterations'], label='Value Iteration', **PLOT_OPTIONS)
    ax.plot(dfp[x_field], dfp['Sub Iterations'], label='Policy Iteration', **PLOT_OPTIONS)
    ax.legend()
    plt.xlabel(x_field)
    #plt.xscale('log')
    save_plot(x_field + ' - Value Iterations')

def plot_cpu_time(dfv, dfp, x_field):
    plt.title("CPU Time required for Convergence")
    ax = plt.gca()
    ax.set_ylabel('CPU Time (ms)')
    ax.plot(dfv[x_field], dfv['CPU Time'] / NANOS_IN_MS, label='Value Iteration', **PLOT_OPTIONS)
    ax.plot(dfp[x_field], dfp['CPU Time'] / NANOS_IN_MS, label='Policy Iteration', **PLOT_OPTIONS)
    ax.legend()
    plt.xlabel(x_field)
    save_plot(x_field + ' - CPU Time')

def plot_final_policy_rewards(dfv, dfp, x_field):
    plt.title("Convergence Policy Avg Reward")
    ax = plt.gca()
    ax.set_ylabel('Avg Reward')
    if 'ridworld' in data_dir:
        ax.set_yscale('symlog')
    ax.plot(dfv[x_field], dfv['Average Reward'], label='Value Iteration', **PLOT_OPTIONS)
    ax.plot(dfp[x_field], dfp['Average Reward'], label='Policy Iteration', **PLOT_OPTIONS)
    ax.legend()
    plt.xlabel(x_field)
    save_plot(x_field + ' - Average Reward')

def plot_num_steps(dfv, dfp):
    plt.title("Number of steps in final policy")
    ax = plt.gca()
    ax.set_ylabel('Number of Steps')
    ax.plot(np.sqrt(dfv['Num States'])*2, dfv['Num Steps'],
            marker='>', markersize=10, label='Value Iteration', alpha=0.75)
    ax.plot(np.sqrt(dfp['Num States'])*2, dfp['Num Steps'],
            marker='<', markersize=10, label='Policy Iteration', alpha=0.75)
    ax.legend()
    plt.grid()
    plt.xlabel("Manhattan distance to goal")
    save_plot('Steps')

def find_convergence(avg_rewards_avg):
    avg_rewards_clean = avg_rewards_avg.copy()
    lower_bound = avg_rewards_clean[~np.isnan(avg_rewards_avg)].min()
    avg_rewards_clean[np.isnan(avg_rewards_avg)] = lower_bound
    lower_limit = min(0, min(avg_rewards_clean))
    avg_rewards_clean -= lower_limit
    upper_limit = max(avg_rewards_clean)
    avg_rewards_diff = upper_limit - avg_rewards_clean
    convergence_index = np.argmax(avg_rewards_diff < 0.02 * upper_limit)
    return convergence_index

def read_q_learning(dirName, filepart, title, smoothing):
    print(dirName, title)
    xscale = 'linear'
    yscale = 'linear'
    if dirName.endswith('Learning Rate Decay'):
        xscale = 'log'
        # yscale = 'symlog'
        pass
    
    plt.figure(1)
    plt.title('Q-Learning Average Episode Reward by ' + title)
    plt.xlabel(title)
    plt.ylabel('Average Reward')

    plt.figure(2)
    plt.title('Q-Learning Average Episode Steps by ' + title)
    plt.xlabel(title)
    plt.ylabel('Average Steps')
    #plt.yscale('log')
    
    plt.figure(3)
    plt.title('Q-Learning CPU Time for Convergence ' + title)
    plt.xlabel(title)
    plt.ylabel('CPU Time (ms)') # val / 1000000

    #plt.figure(4)
    #plt.title('Q-Learning CPU Time for Convergence ' + title)
    #plt.xlabel(title)
    #plt.ylabel('Num Steps')
    
    x_fields = []
    '''
    problem_sizes = {}
    discount_factors = {}
    learning_rates = {}
    qinits = {}
    '''
    for d in glob.glob(dirName + '/*.csv'):
        basename = os.path.basename(d)
        filename = os.path.splitext(basename)[0]
        parts = filename.split(',')
        x_field = float(parts[filepart])
        '''
        problem_size = int(parts[0])
        discount_factor = int(parts[1])
        learning_rate = int(parts[2])
        qinit = int(parts[3])
        '''
        x_fields.append((x_field, d))
    x_fields.sort()
    
    num_params = len(x_fields)
    # For each field value, get the avg reward and total steps per trial. Plot afterward.
    avg_rewards_by_trial = np.zeros((num_params, NUM_TRIALS))
    convergence_cpu_time_by_trial = np.zeros((num_params, NUM_TRIALS))
    convergence_steps_by_trial = np.zeros((num_params, NUM_TRIALS))
    avg_steps_by_trial = np.zeros((num_params, NUM_TRIALS))
    field_index = 0
    for x_field, d in x_fields:
        print('Attempting to read ' + d)
        dfq = pd.read_csv(d, dtype={
                'trial': 'int32',
                'episode': 'int32',
                'cumulativeReward': 'float',
                'averageReward': 'float',
                'numSteps': 'int32',
                'cumulativeSteps': 'int32',
                'cpuTime': 'int64',
                })
        dfq['episodeReward'] = dfq['numSteps'] * dfq['averageReward']
        if dirName.find('treble') >= 0:
            dfq.loc[dfq['episodeReward'] < 0, 'episodeReward'] = 0
        dfq['rollingReward'] = dfq['episodeReward'].rolling(smoothing).mean()
        grouped = dfq.groupby('trial')
        for trial, indices in grouped.groups.items():
            df_trial = dfq.iloc[indices, :].copy()
            avg_rewards_by_trial[field_index][trial] = df_trial['rollingReward'].mean()
            convergence_index = find_convergence(df_trial['rollingReward'])
            convergence_cpu_time_by_trial[field_index][trial] = df_trial['cpuTime'].values[convergence_index] / NANOS_IN_MS
            convergence_steps_by_trial[field_index][trial] = df_trial['cumulativeSteps'].values[convergence_index]
            avg_steps_by_trial[field_index][trial] = df_trial['numSteps'].mean()
        field_index += 1

    x = [f[0] for f in x_fields]
    xlabel = title
    if dirName.endswith('Learning Rate Decay'):
        x = [1-f for f in x]
        xlabel = '1 - ' + title
    avg_rewards_avg = np.mean(avg_rewards_by_trial, axis=1)
    avg_rewards_std = np.std(avg_rewards_by_trial, axis=1)
    plt.figure(1)
    color = 'tab:red';
    plt.xlabel(xlabel)
    plt.ylabel('Average Reward', color=color)
    if dirName.endswith('Learning Rate Decay'):
        #plt.ylim(bottom=min(avg_rewards_avg) * 10, top=max(avg_rewards_avg) * 10)
        plt.xlim(left=max(x), right=min(x))
    plt.plot(x, avg_rewards_avg, color=color, **PLOT_OPTIONS)
    lower_bound = avg_rewards_avg - avg_rewards_std
    upper_bound = avg_rewards_avg + avg_rewards_std
    plt.fill_between(x, lower_bound, upper_bound, alpha=0.2, color=color)
    plt.gca().tick_params(axis='y', labelcolor=color)
    plt.xscale(xscale)
    plt.yscale(yscale)

    avg_steps_avg = np.mean(avg_steps_by_trial, axis=1)
    avg_steps_std = np.std(avg_steps_by_trial, axis=1)
    plt.figure(2)
    plt.xlabel(xlabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.plot(x, avg_steps_avg, **PLOT_OPTIONS)
    lower_bound = avg_steps_avg - avg_steps_std
    upper_bound = avg_steps_avg + avg_steps_std
    plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)

    convergence_cpu_time_avg = np.mean(convergence_cpu_time_by_trial, axis=1)
    #convergence_cpu_time_std = np.std(convergence_cpu_time_by_trial, axis=1)
    plt.figure(3)
    plt.xlabel(xlabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.plot(x, convergence_cpu_time_avg, **PLOT_OPTIONS)
    #lower_bound = convergence_cpu_time_avg - convergence_cpu_time_std
    #upper_bound = convergence_cpu_time_avg + convergence_cpu_time_std
    lower_bound = np.min(convergence_cpu_time_by_trial, axis=1)
    upper_bound = np.max(convergence_cpu_time_by_trial, axis=1)
    plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)

    convergence_steps_avg = np.mean(convergence_steps_by_trial, axis=1)
    #convergence_steps_std = np.std(convergence_steps_by_trial, axis=1)
    plt.figure(1)
    ax = plt.gca().twinx()
    color = 'tab:blue';
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Num Steps for Convergence (thousands)', color=color)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.tick_params(axis='y', labelcolor=color)
    ax.plot(x, convergence_steps_avg / 1000, color=color, **PLOT_OPTIONS)
    #lower_bound = convergence_steps_avg - convergence_steps_std
    #upper_bound = convergence_steps_avg + convergence_steps_std
    lower_bound = np.min(convergence_steps_by_trial, axis=1) / 1000
    upper_bound = np.max(convergence_steps_by_trial, axis=1) / 1000
    ax.fill_between(x, lower_bound, upper_bound, alpha=0.2, color=color)
    

    plt.figure(1)
    save_plot('qlearning/' + title + '/avg_rewards')
    plt.figure(2)
    save_plot('qlearning/' + title + '/avg_steps')
    plt.figure(3)
    save_plot('qlearning/' + title + '/convergence_cpu_time')
    #plt.figure(4)
    #save_plot('qlearning/' + title + '/convergence_steps')

# Nicer plots for problem size based convergence
def q_learning_convergence(dirName, filepart, title, smoothing):
    plt.figure(1)
    plt.title('Q-Learning Average Reward by Number of Episodes')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Reward')

    plt.figure(2)
    plt.title('Q-Learning Number of Steps by Number of Episodes')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Number of Steps')
    plt.yscale('log')

    plt.figure(3)
    plt.title('Q-Learning Reward by Number of Steps')
    plt.xlabel('Number of Steps')
    plt.ylabel('Average Reward')
    #plt.xscale('log')
    
    x_fields = []
    for d in glob.glob(dirName + '/*.csv'):
        basename = os.path.basename(d)
        filename = os.path.splitext(basename)[0]
        parts = filename.split(',')
        x_field = float(parts[filepart])
        x_fields.append((x_field, d))
    x_fields.sort()
    
    p1_vlines = []
    p3_vlines = []
    ymin = 1000000
    
    for x_field, d in x_fields:
        dfq = pd.read_csv(d, dtype={
                'trial': 'int32',
                'episode': 'int32',
                'averageReward': 'float',
                'numSteps': 'int32',
                'cpuTime': 'int64',
                })
        dfq = dfq.fillna(0)
        dfq['episodeReward'] = dfq['numSteps'] * dfq['averageReward']
        if dirName.find('treble') >= 0:
            dfq.loc[dfq['episodeReward'] < 0, 'episodeReward'] = 0
        dfq['rollingReward'] = dfq['episodeReward'].rolling(smoothing).mean()
        grouped = dfq.groupby('trial')
        num_episodes = sys.maxsize
        for group in grouped.groups.values():
            num_episodes = min(num_episodes, group.size)
        #num_episodes = grouped.groups[0].size
        #print(grouped.groups[0].size)
        num_trials = len(grouped.groups)
        #print('num_episodes', num_episodes, 'num_trials', num_trials)
        avg_rewards = np.zeros((num_trials, num_episodes))
        num_steps = np.zeros((num_trials, num_episodes))
        cum_steps = np.zeros((num_trials, num_episodes))
        for trial, indices in grouped.groups.items():
            avg_rewards[trial] = dfq.iloc[indices, :]['rollingReward'].values[:num_episodes]
            num_steps[trial] = dfq.iloc[indices, :]['numSteps'].values[:num_episodes]
            cum_steps[trial] = dfq.iloc[indices, :]['cumulativeSteps'].values[:num_episodes]

        x = np.arange(0, num_episodes)

        avg_rewards_avg = np.mean(avg_rewards, axis=0)
        avg_rewards_min = np.min(avg_rewards, axis=0)
        avg_rewards_max = np.max(avg_rewards, axis=0)
        avg_rewards_std = np.std(avg_rewards, axis=0)
        plt.figure(1)

        convergence_index = find_convergence(avg_rewards_avg)
        '''
        avg_rewards_clean = avg_rewards_avg.copy()
        avg_rewards_clean[np.isnan(avg_rewards_avg)] = 0
        limit = max(avg_rewards_clean)
        avg_rewards_diff = limit - avg_rewards_clean
        convergence_index = np.argmin(avg_rewards_diff > 0.01)
        '''
        convergence = x[convergence_index]
        
        p = plt.plot(x, avg_rewards_avg,
                     label='#%s=%d, convergence=%d' % (title, x_field, convergence))
        lower_bound = avg_rewards_min
        upper_bound = avg_rewards_max
        #lower_bound = avg_rewards_avg - avg_rewards_std
        #upper_bound = avg_rewards_avg + avg_rewards_std
        plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)
        #print(convergence_index, convergence)
        ymax = avg_rewards_avg[convergence_index]
        ymin = min(ymin, min(avg_rewards_avg[~np.isnan(avg_rewards_avg)]))
        p1_vlines.append((convergence, p[-1].get_color(), ymax))
        #plt.vlines(x=convergence, linestyle='--', color=p[-1].get_color(), ymin=ymin, ymax=ymax)
        
        num_steps_avg = np.mean(num_steps, axis=0)
        num_steps_std = np.std(num_steps, axis=0)
        plt.figure(2)
        plt.plot(x, num_steps_avg, label='#%s=%d' % (title, x_field))
        lower_bound = num_steps_avg - num_steps_std
        upper_bound = num_steps_avg + num_steps_std
        plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)

        cum_steps_avg = np.mean(cum_steps, axis=0)
        #cum_steps_std = np.std(cum_steps, axis=0)
        plt.figure(3)
        convergence = cum_steps_avg[convergence_index]
        p = plt.plot(cum_steps_avg, avg_rewards_avg, label='#%s=%d, convergence=%d'
                 % (title, x_field, convergence))
        lower_bound = avg_rewards_avg - avg_rewards_std
        upper_bound = avg_rewards_avg + avg_rewards_std
        plt.fill_between(cum_steps_avg, lower_bound, upper_bound, alpha=0.2)
        p3_vlines.append((convergence, p[-1].get_color(), ymax))
        #plt.vlines(x=convergence, linestyle='--', color=p[-1].get_color(), ymin=ymin, ymax=ymax)


    plt.figure(1)
    for convergence, color, ymax in p1_vlines:
        plt.vlines(x=convergence, linestyle='--', color=color, ymin=ymin, ymax=ymax)
    plt.legend()
    save_plot('qlearning/' + title + '/average_rewards_convergence')

    plt.figure(2)
    plt.legend()
    save_plot('qlearning/' + title + '/num_steps_convergence')
    
    plt.figure(3)
    for convergence, color, ymax in p3_vlines:
        plt.vlines(x=convergence, linestyle='--', color=color, ymin=ymin, ymax=ymax)
    plt.legend()
    save_plot('qlearning/' + title + '/average_reward_by_num_steps_convergence')


def plot_iter_convergence(filePrefix, filepart, title):
    plt.figure(1)
    plt.title('Max Delta Reward by Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Max Delta Reward')

    x_fields = []
    for d in glob.glob(filePrefix + '*.csv'):
        basename = os.path.basename(d)
        filename = os.path.splitext(basename)[0]
        parts = filename.split(',')
        x_field = float(parts[filepart])
        x_fields.append((x_field, d))
    x_fields.sort()
    if len(x_fields) > 5:
        every = len(x_fields) // 5
        x_fields = x_fields[::every]
    
    for x_field, d in x_fields:
        dfq = pd.read_csv(d, dtype={
                'Iteration': 'int32',
                'Max Delta Reward': 'float'
                })
        #print('num_episodes', num_episodes, 'num_trials', num_trials)
        plt.figure(1)
        #dfq.plot(ax=plt.gca(), x='Iteration', y='Max Delta Reward')
        plt.plot(dfq['Iteration'].values, dfq['Max Delta Reward'].values,
                 label='#%s=%s' % (title, x_field), **PLOT_OPTIONS)
        
    plt.figure(1)
    plt.legend()
    save_plot(title + ' delta_rewards_convergence')

def plot_mdp(title, subdir, smoothing):
    global data_dir
    global problem_title
    problem_title = title
    data_dir = BASE_DIR + subdir + '/'
    dfvs = pd.read_csv(data_dir + 'value_by_state.csv')
    dfps = pd.read_csv(data_dir + 'policy_by_state.csv')
    
    dfvd = pd.read_csv(data_dir + 'value_by_discount_factor.csv')
    dfpd = pd.read_csv(data_dir + 'policy_by_discount_factor.csv')
    
    for it in ['vi', 'pi']:
        for t, filepart in [('state', 1), ('discount_factor', 2)]:
            plot_iter_convergence(data_dir + it + '_by_' + t + '/' + it, filepart, it + ' by ' + t)

    for dfv, dfp, title in [(dfvs, dfps, "Num States"), (dfvd, dfpd, "Discount Factor")]:
        plot_bellman(dfv, dfp, title)
        plot_final_policy_rewards(dfv, dfp, title)
        plot_iterations(dfv, dfp, title)
        plot_value_iterations(dfv, dfp, title)
        plot_cpu_time(dfv, dfp, title)
        
    if os.path.exists(data_dir + 'qlearning/Num States'):
        q_learning_convergence(data_dir + 'qlearning/Num States', 0, 'Num States', smoothing)

    subdirs = ['Num States', 'Discount Factor', 'Learning Rate', 'Learning Rate Decay', 'QInit', 'Epsilon']
    for i in range(len(subdirs)):
        part = i
        subdir = subdirs[i]
        if os.path.exists(data_dir + 'qlearning/' + subdir):
            read_q_learning(data_dir + 'qlearning/' + subdir, part, subdir, smoothing)
    
    plot_num_steps(dfvs, dfps)

plot_mdp('Treblecross #States=84', 'treblecross_small', 1000)
plot_mdp('Treblecross #States=5104', 'treblecross_big', 1000)
plot_mdp('Gridworld #States=200', 'gridworld_small', 10)
plot_mdp('Gridworld #States=845', 'gridworld_big', 10)

os.system("./glue_plots.sh")