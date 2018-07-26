import numpy as np 
import itertools
import nengo
import pandas
import random
import nengo_spa as spa
import seaborn
import matplotlib.pyplot as plt

import warnings

# avoid seeing all the SPA max similarity warnings
warnings.simplefilter("ignore")


def build_goal_paths(object_keys, n_goals, n_steps):
    # for now, cannot have shared preconditions
    assert n_goals * n_steps <= len(object_keys)
    paths = [[] for x in range(n_goals)]
    
    for step in range(n_steps):
        effects = np.random.choice(object_keys, n_goals, replace=False)
        for effect, path in zip(effects, paths):
            path.insert(0, effect)

        # remove used object_keys to ensure no circular deps
        object_keys = list(set(object_keys) - set(effects))

    return paths


def build_precons_map(paths, base_vocab, goal_keys):
    '''Built key, value vocabs for matching actions to preconditions'''
    actions = []
    precons = []

    for path in paths:
        for idx in range(len(path) - 1):
            actions.append('ADD_' + path[idx + 1])
            precons.append(path[idx])

    action_vocab = base_vocab.create_subset(actions)
    precon_vocab = base_vocab.create_subset(precons)

    return action_vocab, precon_vocab


def build_objects_map(paths, base_vocab, goal_keys):
    '''Built key, value vocabs for matching goals to collections of objects'''
    object_sums = []

    for idx, goal_key in enumerate(goal_keys): 
        path = paths[idx][:]
        tag = goal_key + '_SUM'
        base_vocab.populate(tag + ' = ' + '+'.join(path))
        object_sums.append(tag)

    goal_vocab = base_vocab.create_subset(goal_keys)
    sums_vocab = base_vocab.create_subset(object_sums)

    return goal_vocab, sums_vocab


def build_actions_map(paths, base_vocab):
    '''Built key, value vocabs for matching obj, goal combos to actions'''

    action_key_vocab = spa.Vocabulary(dim)
    actions = []

    for idx, path in enumerate(paths): 

        goal_sum = 'GOAL_' + str(idx) + '_SUM' 

        for x in range(len(path) - 1):
            immediate_goal = path[x]
            action = 'ADD_' + immediate_goal

            sp = base_vocab['OBJECT'] * base_vocab[goal_sum] + \
                 base_vocab['GOAL'] * base_vocab[immediate_goal]

            actions.append(action)
            action_key_vocab.add(action + '_KEY', sp.v)

        # add (main goal to last action) mapping
        action = 'ADD_' + path[-1]
        sp = base_vocab['OBJECT'] * base_vocab[goal_sum] + \
             base_vocab['GOAL'] * base_vocab['GOAL_' + str(idx)]

        actions.append(action)
        action_key_vocab.add(action + '_KEY', sp.v)

    action_val_vocab = base_vocab.create_subset(actions)

    return action_key_vocab, action_val_vocab


def associate(sp, vocab_1, vocab_2):
    '''Use two vocabs to compute association using an input SP'''
    key = np.argmax(np.dot(vocab_1.vectors, sp.v))
    val = list(vocab_2.keys())[key]
    return val, vocab_2[val]

def check_plan(sp):
    pass


def run_trial(dim, n_objects, n_goals, n_steps):
    '''Collect accuracy measure for a trial using the method'''
    goal_keys = ['GOAL_' +str(x) for x in range(n_goals)]
    object_keys = ['OBJ_' + str(x) for x in range(n_objects)]
    action_keys = ['ADD_' + x for x in object_keys]
    tags = ['GOAL', 'OBJECT']

    paths = build_goal_paths(object_keys, n_goals, n_steps)

    # build all the vocabs for mapping between objects, actions, goals, precons
    base_vocab = spa.Vocabulary(dim)
    base_vocab.populate(';'.join(goal_keys+object_keys+action_keys+tags))

    action_vocab, precon_vocab = build_precons_map(paths, base_vocab, goal_keys)
    goal_vocab, sums_vocab = build_objects_map(paths, base_vocab, goal_keys)
    action_key_vocab, action_val_vocab = build_actions_map(paths, base_vocab)

    total = 0
    correct = 0
    results = []

    for idx, goal in enumerate(goal_keys):
        print('Current Goal: %s' % goal)
        print('Target Sequence: ', paths[idx])
        plan = []
        current_goal = goal

        for step in range(n_steps):
            goal_tag = 'GOAL_'+str(idx)+'_SUM'
            action_key = base_vocab['OBJECT'] * base_vocab[goal_tag] + \
                         base_vocab['GOAL'] * base_vocab[current_goal]

            action, _ = associate(
                action_key, action_key_vocab, action_val_vocab)
            
            print('Planned action: % s' % action)
            plan.append(action)
            
            precon, _ = associate(
                base_vocab[action], action_vocab, precon_vocab)
            
            current_goal = precon

        plan = [x.strip('ADD_') for x in plan]
        if list(reversed(plan)) == paths[idx]:
            correct += 1
        total += 1
        print('')

    return correct / total


def run_sweep(dim, goals_range, n_objects, n_steps):
    '''Run trials over range of numbers of goal states'''
    dicts = []
    for goal_count in goals_range:

        results = run_trial(dim, n_objects, goal_count, n_steps)
        entry = dict(n_goals=goal_count, accuracy=results, dim=dim)

        dicts.append(entry)

    return dicts

all_data = []
n_trials = 20
dims = [64, 128, 256, 512]
goalset = [10, 20, 30, 40, 50, 60, 70, 80]

for _ in range(n_trials):
    for dim in dims:
        values = run_sweep(dim, goalset, 500, 6)
        all_data.extend(values)


df = pandas.DataFrame(all_data)
print(df)
plt.figure(figsize=(6,6))
seaborn.pointplot('n_goals', 'accuracy', data=df, hue='dim')
plt.xlabel('Number of distinct goals')
plt.ylabel('Percentage of plans with no errors')
plt.title('HRR Planning with No Branching in Action Dependencies')
plt.show()


