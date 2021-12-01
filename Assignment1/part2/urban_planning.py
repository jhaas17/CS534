# Urban Planning Map- Assignment 1
# Hill climb restart and simulated annealing
import argparse

import math
import numpy as np
import random

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import time

np.set_printoptions(threshold=np.inf)

def get_terrain(filename):
    # Read in the file and get terrain array and list of maxes
    file = open(filename, "r")
    lines = file.readlines()

    rows = len(lines) - 3  # Subtract the first three lines!
    longest_line = max(open(filename), key=len)
    columns = len(longest_line.split(","))

    # Maxes order: Industrial, Commercial, Residential
    maxes = [int(lines[0]), int(lines[1]), int(lines[2])]

    terrain = np.zeros([rows, columns], dtype=object)

    i = 0
    for row in lines[3:]:
        j = 0
        for x in row:
            if x != "," and x != "\n":
                if x.isnumeric():
                    terrain[i, j] = int(x)
                else:
                    terrain[i, j] = x
                j += 1
        i += 1

    return terrain, maxes


def restart(terrain, maxes):
    height, width = terrain.shape[0], terrain.shape[1]
    start_state = terrain.copy()

    # Generate random number of each item to put on terrain, get random coordinates and check if we can build there
    for _I in range(random.randint(0, maxes[0])):
        while True:
            i = random.randint(0, height - 1)
            j = random.randint(0, width - 1)
            if isinstance(start_state[i][j], int):
                start_state[i][j] = "I"
                break
    for _C in range(random.randint(0, maxes[1])):
        while True:
            i = random.randint(0, height - 1)
            j = random.randint(0, width - 1)
            if isinstance(start_state[i][j], int):
                start_state[i][j] = "C"
                break
    for _R in range(random.randint(0, maxes[2])):
        while True:
            i = random.randint(0, height - 1)
            j = random.randint(0, width - 1)
            if isinstance(start_state[i][j], int):
                start_state[i][j] = "R"
                break
    return start_state


def score(terrain, state):
    score = 0
    # Get coordinates of all X S I R C
    toxic_sites = tuple(zip(*np.where(state == "X")))
    scenic_views = tuple(zip(*np.where(state == "S")))
    industrial_sites = tuple(zip(*np.where(state == "I")))
    commercial_sites = tuple(zip(*np.where(state == "C")))
    residential_sites = tuple(zip(*np.where(state == "R")))

    # Score rules
    for ind in industrial_sites:
        if terrain[ind[0]][ind[1]] == 'S':
            score -= 1
        else:
            score -= terrain[ind[0]][ind[1]]
        for toxic in toxic_sites:
            if distance(toxic, ind) <= 2:
                score -= 10
        for ind2 in industrial_sites:
            ii_dist = distance(ind, ind2)
            if ii_dist <= 2 and ii_dist != 0:
                score += 2
    for com in commercial_sites:
        if terrain[com[0]][com[1]] == 'S':
            score -= 1
        else:
            score -= terrain[com[0]][com[1]]
        for toxic in toxic_sites:
            if distance(toxic, com) <= 2:
                score -= 20
        for res in residential_sites:
            if distance(res, com) <= 3:
                score += 4
        for com2 in commercial_sites:
            cc_dist = distance(com, com2)
            if cc_dist <= 2 and cc_dist != 0:
                score -= 4
    for res in residential_sites:
        if terrain[res[0]][res[1]] == 'S':
            score -= 1
        else:
            score -= terrain[res[0]][res[1]]
        for toxic in toxic_sites:
            if distance(toxic, res) <= 2:
                score -= 20
        for scenic in scenic_views:
            if distance(scenic, res) <= 2:
                score += 10
        for ind in industrial_sites:
            if distance(ind, res) <= 3:
                score -= 5
        for com in commercial_sites:
            if distance(com, res) <= 3:
                score += 4
    return score


def distance(site1, site2):
    '''
    Manhattan distance

    :param site1:
    :param site2:
    :return:
    '''
    dist = abs(site1[0] - site2[0]) + abs(site1[1] - site2[1])
    return dist


def is_valid_tile(terrain, tilePosition):
    '''

    :param terrain:
    :param tilePosition: position to validate
    :return: true if tile is a valid position in terrain
    '''
    num_rows = terrain.shape[0]
    num_cols = terrain.shape[1]

    current_row = tilePosition[0]
    current_col = tilePosition[1]

    return current_row < num_rows and current_col < num_cols


def simulated_annealing_repeat(init_terrain, max_locs, time_limit):
    '''
    Repeats simulated annealing algorithm
    :param num_repeats: number of times to run algorithm
    :param init_terrain: geographical configuration given to us by professor
    :param max_locs: maximum number of industrial, commercial, and residential locations (respectively): (I, C, R)
    :return:
    '''
    # Maxes after running simulated annealing for given number of times
    # In the format: [(score1, state1), (score2, state2), ...]
    repeat_max = []

    random_terrain = restart(init_terrain, max_locs)

    start_time = time.time()
    while time.time() - start_time < time_limit:
        # Perform simulated annealing
        repeat_max.append(simulated_annealing(terrain, random_terrain, max_locs, start_time))

    print("Time taken for HC: {}".format(time.time() - start_time))

    # Returns the state with maximum score
    max_score = max(repeat_max, key=lambda x: x[0])[0]
    max_pool = filter(lambda x: x[0] == max_score, repeat_max)
    return min(max_pool, key=lambda x: x[2])



def simulated_annealing(init_terrain, config_turrain, maxes, start_time, init_temp=10, temp_threshold=1,
                        max_sideways=10):
    '''
    Simulated annealing, assuming we have all the locations to construct configured on the config_turrain
    to start off with (randomly distributed)

    :param init_terrain: geographical configuration given to us by professor (which we are NOT allowed to modify)
    :param config_turrain: random configuration to work with from restart(turrain, maxes)
    :param init_temp: temperature to start off with
    :param temp_threshold: when to stop running the algorithm
    :param num_sideways: number of sideway moves allowed (for equally good as current position cases)
    :return:
    '''

    temp = init_temp

    max_state = (score(init_terrain, config_turrain), config_turrain, time.time() - start_time)  # Initial MAX state

    # List of items that are not on the map
    off_map = []
    used_i = (max_state[1] == 'I').sum()
    used_r = (max_state[1] == 'R').sum()
    used_c = (max_state[1] == 'C').sum()

    off_map += ['I' for i in range(0, maxes[0] - used_i)]
    off_map += ['R' for i in range(0, maxes[1] - used_r)]
    off_map += ['C' for i in range(0, maxes[2] - used_c)]

    num_sideways = 0

    while temp > temp_threshold:

        # Make a copy of max terrain since we are updating or tyring to get a better state than this current max
        new_terrain = max_state[1].copy()  # Editable terrain

        # Randomly pick an occupied position on the config map to move
        # This can be I, C, or R
        occupied_spots = get_occupied_positions(max_state[1])
        froms = occupied_spots + off_map
        move_from = random.choice(froms)

        removed = None

        if type(move_from) is tuple:
            tos = get_available_positions(max_state[1])
            tos.append(move_from)
            move_to = random.choice(tos)

            # Randomly decide to either remove from board or keep in the same spot
            if move_from == move_to:
                rand = random.randint(0, 1)
                if rand == 1:
                    removed = new_terrain[move_from[0]][move_from[1]]
                    new_terrain[move_from[0]][move_from[1]] = init_terrain[move_from[0]][
                        move_from[1]]  # Replace value at from with original
                else:
                    new_terrain = max_state[1].copy()
            else:
                new_terrain[move_to[0]][move_to[1]] = new_terrain[move_from[0]][move_from[1]]  # Update value at to
                new_terrain[move_from[0]][move_from[1]] = init_terrain[move_from[0]][
                    move_from[1]]  # Replace value at from with original

        else:
            tos = get_available_positions(max_state[1])
            move_to = random.choice(tos)
            new_terrain[move_to[0]][move_to[1]] = move_from  # Update value at to

        # Calculate score of new state
        new_score = score(init_terrain, new_terrain)

        new_state = (new_score, new_terrain, time.time() - start_time)
        isbetter, sideways = is_better(max_state, new_state, temp, num_sideways)
        if isbetter and sideways <= max_sideways:  # If better than current state, go there with current temp
            # Update max
            max_state = new_state

            if not type(move_from) is tuple:
                off_map.remove(move_from)
            elif not removed is None:
                off_map.append(removed)

        # Update temperature
        temp = get_updated_temperature(temp)

    return max_state


def get_available_positions(current_terrain):
    '''

    :param current_terrain: current map to examine
    :return: a list of positions [(row, column), ... ] that do not contain an X
    '''
    return list(zip(*np.where(
        (current_terrain != "X") & (current_terrain != "R") & (current_terrain != "I") & (current_terrain != "C"))))


def get_occupied_positions(current_terrain):
    '''

    :param current_terrain: current map to examine
    :return: a list of positions [(row, column), ... ] that are occupied on the given map
    '''

    return list(zip(*np.where((current_terrain == "I") | (current_terrain == 'C') | (current_terrain == 'R'))))


def get_updated_temperature(current):
    '''

    :param current: temp to decrease from
    :return: new temperature after the decrease
    '''

    return current * 0.999


def is_better(current_max, new_state, current_temp, num_sideways):
    '''

    :param num_sideways:
    :param current_max: (max_score, max_state)
    :param new_state: (new_score, new_state)
    :param current_temp: current temperature of run
    :return: true if new state is better than current max state
    '''

    if new_state[0] > current_max[0]:
        num_sideways = 0
        return True, num_sideways
    elif new_state[0] == current_max[0]:
        num_sideways += 1
        return True, num_sideways

    # If it’s worse, go there with some probability (aka how likely to accept states with lower scores?)
    p = math.exp((new_state[0] - current_max[0]) / current_temp)

    if .8 <= p:
        num_sideways = 0
        return True, num_sideways
    return False, num_sideways


def get_child(terrain, parent1, parent2, max_sites, population_prob=0.2):
    """

    :param terrain: initial terrain configuration of map
    :param parent1: first parent
    :param parent2: second parent
    :param max_sites:
    :param population_prob: probability of population for mutation
    :return child terrain: new terrain generated from site locations inherited from both p1 and p2 [row, column]
    """

    child = perform_cross_over(terrain, parent1[1], parent2[1])

    # Randomly generate a probability
    rand = random.random()  # Generates a float number [0.0, 1.0)

    if rand < population_prob:  # Accept mutation for this child
        perform_mutation(terrain, child, max_sites)

    return child


def perform_cross_over(terrain, parent_terrain_1, parent_terrain_2):
    # New editable child terrain to place inherited site locations
    child = terrain.copy()

    # Cross over
    i_parent1 = list(zip(*np.where(parent_terrain_1 == "I")))
    c_parent1 = list(zip(*np.where(parent_terrain_1 == "C")))
    r_parent1 = list(zip(*np.where(parent_terrain_1 == "R")))

    i_parent2 = list(zip(*np.where(parent_terrain_2 == "I")))
    c_parent2 = list(zip(*np.where(parent_terrain_2 == "C")))
    r_parent2 = list(zip(*np.where(parent_terrain_2 == "R")))

    i_positions = i_parent1 + i_parent2
    c_positions = c_parent1 + c_parent2
    r_positions = r_parent1 + r_parent2

    # Cross over: add sites from both parents to child
    i_num = random.choice([len(i_parent1), len(i_parent2)])
    c_num = random.choice([len(c_parent1), len(c_parent2)])
    r_num = random.choice([len(r_parent1), len(r_parent2)])

    inherit_random_sites(child, "I", i_num, i_positions)
    inherit_random_sites(child, "C", c_num, c_positions)
    inherit_random_sites(child, "R", r_num, r_positions)

    return child


def inherit_random_sites(child, site_type, num, positions):
    for _ in range(0, num):
        site_position = random.choice(positions)

        # Place site on position in child terrain
        child[site_position[0], site_position[1]] = site_type


# Randomly choose which type of mutation to perform
def perform_mutation(terrain, child, max_sites):
    mutation_done = False
    available_actions = [0, 1, 2]

    while (len(available_actions) > 0 and not mutation_done):
        action = random.choice(available_actions)

        if action == 0:
            mutation_done = mutate_any_position(terrain, child)
        elif action == 1:
            mutation_done = mutate_remove_site(terrain, child)
        else:
            mutation_done = mutate_add_site(child, max_sites)

        available_actions.remove(action)


def mutate_any_position(terrain, child):
    # Pick a location
    locations = get_occupied_positions(child)

    if len(locations) > 0:
        position_to_mutate = random.choice(locations)
        site_to_move = child[position_to_mutate[0], position_to_mutate[1]]

        # Restore mutated position to initial value
        child[position_to_mutate[0], position_to_mutate[1]] = terrain[position_to_mutate[0], position_to_mutate[1]]

        # Mutation: randomly generate a direction and update a random coordinate axis in child
        mutated_position = get_mutated_position(child, position_to_mutate)

        # Move location to mutated position on child terrain
        child[mutated_position[0], mutated_position[1]] = site_to_move
        return True

    return False


def mutate_remove_site(terrain, child):
    # Pick a location
    locations = get_occupied_positions(child)

    if len(locations) > 0:
        site_to_remove = random.choice(locations)

        # Restore mutated position to initial value
        child[site_to_remove[0], site_to_remove[1]] = terrain[site_to_remove[0], site_to_remove[1]]

        return True

    return False


def mutate_add_site(child, max_locs):
    i = list(zip(*np.where(child == "I")))
    c = list(zip(*np.where(child == "C")))
    r = list(zip(*np.where(child == "R")))

    sites = {0: "I", 1: "C", 2: "R"}
    on_sites = [len(i), len(c), len(r)]

    # Check to see if there is any missing pieces to add
    if on_sites == max_locs:
        return False

    while True:
        site_type = random.choice(list(sites.keys()))

        if on_sites[site_type] != max_locs[site_type]:
            # Get a random position to add on child terrain
            site_to_add = random.choice(get_available_positions(child))
            # Add the site to the position
            child[site_to_add[0], site_to_add[1]] = sites[site_type]

            return True


def get_mutated_position(child, position):
    while True:
        mutated_position = list(position)  # Tuple object does not support item assignment when swapping values

        position_index = random.choice([0, 1])
        position_direction = random.choice([-1, 0, 1])  # Let's change position by 1 (or no change)

        mutated_position[position_index] = mutated_position[position_index] + position_direction

        if is_valid_tile(child, tuple(mutated_position)):
            return position


def genetic_algorithm(terrain, max_locs, time_limit, pool_size=250, elite_factor=.2, culling_factor=.2):
    start_time = time.time()
    gene_pool = []
    for _ in range(0, pool_size):
        state = restart(terrain, max_locs)
        gene_pool.append((score(terrain, state), state, time.time() - start_time))

    while (time.time() - start_time < time_limit):
        gene_pool.sort(key=lambda x: x[0], reverse=True)

        # Elitism - best parents go straight to next generation
        new_pool = gene_pool[0:int(pool_size * elite_factor)]

        # Culling - worst parents get removed before we breed children
        gene_pool = gene_pool[0:int((1 - culling_factor) * pool_size)]
        
        # Fitness
        gene_pool_weights = []
        for member in gene_pool:
            dst = gene_pool[0][0] - member[0] + 1
            weight = (1 / dst)
            gene_pool_weights.append(weight)

        # Make children and add to next generation
        for _ in range(0, int((1 - elite_factor) * pool_size)):
            parents = random.choices(gene_pool, weights=gene_pool_weights, k=2)
            child = get_child(terrain, parents[0], parents[1], max_locs)
            new_pool.append((score(terrain, child), child, time.time() - start_time))

        gene_pool = new_pool

    print("Time taken for GA: {}".format(time.time() - start_time))

    max_score = max(gene_pool, key=lambda x: x[0])[0]
    max_pool = filter(lambda x: x[0] == max_score, gene_pool)
    return min(max_pool, key=lambda x: x[2])


def testing(terrain, maxes):
    time_limits = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    GA_scores = []
    SA_scores = []

    for i in time_limits:
        GA_scores.append(genetic_algorithm(terrain, maxes, i)[0])
        SA_scores.append(simulated_annealing_repeat(terrain, maxes, i)[0])
    plot_test(GA_scores, SA_scores, time_limits)


def plot_test(GA_scores, SA_scores, time_limits):
    plt.scatter(time_limits, GA_scores, c='r', label='Genetic Algorithm')
    plt.scatter(time_limits, SA_scores, c='b', label='Simulated Annealing')
    plt.xlabel("Time Limit (s)")
    plt.ylabel("Score")
    plt.title("Time Limit vs. Score")
    plt.legend(loc='lower right')
    plt.ylim((0, 60))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='F', type=str, nargs=1)
    parser.add_argument('algorithm', metavar='A', type=str, nargs=1)

    args = parser.parse_args()

    terrain, maxes = get_terrain(args.file[0])

    # Time limit in seconds
    time_limit = 10
    if args.algorithm[0] == 'GA':
        GA = genetic_algorithm(terrain, maxes, time_limit)
        print("Score: %s\nMap: \n%s\nTime best state was first reached: %s" % (GA[0], GA[1], GA[2]))

    elif args.algorithm[0] == 'HC':

        HC = simulated_annealing_repeat(terrain, maxes, time_limit)
        print("Score: %s\nMap: \n%s\nTime best state was first reached: %s" % (HC[0], HC[1], HC[2]))

    elif args.algorithm[0] == 'TEST':
        testing(terrain, maxes)
