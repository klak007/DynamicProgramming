import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

## Define the problem parameters


# N = 3
# u_values = [0, 0.25, 0.5, 0.75, 1]
# x0 = 0
# xN = 1
# a = 1
# b = 1
# c = 1
#
# N = 3
# u_values = [0, 1, 2, 4, 5, 6, 7]
# x0 = 0
# xN = 8
# a = 1
# b = 1
# c = 1
#
# N = 4
# u_values = [0, 1, 2, 4, 6]
# x0 = 0
# xN = 8
# a = 1
# b = 1
# c = 1
#
# N = 4
# u_values = [0, 1, 2, 4, 6]
# x0 = 0
# xN = 8
# a = 9
# b = 5
# c = 7
#
# N = 4
# u_values = [0, 1, 2, 4, 6]
# x0 = 0
# xN = 8
# a = 3
# b = 4
# c = 7

# N = 4
# u_values = [0, 1, 2, 4, 6]
# x0 = 0
# xN = 8
# a = 2
# b = 8
# c = 1

N = 4
u_values = [1, 2, 3, 4]
x0 = 0
xN = 8
a = 2
b = 1
c = 4


def min_u():
    return min(filter(lambda x: x > 0, u_values))


def is_integer(n):
    return n % 1 == 0


# Define the state values
# if xN > 1 and is_integer(xN):
#     state_values = list(range(int(xN) + 1))  # Possible states from 0 to xN,
# else:
state_values = np.arange(0, xN + 0.1, min_u())  # Possible states from 0 to xN, with step min(u_values)

# Initialize dictionaries to store the cost-to-go (J) and the control actions
J = {}
u_optimal = {}


# Cost function definition
def cost_function(x, u, stage):
    return b * (x ** 2) + c * (u ** 2)


# Initialization of the final cost
J[(xN, N)] = a * (xN ** 2)


def print_initialization(final_cost, stage):
    print(
        f"J^{(stage)} {final_cost} -> {final_cost} = C( {final_cost} ) = {a} * {final_cost}^2 = {a * (final_cost ** 2)}")


final_cost = xN
J[(final_cost, N)] = a * (final_cost ** 2)

# Print the problem parameters
print(f"Problem Parameters:\nN = {N}\nu_values = {u_values}\nx0 = {x0}\nxN = {xN}\n")
print(f"State values: {state_values}\n")
print(f"Cost Function: \n C(x, u) = {b} * x^2 + {c} * u^2\n")

print_initialization(final_cost, N)

for k in range(N - 1, -1, -1):
    print(f"\nStage N={k}:")
    for x in reversed(state_values):
        if k == 0:
            min_cost = float('inf')
            best_u_list = []
            for u in u_values:
                x_next = x + u
                if (x_next, k + 1) in J:
                    cost = cost_function(x, u, k) + J[(x_next, k + 1)]
                    if cost < min_cost:
                        min_cost = cost
                        best_u_list = [u]
                    elif cost == min_cost:
                        best_u_list.append(u)
                    if x == 0:
                        print(
                            f"J^({k}) {xN:.2f} -> {x:.2f} = J^({k + 1}) {xN} -> {x_next} + T({x_next:.2f} -> {x:.2f}, {u}) = {J[(x_next, k + 1)]} + ({b} * {x:.2f}^2 + {c} * {u}^2) = {cost:.2f}")
            if min_cost == float('inf'):
                continue
            if x == 0:
                print(f"J^({k}) {xN:.2f} -> {x:.2f} = min({min_cost:.2f}) = {min_cost:.2f}")
            J[(x, k)] = min_cost
            u_optimal[(x, k)] = best_u_list

        else:
            min_cost = float('inf')
            best_u_list = []
            for u in u_values:
                x_next = x + u
                if (x_next, k + 1) in J:
                    cost = cost_function(x, u, k) + J[(x_next, k + 1)]
                    if cost < min_cost:
                        min_cost = cost
                        best_u_list = [u]
                    elif cost == min_cost:
                        best_u_list.append(u)
                    print(
                        f"J^({k}) {xN:.2f} -> {x:.2f} = J^({k + 1}) {xN} -> {x_next} + T({x_next:.2f} -> {x:.2f}, {u}) = {J[(x_next, k + 1)]} + ({b} * {x:.2f}^2 + {c} * {u}^2) = {cost:.2f}")
            if min_cost == float('inf'):
                continue
            if k != 0:
                print(f"J^({k}) {xN:.2f} -> {x:.2f} = min({min_cost:.2f}) = {min_cost:.2f}")
            J[(x, k)] = min_cost
            u_optimal[(x, k)] = best_u_list


def generate_sequences(x, k):
    if k == N:
        return [[]]
    sequences = []
    for u in u_optimal[(x, k)]:
        x_next = x + u
        for seq in generate_sequences(x_next, k + 1):
            sequences.append([u] + seq)
    return sequences


optimal_control_sequences = generate_sequences(x0, 0)

print("\nOptimal control sequences:")
for seq in optimal_control_sequences:
    print(seq)


def plot_transitions():
    # Create a color map for the control values
    colors = cm.rainbow(np.linspace(0, 1, len(u_values)))

    # Function to check if a state is reachable from x0 in one step
    def is_reachable_from_x0(state):
        return any(state == x0 + u for u in u_values)

    # Function to check if a state can reach xN in one step
    def can_reach_xN(state):
        return any(state + u == xN for u in u_values)

    # Plotting
    plt.figure(figsize=(10, 8))
    for n in range(N):
        for x in state_values:
            for u_index, u in enumerate(u_values):
                next_x = x + u
                if next_x in state_values:
                    # Only plot transitions that start from x0 when in the first stage
                    if n == 0 and (x != x0 or not is_reachable_from_x0(next_x)):
                        continue
                    # Skip states that are not reachable from x0 in the first stage
                    if n == 1 and not is_reachable_from_x0(x):
                        continue
                    # Only plot transitions that end in xN when in the final stage
                    if n == N - 1 and (next_x != xN or not can_reach_xN(x)):
                        continue
                    # Skip transitions that cannot reach xN in the remaining stages
                    if n == N - 2 and (not can_reach_xN(next_x)):
                        continue
                    # Skip states that can't reach any state that can reach xN
                    if n == N - 3 and all(abs(next_x + u - xN) > max(u_values) for u in u_values):
                        continue
                    plt.plot([n, n + 1], [x, next_x], color=colors[u_index], lw=1)
                    plt.plot(n, x, 'ko')
                    plt.plot(n + 1, next_x, 'ko')

    plt.xlabel('Stage (n)')
    plt.ylabel('State (x)')
    plt.yticks(state_values)
    plt.title(f'State Transitions for N = {N} \n u = {u_values}\n x0 = {x0}, xN = {xN} ')
    plt.grid(True)
    # save in plots directory
    plt.savefig(f'plots/N_{N}_u_{u_values}_x0_{x0}_xN_{xN}.png')
    plt.show()


# Call the function at the end of your script
plot_transitions()
