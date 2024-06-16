# plot_transitions.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters
a, b, c = 3, 4, 7
N = 4
control_values = [0, 1, 2, 4, 6]
xN = 8
x0 = 0
# State values
x_values = list(range(9))  # Possible states from 0 to 8

# Create a color map for the control values
colors = cm.rainbow(np.linspace(0, 1, len(control_values)))

# Function to check if a state is reachable from x0 in one step
def is_reachable_from_x0(state):
    return any(state == x0 + u for u in control_values)

# Function to check if a state can reach xN in one step
def can_reach_xN(state):
    return any(state + u == xN for u in control_values)

# Plotting
plt.figure(figsize=(10, 8))
for n in range(N):
    for x in x_values:
        for u_index, u in enumerate(control_values):
            next_x = x + u
            if next_x in x_values:
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
                if n == N - 3 and all(abs(next_x + u - xN) > max(control_values) for u in control_values):
                    continue
                plt.plot([n, n + 1], [x, next_x], color=colors[u_index], lw=1)
                plt.plot(n, x, 'ko')
                plt.plot(n + 1, next_x, 'ko')

plt.xlabel('Stage (n)')
plt.ylabel('State (x)')
plt.title(f'State Transitions for N = {N} \n u = {control_values}\n x0 = {x0}, xN = {xN} ')
plt.grid(True)
plt.savefig(f'N_{N}_u_{control_values}_x0_{x0}_xN_{xN}.png')
plt.show()
