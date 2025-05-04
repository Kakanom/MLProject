import matplotlib
import matplotlib.pyplot as plt

# Create a figure with a 2x1 grid of subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column, adjust figure size

# First Subplot: Two graphs overlaid
ax1.plot([1, 2, 3, 4], [10, 20, 25, 300], label='Graph 1', color='blue')    # First graph
ax1.plot([1, 2, 3, 4], [30, 25, 20, 10], label='Graph 2', color='orange')  # Second graph
ax1.set_title('Subplot 1: Two Graphs')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.legend()
ax1.grid(True)

# Second Subplot: Two graphs overlaid
ax2.plot([0, 1, 2, 3], [1, 4, 9, 16], label='Graph 1', color='green')     # First graph
ax2.plot([0, 1, 2, 3], [16, 9, 4, 1], label='Graph 2', color='purple')    # Second graph
ax2.set_title('Subplot 2: Two Graphs')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.legend()
ax2.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('grid_two_plots.png', dpi=300, bbox_inches='tight')

# Display the plot (for VSCode Python script)
plt.show()