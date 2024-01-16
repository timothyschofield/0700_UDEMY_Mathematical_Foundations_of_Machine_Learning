import matplotlib.pyplot as plt
import numpy as np

start_mins = 0
finish_mins = 40
num_points = 1000
time_steps = np.linspace(start_mins, finish_mins, num_points) # start, finish, n points

distance_robber = 2.5 * time_steps
distance_sheriff = 3 * (time_steps - 5)

"""
matplotlib.pyplot.subplots
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
Example down below

Create a figure and a set of subplots.
This utility wrapper makes it convenient to create common layouts of subplots, 
including the enclosing figure object, in a single call.

matplotlib.pyplot.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, 
    squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw)

"""
# fig is for everything to do with the "figure" - called via plt
# sub1 is the first sub-plot: what is plotted "on" the figure
fig, sub1 = plt.subplots()

# Title and label axes
plt.title('A Bank Robber Caught')
plt.xlabel('time (in minutes)')
plt.ylabel('distance (in km)')

# The range to plot over
sub1.set_xlim([0, 40])
sub1.set_ylim([0, 100])

# Do the plot
sub1.plot(time_steps, distance_robber, c='green')
sub1.plot(time_steps, distance_sheriff, c='brown')

# Cheat because he know the solution
plt.axvline(x=30, color='purple', linestyle='--') 
plt.axhline(y=75, color='purple', linestyle='--')

plt.show()
###################################################################

fig, ax = plt.subplots()

fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]

# The lables are the "internal" name of the bars
# The number of lables must match number of bars in bar_colors
# lables with a preceeding underscore will not show up in the legend
bar_labels = ['red', 'blue', '_red', 'orange']

bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')

# Without this - no legend, of course
# WITH this - title gets drawn, then
# display in a column the bar_labels that do not start with an "_"
# and to the left of this lable a little bar in the color of the bar_label's index in bar_colors
# With a nice boarder around the whole thing
ax.legend(title='Fruit color')

plt.show()
###################################################################