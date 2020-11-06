import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import stats

train_data = pd.read_csv('train-data.csv', sep=',', header=None)
train_labels = pd.read_csv('train-labels.csv', sep=',', header=None)


# let's make the data talk.

number_of_classes = train_labels.iloc[:, 1].max() + 1  # There are 10 classes.
distribution_among_classes = [list(train_labels.iloc[:, 1]).count(i) for i in range(number_of_classes)]
number_of_training_data = train_labels.max()[0]  # There are 57500 samples.
dimension_of_data = len(train_data.columns) - 1  # 717 = 3 * 239 is the dimension of the data.


# what are those fucking data?

fig_random_sample, ax_random_sample = plt.subplots()
fig_random_sample.set_size_inches(3, 2)
random_number = random.randint(1, number_of_training_data)
random_sample = train_data.iloc[random_number - 1, 1:]
ax_random_sample.bar(range(dimension_of_data), random_sample)
fig_random_sample.show()  # what the fuck is this?


# If you run it multiple times, the position of the cluster of spikes seems to move... let's see how.

first_non_zero = []
last_non_zero = []
width_non_zero = []
plot_length = 10000
for sample in range(plot_length):
    first_non_zero.append(np.nonzero(np.array(train_data.iloc[sample, 1:]))[0][0])  # the first [0] is to make it work.
    last_non_zero.append(np.nonzero(np.array(train_data.iloc[sample, 1:]))[0][-1])
    width_non_zero.append(last_non_zero[sample]-first_non_zero[sample])

fig_non_zero, ax_non_zero = plt.subplots(2, 2)
fig_non_zero.set_size_inches(5, 3)
ax_non_zero[0][0].plot(range(1, plot_length + 1), first_non_zero, '.k')
ax_non_zero[0][1].plot(range(1, plot_length + 1), last_non_zero, '.r')
ax_non_zero[1][0].plot(range(1, plot_length + 1), width_non_zero, '.b')
fig_non_zero.show()

# Ok dude it seems that the data has kind of effective length and the position is kind of randomly selected.
# Let's investigate more...

effective_length = max(width_non_zero)
r_first_non_zero_class = stats.pearsonr(train_labels.iloc[:, 1][0:plot_length], first_non_zero)[0]
r_last_non_zero_class = stats.pearsonr(train_labels.iloc[:, 1][0:plot_length], last_non_zero)[0]

# oh shit the correlation is positive... One more effort...

fig_non_zero_by_class, ax_non_zero_by_class = plt.subplots(5, 2)
fig_non_zero_by_class.set_size_inches(5, 9)
investigated_classes = range(number_of_classes)
for investigated_class in investigated_classes:
    first_non_zero_investigated_class = []
    for sample in range(plot_length):
        if train_labels.iloc[sample, 1] == investigated_class:
            first_non_zero_investigated_class.append(np.nonzero(np.array(train_data.iloc[sample, 1:]))[0][0])
    my_ax = ax_non_zero_by_class[investigated_class // 2][investigated_class % 2]
    my_ax.plot(range(len(first_non_zero_investigated_class)), first_non_zero_investigated_class, '.g', label='class {}'.format(investigated_class))
    my_ax.legend(loc='best')
fig_non_zero_by_class.show()
