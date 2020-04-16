import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import config, get_data#, results
import ScipyBaseModel as spbm

path = "Iteration_History/SLSQPpenalty_iters.csv"
df = pd.read_csv(path)

def update(i):
    x = df[str(i)]
    T_hist = spbm.get_T(x, time, net_load, config)

    ax1.clear()
    ax1.plot(time.values, net_load, label='Net Load')
    ax1.plot(time.values, x, label='Nuclear optimized')
    if guess is not None:
        ax1.plot(time.values, guess, label='Nuclear Initial')
    ax1.set_ylabel('Energy (MW)')
    ax1.legend(loc='upper left')

    ax2.clear()
    ax2.plot(time_limits, [config['tes_min_t'], config['tes_min_t']], 
            '--r', label='Temperature Constraints')
    ax2.plot(time_limits, [config['tes_max_t'], config['tes_max_t']], '--r')
    ax2.plot(time.values, T_hist, label='Optimized TES')
    if guess is not None:
        T_hist_compare = spbm.get_T(guess, time, net_load, config)
        ax2.plot(time.values, T_hist_compare, label='TES Initial')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_xlabel('Time')
    ax2.legend(loc='upper left')
    plt.gcf().autofmt_xdate()


# res = results(df[0], config)
time, net_load = get_data(config['month'], config['year'])
guess = np.ones(len(time))*config['guess_coef']
x = guess

T_hist = spbm.get_T(x, time, net_load, config) #res['T']



fig, (ax1, ax2) = plt.subplots(nrows=2, figsize =(9, 7))

ax1.plot(time.values, net_load, label='Net Load')
ax1.plot(time.values, x, label='Nuclear optimized')
if guess is not None:
    ax1.plot(time.values, guess, label='Nuclear Initial')
ax1.set_ylabel('Energy (MW)')
ax1.legend(loc='upper left')

time_limits = [time.values[0], time.values[-1]]
ax2.plot(time_limits, [config['tes_min_t'], config['tes_min_t']], 
            '--r', label='Temperature Constraints')
ax2.plot(time_limits, [config['tes_max_t'], config['tes_max_t']], '--r')
ax2.plot(time.values, T_hist, label='Optimized TES')
if guess is not None:
    T_hist_compare = spbm.get_T(guess, time, net_load, config)
    ax2.plot(time.values, T_hist_compare, label='TES Initial')
ax2.set_ylabel('Temperature (K)')
ax2.set_xlabel('Time')
ax2.legend(loc='upper left')
plt.gcf().autofmt_xdate()
# plt.savefig(f'saved_plots/{optimizer}-{opt_type}-{date}.png')
# print(f"Figure saved at: saved_plots/{optimizer}-{opt_type}-{date}.png")

Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

anim = animation.FuncAnimation(fig, update, frames=len(df.keys()), interval = 1/10)
# anim.save('GA.mp4', writer=writer, dpi=100)

plt.show()