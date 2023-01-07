import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time as TIME

from utils import PendulumBayes, normalize, stats
import pandas as pd
plt.rcParams['figure.figsize'] = (10, 8)
            
    
def make_title(f, a, b, pend, decimals, i):
    d_w, d_a, d_b = pend.d_w, pend.d_a, pend.d_b
    variables = [f, a, b, d_w, d_a, d_b]
    f, a, b, d_w, d_a, d_b = [np.round(i, decimals=decimals) for i in variables]
    title = fr'$w_{{{i}}}$={f} $\pm$ {d_w}, $A_{{{i}}}$= {a} $\pm$ {d_a}, $B_{{{i}}}$ ={b} $\pm$ {d_b}'
    return title

def data_and_probability_animate(i):
    
    point = data[i], time[i]
    y_vals.append(point[0])
    x_vals.append(point[1])
    information = gen.send(point)
    next(gen)
    
    p_w, p_a, p_b = information[:3]
    w_range, sin_range, cos_range = information[3:6]
    w = pendulum.linspace(w_range, len(p_w))
    a = pendulum.linspace(sin_range, len(p_a))
    b = pendulum.linspace(cos_range, len(p_b))
    freq, A, B = information[-2]

    array = time[:i]
    teor = A * np.sin(freq * array) + B * np.cos(freq * array)
    fig.suptitle(f'$d_w$ = {pendulum.d_w}, $d_a$ = {pendulum.d_a}, $d_b$ = {pendulum.d_b}')
    
    # teórica y datos
    ax[0][0].clear()
    ax[0][0].set_title(fr'$w_{{{i}}}$ = {np.round(freq, decimals=decimals)}, $A_{{{i}}}$ = {np.round(A, decimals=decimals)}, $B_{{{i}}}$ = {np.round(B, decimals=decimals)}')
    ax[0][0].scatter(x_vals, y_vals, c='black', label='Datos experimentales', s=3.0)
    ax[0][0].plot(array, teor, '.-', label='función teórica', color='red', alpha=0.3)
    ax[0][0].set_xlim(-1, len(data))
    ax[0][0].legend()
    
    # probabilidad de la frecuencia
    ax[0][1].clear()
    ax[0][1].set_title(fr'distribución de probabilidad $p_w$. $w_{{{i}}}$ = {np.round(freq, decimals=decimals)}')
    ax[0][1].plot(w, p_w, lw=1, color='black')
    #ax[0][1].set_xlim(0.05, 0.4)
    
    # probabilidad primer amplitud
    ax[1][0].clear()
    ax[1][0].set_title(fr'distribución de probabilidad $p_a$. $A_{{{i}}}$ = {np.round(A, decimals=decimals)}')
    ax[1][0].plot(a, p_a, lw=1, color='black')
    #ax[1][0].set_xlim(-1.0, 1.0)
    
    # probabilidad seguna amplitud
    ax[1][1].clear()
    ax[1][1].set_title(fr'distribución de probabilidad $p_b$. $B_{{{i}}}$ = {np.round(B, decimals=decimals)}')
    ax[1][1].plot(b, p_b, lw=1, color='black')
    
def frequency_probability(i):
    point = data[i], time[i]
    y_vals.append(point[0])
    x_vals.append(point[1])
    information = gen.send(point)
    next(gen)
        
    p_w = information[0]
    w_range = information[3]
    w = pendulum.linspace(w_range, len(p_w))
    
    most_probable, _, var = stats(w, p_w)
    
    plt.cla()
    plt.title(rf'$w_{{{i}}}$ = {np.round(w[np.argmax(p_w)], decimals=3)} $\pm {np.round(var, decimals=3)}$ ', font='helvetica', 
            fontsize=20)
    plt.scatter(w, p_w, s=20, color='red', alpha=0.8)
    plt.scatter(most_probable, np.max(p_w), edgecolor='black', color='blue', s=50)
    plt.xlabel('Frecuencia', font='helvetica', fontsize=20)
    plt.ylabel(r'p(w)', font='helvetica', fontsize=20)
    plt.ticklabel_format(useOffset=False)   
    plt.xlim(3.0, 4.0)
    
def data_animate(i):
    
    j = i 
    point = data[j], time[j]
    y_vals.append(point[0])
    x_vals.append(point[1])
    information = gen.send(point)

    next(gen)
    freq, A, B = information[-2]

    diff = A * np.sin(freq * time[j]) + B * np.cos(freq * time[j])
    
    array = time[:j]
    teor = A * np.sin(freq * array) + B * np.cos(freq * array)
    guess_array = time[j+1:]
    guess_curve = A * np.sin(freq * guess_array) + B * np.cos(freq * guess_array)
    
    f_min, f_max = (freq - pendulum.d_w, freq + pendulum.d_w)
    A_min, A_max = (A - pendulum.d_a, A + pendulum.d_a)
    B_min, B_max = (B - pendulum.d_b, B + pendulum.d_b)
        
    guess_min_curve = A_min * np.sin(f_min * guess_array) + B_min * np.cos(f_min * guess_array)
    guess_max_curve = A_max * np.sin(f_max * guess_array) + B_max * np.cos(f_max * guess_array)
    
    plt.cla()
    plt.scatter(x_vals, y_vals, s=20.0, c='black', label='Datos experimentales')
    plt.plot(array, teor, '-o', label='Función teórica', 
             color='blue', alpha=0.7, 
             lw=3.0)
    plt.scatter(guess_array, guess_curve, s=30.0, 
                label='Futura curva', alpha=0.3, c='red')
    plt.fill_between(guess_array, guess_max_curve, guess_min_curve, 
                     color='gray', alpha=0.2, label='Incertidumbre')
    plt.legend(loc=(-0.1, 0.0))
    plt.title(make_title(freq, A, B, pendulum, decimals, i))
    #plt.xlim(len(data[:i]) - 30, len(data[:i]) + 100)
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, time[i+200])
           
if __name__ == "__main__":
    
    # files and data arrays including time
    path = './Datos/pendulo-CM.csv'
    file = pd.read_csv(path)
    MAX = 20355
    num_data = 2000
    x = file['y'][:num_data]
    minutes = (num_data * 12) / MAX
    N = len(x)
    total_time = minutes * 60
    dt = total_time / N #
    time = np.arange(0, N * dt, dt)
    data = normalize(x)
    test = np.inf
    ALPHA = (0.01,) 
    max_values = []
    max_freq, max_p = [], []

    # different dt
    dt1 = (12 * 60) / MAX # real dt
    dt2 = dt # bad dt
    t1 = np.arange(0, N * dt1, dt1)
    t2 = np.arange(0, N * dt2, dt2)
    points_1 = []
    points_2 = []
    
    # pendulum instance of PendulumBayes class
    N, M, K = (100, 100, 100)
    want_to_iterate = True
    if want_to_iterate:
        init_ranges={'w': (0.01, 2.0), 
                     'A': (-0.5, 0.5), 
                     'B': (-0.5, 0.5)}
    else:
        init_ranges={'w': (3.0, 4.0), 
                     'A': (-0.1, 0.1), 
                     'B': (0.7, 1.0)}
        
    pendulum = PendulumBayes(init_ranges=init_ranges, 
                             lengths={'w': N, 'A': M, 'B': K}, 
                             d_parameter_min=1e-4, 
                             width_denominator=2, N=4.0, M=1.1, step=0.1, M_max=4, 
                             want_to_iterate=want_to_iterate, iterations=20, n_max=test, iterations_growth=100)
    
    decimals = 4
    # generator (improve_parameters() method of the pendulum instance)
    gen = pendulum.improve_parameters()
    
    # we send None in order to initialize the generator. 
    gen.send(None)
    
    # animation tools
    function = data_animate
    both = False # true for see the whole animation
    if function == data_and_probability_animate:
        both = True
        
    if both:
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 5), 
                               constrained_layout=True)
    animate = True
    
    
    if animate:
        TIME.sleep(2)
        x_vals = []
        y_vals = []
        ani = FuncAnimation(plt.gcf(), function, interval=100)        
        plt.show()
                    