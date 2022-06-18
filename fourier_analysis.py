import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft
import pandas as pd
from utils import PendulumBayes, normalize
from matplotlib.animation import FuncAnimation
import seaborn as sns
sns.set_theme('paper')

# ojo al dt. Es siempre el mismo, cojamos más datos o no. Podemos cambiarlo para poder analizar qué ocurre. 
def d_t(num_data, max_data):
    minutes = (num_data * 12) / max_data
    N = num_data
    total_time = minutes * 60
    dt = total_time / N 
    return dt
    
def ruido(i):
    
    j = i 
    point = data[j], time[j]
    information = gen.send(point)
    next(gen)
    
    f, A, B = information[-2]
    punto_teorico = A * np.sin(f * time[j]) + B * np.cos(f * time[j])
    diff = point[0] - punto_teorico
    noise.append(diff)
        
    fig.suptitle(f'{j}')
    if fourier:
        ft = fft(noise)
        freq = fftfreq(len(noise), dt)
        max_freq = freq[np.argmax(ft)]
        ax[0].clear()
        ax[0].plot(freq[:len(ft)//2], np.abs(ft)[:len(ft)//2])
        ax[0].set_title(f'TF del ruido, {np.round(np.abs(max_freq), decimals=5)}, $f_N = {np.round(nyquist, decimals=5)}$')
        ax[0].axvline(x=nyquist, color='red', linestyle='--')
   
    else:
        ax[0].scatter(time[j], diff, c='black', s=2.0)
        ax[0].set_xlim(-1, len(data[:i+100]))
        ax[0].set_ylim(y_min, y_max)
        ax[0].set_title('Ruido en función del tiempo')
    
    
    # histogram to represent the error
    ax[1].clear()
    ax[1].hist(noise, bins=np.arange(y_min, y_max+d_y, d_y), histtype='stepfilled', align='mid')
    ax[1].set_title('Histograma del ruido: gaussiano?')

def fourier_transform_of_data(i):
    point = data[i], time[i]
    y_vals.append(point[0])
    x_vals.append(point[1])
    
    plt.cla()
    freq = fftfreq(len(y_vals), dt) * 2 * np.pi 
    fourier_transform = np.abs(fft(y_vals))
    (freqs.append(freq[np.argmax(fourier_transform[:len(fourier_transform)//2])]) if len(fourier_transform) > 1 else 0)
    
    # data
    ax[0].plot(x_vals, y_vals, c='red', alpha=0.7, lw=2.0)
    
    # FT
    ax[1].clear()
    ax[1].plot(freq[:len(freq) // 2], fourier_transform[:len(fourier_transform) // 2])
    ax[1].axvline(x=nyquist * 2 * np.pi, color='red', linestyle='--')
    ax[1].set_title(f'{freq[np.argmax(fourier_transform[:len(fourier_transform)//2])]}' if len(fourier_transform) > 1 else 0)
    
       
if __name__ == "__main__":
    #path = '/Users/Jaime/TFG/Pendulo/Datos/pendulo-CM.csv'
    path = '/Users/Jaime/Desktop/TFG/Pendulo/Datos/1-intento-Foucault.csv'
    file = pd.read_csv(path)
    MAX = 20355
    N = MAX
    x = file['y'][:N]

    dt = d_t(N, MAX)

    time = np.arange(0, N * dt, dt)
    nyquist = 1 / (2 * dt)

    data = normalize(x)
    test = np.inf

    # pendulum instance of PendulumBayes class
    N, M, K = (50, 50, 50)
    want_to_iterate = True
    if want_to_iterate:
        y_min = -0.05
        y_max = 0.05
        d_y = 0.005
    else:
        y_min = -0.5
        y_max = 0.5
        d_y = 0.05
        
    fourier = True

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
                                width_denominator=1, N=4.0, M=1.1, step=0.1, M_max=4, 
                                want_to_iterate=want_to_iterate, iterations=10, n_max=test, iterations_growth=20)

    gen = pendulum.improve_parameters()
    gen.send(None)
    noise = []

    # figures after the animations and the animation itself
    functions = {'ruido': ruido, 'fourier transform of data': fourier_transform_of_data}
    func = 'ruido'
    x_vals = []
    y_vals = []
    freqs = []
    fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
    ani = FuncAnimation(fig, functions[func], interval=100)
    plt.show()

    if func == 'ruido':
        tf = fft(noise)
        freq = fftfreq(len(noise), 1)
        bad_frequencies = tf <= np.max(tf) / 4
        tf[bad_frequencies] = 0

        good_noise = np.real(ifft(tf))

        plt.plot(freq[:len(noise) // 2], np.real(tf[:len(noise) // 2]))
        plt.show()

        plt.scatter(time[:len(good_noise)], good_noise, 
                    label='Filtrado', c='red', alpha=0.5, s=30.0)
        plt.scatter(time[:len(noise)], noise, 
                    label='original', c='blue', alpha=0.5, s=30.0)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.show()

    elif func == 'fourier transform of data':
        plt.scatter(time, data, label='Original', alpha=0.7, s=20.0)
        plt.plot(time, np.cos(freqs[-1] * time), label='Filtrado', alpha=0.5, lw=2.0, color='blue')
        plt.show()
        