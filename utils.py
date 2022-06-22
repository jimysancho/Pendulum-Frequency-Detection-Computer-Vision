import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as font_manager

font = font_manager.FontProperties(family='Helvetica',
                                          weight='bold',
                                          style='normal', size=16)


def normalize(data):
    data -= np.mean(data)
    data /= np.max(data)
    return data

def stats(parameter, distribution):
    d_p = (np.max(parameter) - np.min(parameter)) / (len(parameter) - 1)
    most_probable = parameter[np.argmax(distribution)]
    mean = np.sum(parameter * distribution * d_p)
    var = np.sum((parameter - mean) ** 2 * distribution * d_p)
    deviation = np.sqrt(var)
    return (most_probable, mean, deviation)    

    
class PendulumBayes:
    
    def __init__(self, init_ranges={'w': (0.1, 1.0), 
                                    'A': (0.0, 0.3), 
                                    'B': (-1.0, -0.5)}, 
                 lengths={'w': 10, 'A': 10, 'B': 10},
                 d_parameter_min=0.05, iterations=20,
                 want_to_iterate=False, step=0.1,
                 width_denominator=2, N=2, M=2, M_max=4, 
                 n_max=np.inf, iterations_growth=25):
        
        """
        init_ranges: dict. It corresponds to the intial ranges in which we'll look for the optimal values for the pendulum. 
        lengths: dict. The length of the linspaces of the different parameter's ranges. 
        d_parameter_min: float. The minimum value to create the ranges to look for the optimal values. 
        iterations: int. The ratio of iterations (every nth data) in which we reset the ranges and the self.q function. 
        want_to_iterate: bool. If true, we reset the function q. 
        step: float. Every time we are getting closer to the optimal values, we increase the number by which the d_w will be divided 
                     to make the ranges smaller. 
        width_denominator: float. It determines the first d_parameter. 
        N: float or int. It determines the threshold. 
        M: float or int. It determines the ratio in which we decrease the range. 
        M_max: float or int. Minimum ratio to decrease the range. 
        n_max: int. Number from which we start to reset the q less often. 
        iterations_growth: int. When n_max is reached, how less often we want to reset the q. 
        
        """
        
        self.init_ranges = init_ranges
        self.lengths = lengths
        self.d_min = d_parameter_min
        self.width_denominator = width_denominator
        self.N = N
        self.M = M
        self.iterations = iterations
        self.want_to_iterate = want_to_iterate
        self.step = step
        self.M_max = M_max
        self.M_min = M
        self.n_max = n_max
        self.iterations_growth = iterations_growth
                
        # frequency variables
        self.w_info = {0: init_ranges['w']}
        self.w_length = lengths['w']
        self.d_w = self.width()[0]
        self.d_w_max = self.width()[0]
        
        # sin amplitude variables
        self.sin_info = {0: init_ranges['A']}
        self.sin_length = lengths['A']
        self.d_a = self.width()[1]
        self.d_a_max = self.width()[1]
        
        # cos amplitude variables
        self.cos_info = {0: init_ranges['B']}
        self.cos_length = lengths['B']
        self.d_b = self.width()[2]
        self.d_b_max = self.width()[2]
        
        # structures to store the information as we iterate
        self.optimal_values = []
        self.q = np.zeros((self.w_length, self.sin_length, self.cos_length))
        
    @staticmethod
    def check_difference(d_n, d_m, threshold):
        diff = abs(d_n - d_m)
        if diff <= threshold:
            return True
        return False
        
    @staticmethod
    def linspace(param_range, length, c_p=None, index=None):

        if c_p is None and index is None:
            return np.linspace(param_range[0], param_range[1], length)
        
        begin, end = param_range
        first_term = np.linspace(begin, c_p, num=index, endpoint=False)
        second_term = np.linspace(c_p, end, num=(length-index))
        return np.concatenate((first_term, second_term))
    
    @staticmethod
    def Q(data, w, A, B, t):
        teor = A * np.sin(w * t) + B * np.cos(w * t)
        diff = (teor - data) ** 2
        return diff

    def update_range(self, distributions, parameters):
        if len(distributions) != len(parameters):
            raise LookupError('Length of distributions does not match the lenght of the parameters')

        values = []
        for distribution, parameter in zip(distributions, parameters):
            index, = np.where(distribution == np.max(distribution))
            max_param = np.mean(parameter[index])
            values.append(max_param)
        return values
    
    def width(self):
        widths = []
        for key in self.init_ranges.keys():
            width = abs((np.max(self.init_ranges[key]) - np.min(self.init_ranges[key]))) / self.width_denominator
            widths.append(width)
        return widths
    
    def recursive_q(self, n, new_value, values={}):
        if n < 1:
            return 0
        
        if n in values:
            return values[n]
        
        values[n] = self.recursive_q(n-1, new_value, values) + new_value
        return values[n]

    def probability_distribution(self, parameter_name, parameter, N):
        possible = {'w', 'A', 'B'}

        if parameter_name not in possible:
            raise LookupError(f'Not valid. Try one of {possible}')
        else:
            AXES = {'w': 0, 'A': 1, 'B':2}
            log = - (N / 2) * np.log(self.q)
            log_norm = log - np.max(log)
            d_i = (np.max(parameter) - np.min(parameter)) / (len(parameter) - 1)
            margin = AXES[parameter_name]
            P = np.exp(log_norm)
            if margin == 0:
                p_i = np.sum(P, axis=2)
                p_i = np.sum(p_i, axis=1)
            elif margin == 1:
                p_i = np.sum(P, axis=2)
                p_i = np.sum(p_i, axis=0)
            else:
                p_i = np.sum(P, axis=1)
                p_i = np.sum(p_i, axis=0)
            return p_i / (np.sum(p_i * d_i))
    
    def improve_parameters(self):
        
        point = None
        n = 0
        m = 1
        k = n
        
        while True:

            thresholds = [self.d_w , self.d_a , self.d_b]
            point = yield point
                
            if point is None:
                break
            
            w_range = self.w_info[n]
            sin_range = self.sin_info[n]
            cos_range = self.cos_info[n]
            
            w = self.linspace(w_range, self.w_length)
            A = self.linspace(sin_range, self.sin_length)
            B = self.linspace(cos_range, self.cos_length)
            
            w_index, sin_index, cos_index = np.meshgrid(np.arange(0, len(w)), 
                                                        np.arange(0, len(A)), 
                                                        np.arange(0, len(B)), 
                                                        indexing='ij')
            """
            for i, w_i in enumerate(w):
                for j, a_j in enumerate(A):
                    for s, b_s in enumerate(B):
                        Q_n = self.Q(point[0], w_i, a_j, b_s, point[1])
                        self.q[i, j, s] = self.recursive_q(n+1, Q_n)
            """
            
            # this part of the code also works like this: self.q = self.recursive(n+1, Q_n)
            Q_n = self.Q(point[0], w[w_index], A[sin_index], B[cos_index], point[1])
            self.q += Q_n
            
            p_w = self.probability_distribution('w', w, k+1)
            p_a = self.probability_distribution('A', A, k+1)
            p_b = self.probability_distribution('B', B, k+1)
            
            w_n, a_n, b_n = self.update_range((p_w, p_a, p_b), (w, A, B))
            self.optimal_values.append((w_n, a_n, b_n))

            w_m, a_m, b_m = (self.optimal_values[n-m] if (n-m) > 1 else (0, 0, 0))

            if n >= self.n_max:
                self.iterations = self.iterations_growth
                self.N += 2

            if not n % self.iterations and n != 0 and self.want_to_iterate:
                
                check_differences = [self.check_difference(d_i, d_j, i / self.N) for (d_i, d_j, i)
                                    in ((w_n, w_m, thresholds[0]), (a_n, a_m, thresholds[1]), (b_n, b_m, thresholds[2]))]
                
                if all(check_differences):
                    # we reset the matrix q and modify the new ranges
                    k = 0
                    self.q = 0
                    self.d_w /= self.M
                    self.d_a /= self.M
                    self.d_b /= self.M
                    self.M += self.step
                    if self.M >= self.M_max:
                        self.step = 0
                        self.M = self.M_max
                        
                    # the d_parameters can not be lower than the minimum given in the initialization. 
                    if self.d_w < 2 * self.d_min:
                        self.d_w = 2 * self.d_min
                    if self.d_a < 2 * self.d_min:
                        self.d_a = 2 * self.d_min
                    if self.d_b < 2 * self.d_min:
                        self.d_b = 2 * self.d_min
                            
                else:
                    
                    self.q = 0
                    # if the threhsolds are not fullfilled, we increment the range again. 
                    self.M += self.step
                    self.d_w *= self.M
                    self.d_a *= self.M
                    self.d_b *= self.M
                    self.M -= self.step
                    
                    # we don't want to increment forever. There is a limit. 
                    if self.d_w > 1:
                        self.d_w = 1.0
                    if self.d_a > 1:
                        self.d_a = 1
                    if self.d_b > 1:
                        self.d_b = 1

                    # the M parameter can not be too small
                    if self.M <= self.M_min:
                        self.M = self.M_min
                
                # we don't want to considerate negative frequencies
                if (w_n - self.d_w) <= 0:
                        w_range = (0, w_n + self.d_w)
                else:
                    w_range = (w_n - self.d_w, w_n + self.d_w)
                    
                sin_range = (a_n - self.d_a, a_n + self.d_a)
                cos_range = (b_n - self.d_b, b_n + self.d_b)
                
           
            if self.d_w <= self.d_min:
                self.d_w = self.d_min
            
            if self.d_a <= self.d_min:
                self.d_a = self.d_min
                
            if self.d_b <= self.d_min:
                self.d_b = self.d_min
            
            self.w_info[n+1] = w_range
            self.sin_info[n+1] = sin_range
            self.cos_info[n+1] = cos_range
            
            yield (p_w, p_a, p_b, self.w_info[n+1], 
                   self.sin_info[n+1], self.cos_info[n+1], 
                   self.optimal_values[n], abs(w_m - w_n))
            
            n += 1          
            k += 1 
            
def plot(distribution, parameter, distribution_name, real_val):
    names = {'frecuencia': 'Frecuencia', 'A': 'A', 'B': 'B'}
    
    if distribution_name not in names.keys():
        raise LookupError(f'{distribution_name} is not valid. Try one of {names.keys()}')
    
    N = 100
    lw = 10
    color = 'red'
    k = 5
    alpha = 0.0
    d_alpha = k / N
    title = (f'Distribución de probabilidad de la {names[distribution_name]}' 
             if distribution_name == 'frecuencia'
             else f'Distribución de probabilidad de la amplitud {names[distribution_name]}')

    fig = plt.figure(figsize=(12, 7))

    for n, p in enumerate(distribution[::k]):
        alpha += d_alpha
        if alpha > 1:
            alpha = 1

        if n == 5:
            label = 'Iteraciones anteriores'
        elif n == N // k - 1:
            lw=20
            color = 'blue'
            point = parameter[np.argmax(p)], np.max(p)
            label = 'Distribución final'
        else:
            label = None

        plt.scatter(parameter, p, c=color, s=lw, label=label, alpha=alpha)
    plt.axvline(x=real_val, linestyle='--', color='black', label=f'{names[distribution_name]} real')

    plt.scatter(point[0], point[1], edgecolor='black', s=150, c='green', 
                label=f'{names[distribution_name]} final')

    plt.title(title, font='helvetica', fontsize=20)
    plt.xlabel(f'{names[distribution_name]}', fontsize=15, font='helvetica', fontstyle='italic')
    
    plt.ylim(-0.1, np.max(p) + 0.2 * np.max(p))
    plt.legend(prop=font)
    plt.show()
            
                
if __name__ == "__main__":
    
    path = '/Users/Jaime/Desktop/TFG/Pendulo/Datos/2-intento-foucault.csv'
    file = pd.read_csv(path)
    x = file['y'][:5000]
    data = x[:len(x) - 1]
    data = normalize(data)
    dt = 720 / len(data)
    time = np.arange(0, 720, dt)
    pendulum = PendulumBayes(init_ranges={'w': (1.0, 2.0), 
                                          'A': (-0.5, 0.5), 
                                          'B': (-0.5, 0.5)}, 
                             width_denominator=1,
                             d_parameter_min=5e-5, lengths={'w': 100, 'A': 100, 'B': 100}, 
                             N=4, M=1.3, want_to_iterate=True, iterations=20)
    
    gen = pendulum.improve_parameters()
    gen.send(None)
    N = 100
    k = len(data) // N
    d = data[:1000]
    t = time[:1000]
    print(len(d))
    
    p_w, p_a, p_b = np.zeros((len(d), 100)), np.zeros((len(d), 100)), np.zeros((len(d), 100))
    w_range = {}
    sin_range = {}
    cos_range = {}
    
    s = 0.5
    alpha = 0.05
    d_alpha = 0.09
    for n, point in enumerate(zip(d, t)):
        
        if n == N - 1:
            s = 10
        p_w[n], p_a[n], p_b[n], w_range[n], sin_range[n], cos_range[n], values, diff = gen.send(point)
        if not n % 100:
            plt.scatter(pendulum.linspace(w_range[n], len(p_w[n])), p_w[n], color='red', alpha=alpha)
            alpha += d_alpha
            
        a, b = next(gen)
            
    plt.show()
        
    k = 100
    for n, p in enumerate(p_w[::k]):
        lw = 10
        color = 'red'
        k = 5
        alpha = 0.0
        d_alpha = k / N
        w = pendulum.linspace(w_range[n*k], len(p)) #values[0], np.argmax(p_w))
        alpha += d_alpha
        if alpha > 1:
            alpha = 1

        if n == 5:
            label = 'Iteraciones anteriores'
        elif n == N // k - 1:
            lw=20
            color = 'blue'
            point = w[np.argmax(p)], np.max(p)
            label = 'Distribución final'
        else:
            label = None

        plt.scatter(w, p, c=color, s=lw, label=label, alpha=alpha)
        
    plt.show()
    
    