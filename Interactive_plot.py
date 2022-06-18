import cv2 as cv
import numpy as np
from umucv.stream import autoStream
import matplotlib.pyplot as plt

class CV_PLOT:
    
    def __init__(self, image, trajectory_mask_size=(500, 500), point_size=1):
        
        self.image = image
        self.mask = np.zeros_like(image, dtype=np.uint8)
        
        self.neighbors = point_size // 2
        self.shape = image.shape[:2]
        
        rows, cols = trajectory_mask_size
        shape = (rows + rows // 4, 2 * cols)
        self.trajectory_mask = np.hstack((np.zeros(shape, dtype=np.uint8), 
                                          np.zeros(shape, dtype=np.uint8)))
        
    @property
    def display_mask(self):
        cv.imshow('mask', self.mask)
        
    @property
    def display_trajectory(self):
        cv.imshow('Trajectory mask', self.trajectory_mask)
           
    def update(self, points, draw=True):
        
        R, C = self.shape
        size = self.neighbors
        for point in points:
            
            # OJO CON EL ORDEN: COL = X, ROW = Y
            col, row = point
            try:
                self.mask[row - size: row + size + 1, 
                          col - size: col + size + 1] = np.uint8(255)
            except:
                continue
        
        if draw:
            self.display_mask
            
    def trajectory(self, points, live=True, final=True):
        
        R, C = self.trajectory_mask.shape
        time = np.arange(0, len(points))
        points = np.array(points)
        scale_points = points.copy()
        
        w, h = self.shape[::-1]
        scale_factors = (C / w, R / h)
        scale_points = scale_points * scale_factors

        if live:
            x_time = 0
            y_time = C // 2 + 1
            size = self.neighbors
            
            for point in scale_points:
                x, y = point
                x += R // 4
                y += R // 4
                x, y = int(x), int(y)
                x_lims = (x - size, x + size + 1)
                y_lims = (y - size, y + size + 1)
                t_x = (x_time - size, x_time + size + 1)
                t_y = (y_time - size, y_time + size + 1)
                
                self.trajectory_mask[x_lims[0]: x_lims[1],
                                     t_x[0]: t_x[1]] = np.uint8(255)
                    
                self.trajectory_mask[y_lims[0]: y_lims[1],
                                    t_y[0]: t_y[1]] = np.uint8(255)
                x_time += 1
                y_time += 1
                if x_time > C:
                    x_time = 0
                    y_time = C + 1
                    self.trajectory_mask = np.uint8(0)
                else:
                    continue
                    
            self.display_trajectory
                    
        elif not live and final:
            titles = ['Coordenada X', 'Coordenada Y']
            y_labels = ['X', 'Y']
            fig, ax = plt.subplots(nrows=2, figsize=(10, 8), 
                                  constrained_layout = True)
            for index, title in enumerate(titles):
                ax[index].plot(time, points[:, index])
                ax[index].set_title(title)
                ax[index].set_ylabel(y_labels[index])
                ax[index].set_xlabel('time')
                ax[index].xaxis.set_ticklabels([])
                #ax[index].yaxis.set_ticklabels([])
                
            plt.show()
                
        