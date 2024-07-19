#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAC only for now because of AppKit

Created on Tue Jun 18 12:47:36 2024

@author: marleymoore
"""
#%% IMPORTS
import pygame
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import path as mpath
import random
from circle_theorems import circle
import AppKit 

#%% GLOBALS
SCREEN_DIMS = AppKit.NSScreen.screens()[0].frame()
GAME_SCREEN_SIZE = 0.9 * SCREEN_DIMS.size.height #screen height and width

FIGSIZE = (10,10)
DPI = GAME_SCREEN_SIZE/FIGSIZE[0]

# Get working directory
WDR = os.path.dirname(os.path.realpath('Six_Circles.py'))

# Define the colours of game
INCORRECT_COL = (255,0,0) 
CORRECT_COL = (0,255,0)

RAD = 5 #radius of circle on plot
SEP = 1 #separation of circles on plot

#%% FUNCTIONS
def draw_score(surface, text, x, y):
    """
    Draws the score counter to the screen

    Parameters
    ----------
    surface : pygame.Surface
        screen to blit to.
    text : str
        the score text
    x : int
        which pixel to start the text.
    y : int
        which pixel to start the text.

    Returns
    -------
    None.

    """
    font = pygame.font.SysFont('Bauhaus 93', 60)
    img = font.render(text, True, (0,0,0))
    surface.blit(img, (x,y))
    
def plot_6_circles(r, s):
    """
    Plots 6 circles, each containing a random circle theorem, and saves the
    figure. This figure will then be loaded by pygame.
    
    Parameters
    ----------
    r : float
        radius of circles
    s : float
        separation of circles
    
    Returns
    -------
    None.

    """
    x_lim = 6*r + 4*s
    y_lim = x_lim
    
    # Establish figure
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(xmin=0, xmax=x_lim)
    ax.set_ylim(ymin=0, ymax=y_lim)
    ax.set_axis_off()
    ax.axes.set_aspect('equal')
    
    # Establish positions
    pos_DF = pd.DataFrame(columns = ['x', 'y'])
    
    pos_DF.loc[0,'x'] = r+s
    pos_DF.loc[0,'y'] = 3*r+2*s
    pos_DF.loc[1,'x'] = 3*r+2*s
    pos_DF.loc[1,'y'] = 3*r+2*s
    pos_DF.loc[2,'x'] = 5*r+3*s
    pos_DF.loc[2,'y'] = 3*r+2*s
    pos_DF.loc[3,'x'] = r+s
    pos_DF.loc[3,'y'] = r+s
    pos_DF.loc[4,'x'] = 3*r+2*s
    pos_DF.loc[4,'y'] = r+s
    pos_DF.loc[5,'x'] = 5*r+3*s
    pos_DF.loc[5,'y'] = r+s
    
    # Shuffle the dataframe, so that a random theorem is drawn in each circle
    pos_DF = pos_DF.sample(frac=1).reset_index(drop=False)
        
    # Initiate all 6 circles in their positions
    circ1 = circle(pos_DF.loc[0]['x'], pos_DF.loc[0]['y'], r)
    circ2 = circle(pos_DF.loc[1]['x'], pos_DF.loc[1]['y'], r)
    circ3 = circle(pos_DF.loc[2]['x'], pos_DF.loc[2]['y'], r)
    circ4 = circle(pos_DF.loc[3]['x'], pos_DF.loc[3]['y'], r)
    circ5 = circle(pos_DF.loc[4]['x'], pos_DF.loc[4]['y'], r)
    circ6 = circle(pos_DF.loc[5]['x'], pos_DF.loc[5]['y'], r)
    
    # Plot the circles
    circ1_patch = mpatches.Circle((circ1.x,circ1.y),circ1.R, color='k', fill=False)
    circ2_patch = mpatches.Circle((circ2.x,circ2.y),circ2.R, color='k', fill=False)
    circ3_patch = mpatches.Circle((circ3.x,circ3.y),circ3.R, color='k', fill=False)
    circ4_patch = mpatches.Circle((circ4.x,circ4.y),circ4.R, color='k', fill=False)
    circ5_patch = mpatches.Circle((circ5.x,circ5.y),circ5.R, color='k', fill=False)
    circ6_patch = mpatches.Circle((circ6.x,circ6.y),circ6.R, color='k', fill=False)
    
    ax.add_patch(circ1_patch)   
    ax.add_patch(circ2_patch)   
    ax.add_patch(circ3_patch)   
    ax.add_patch(circ4_patch)   
    ax.add_patch(circ5_patch)   
    ax.add_patch(circ6_patch)   
    
    # Choose the seeds
    seed1 = random.randint(0,360)
    seed2 = random.randint(0,360)
    seed3 = random.randint(0,360)
    seed4 = random.randint(0,360)
    seed5 = random.randint(0,360)
    seed6 = random.randint(0,360)
    
    # Circle 1 is angle in semicircle
    semi_circ_coords, right_angled_coords = circ1.angle_in_semicircle(seed1)
    right_angle = mpatches.Polygon(right_angled_coords, facecolor=(0.7,0.7,0.7),
                                   edgecolor=(0.5,0,0), fill=True)
    semi = mpatches.Polygon(semi_circ_coords, color='b', fill=False)
    ax.add_patch(semi)
    ax.add_patch(right_angle)
    
    # Circle 2 is isosceles
    isos_coords, angle_1_coords, angle_2_coords = circ2.isosceles(seed2)
    isos = mpatches.Polygon(isos_coords, color='b', fill=False)
    angle1 = mpatches.Polygon(angle_1_coords, facecolor=(0.7,0.7,0.7),
                                   edgecolor=(0.5,0,0), fill=True)
    angle2 = mpatches.Polygon(angle_2_coords, facecolor=(0.7,0.7,0.7),
                                   edgecolor=(0.5,0,0), fill=True)
    ax.add_patch(isos)
    ax.add_patch(angle2)
    ax.add_patch(angle1)
    
    at_centre_coords = circ3.angle_at_centre(seed3)
    arrow = mpatches.Polygon(at_centre_coords, color='b', fill=False)
    ax.add_patch(arrow)
    
    cyc_quad_coords = circ4.cyclic_quadrilateral(seed4)
    quad = mpatches.Polygon(cyc_quad_coords, color='b', fill=False)
    ax.add_patch(quad)
    
    same_seg_coords = circ5.same_segment_theorem(seed5)
    same = mpatches.Polygon(same_seg_coords, color='b', fill=False)
    ax.add_patch(same)
    
    # Circle 6 is alternate segment theorem
    alt_seg_coords = circ6.alternate_segment_theorem(seed6)
    alt = mpatches.Polygon(alt_seg_coords, color='b', fill=False)
    b, c = circ6.tangent_at_point(seed6)
    ax.plot((b[0],c[0]),(b[1],c[1]), ls='-')
    ax.add_patch(alt)
    
    plt.savefig(WDR + '/PLOT.png')
    plt.close()
    
    return ax, pos_DF

def selection(pos_DF):
    """
    Takes the circle positions, chooses one randomly with the circle theorem
    stored, then returns the coordinates of its centre, with its corresponding
    theorem.

    Parameters
    ----------
    pos_DF : pd.DataFrame
        dataframe containing centre coordinates of the circles.

    Returns
    -------
    centre : (x,y)
        coordinates of centre
    theorem : str
        "index" column of the dataframe represents the circle theorem.

    """
    circle_DF = pos_DF.sample()
    x = circle_DF.iloc[0]['x']
    y = circle_DF.iloc[0]['y']
    
    if circle_DF.index[0] == 0:
        theorem = 'angle in semicircle'
    
    elif circle_DF.index[0] == 1:
        theorem = 'isosceles'
    
    elif circle_DF.index[0] == 2:
        theorem = 'angle at centre is twice angle at circumference'
    
    elif circle_DF.index[0] == 3:
        theorem = 'opposite angles of cyclic quadrilateral sum to 180º'
    
    elif circle_DF.index[0] == 4:
        theorem = 'same segment theorem'
    
    elif circle_DF.index[0] == 5:
        theorem = 'alternate segment theorem'
    
    return x, y, theorem

def transform_circle_to_pygame(x, y, ax):
    """
    Pygame coordinates start from (0,0) in the top left. Bottom right is (W,H)
    Matplotlib pixels start from (0,0) in the bottom left. Top right is (W,H).
    Converts the x,y coordinates from the graph into pixel coordinates to be
    used on the Pygame screen.

    Parameters
    ----------
    x : float
        x pos of circle centre
    y : float
        y pos of circle centre
    bbox_extents : np.array
        contains the bbox coords of plot
    
    Returns
    -------
    x : float
        x pos of circle in pygame coords
    y : float
        y pos of circle in pygame coords
        
    """
    bbox_extents = ax.bbox.extents
    plot_width = bbox_extents[2]-bbox_extents[0] #pixels
    plot_height = bbox_extents[3]-bbox_extents[1] 
    figure_size = (DPI*FIGSIZE[0]) #game screen size is the same as figure size
    
    # In coordinates of plot
    x_prop = x / ax.axes.get_xlim()[-1] #proportion of way along x axis where circle centre is, in plot coordinates
    y_prop = y / ax.axes.get_ylim()[-1]
    
    # In pixels of plot
    x_pixel_coordinate = x_prop * plot_width #x pixel of circle from left of plot
    y_pixel_coordinate = y_prop * plot_height #y pixel of circle from bottom of plot
    
    # In pixels of matplotlib figure
    x_pixel_coordinate += bbox_extents[0] #x pixel of circle from left of figure
    y_pixel_coordinate += bbox_extents[1] #y pixel of circle from bottom of figure
    
    # In pixels of pygame
    scale = GAME_SCREEN_SIZE / figure_size
    x_pixel = scale * x_pixel_coordinate #x pixel of circle in pygame screen
    y_pixel = GAME_SCREEN_SIZE - (scale * y_pixel_coordinate) #pygame y coordinates are 0 at the top of screen
    
    # Scale radius
    r_prop = RAD / ax.axes.get_xlim()[-1]
    r_pygame = r_prop * plot_width * scale #calculate the radius of circle in pixels on pygame screen
    
    return x_pixel, y_pixel, r_pygame

#%% OBJECTS
class circle_button():
    def __init__(self, surface, x, y, R):
        self.surface = surface
        self.x = x
        self.y = y
        self.R = R
        
    def within_circle(self, input_x, input_y):
        """
        Checks whether some input x,y coordinates are within the region enclosed by
        the circle in question.

        Parameters
        ----------
        input_x : int
            x position of point to check.
        input_y : int
            y position of point to check.

        Returns
        -------
        bool : 
            whether or not the point is contained.

        """
        circle = mpath.Path.circle(center=(self.x, self.y), radius=self.R)
        point = (input_x, input_y)
        
        return circle.contains_point(point)

    def draw_button(self, mouse_position):
        """
        Draws a circular button, which is drawn in INCORRECT_COL if mouse
        position is outside the area - returns False; and in CORRECT_COL if
        mouse position is inside the area - returns True.

        Parameters
        ----------
        mouse_position : tuple
            x, y

        Returns
        -------
        bool

        """
        #update the colours to RGB-A
        incorrect_colour_alpha = INCORRECT_COL + (50,)
        correct_colour_alpha = CORRECT_COL + (50,)
        
        target_rect = pygame.Rect((self.x, self.y), (0, 0)).inflate((self.R * 2, self.R * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        
        if not self.within_circle(mouse_position[0], mouse_position[1]):
            pygame.draw.circle(shape_surf, incorrect_colour_alpha,
                               (self.R, self.R), self.R)
            self.surface.blit(shape_surf, target_rect)
            # print('miss')
            return False
            
        else:
            pygame.draw.circle(shape_surf, correct_colour_alpha,
                               (self.R, self.R), self.R)
            self.surface.blit(shape_surf, target_rect)
            # print('hit')
            return True
        
#%% GAME
def main():
    # Initiate pygame
    pygame.init() 
    pygame.font.init()
    my_font = pygame.font.SysFont('Bauhaus 93', 30)
    
    # Establish the screen 
    pygame.display.set_caption("CIRCLE THEOREMS") #title of screen
    screen = pygame.display.set_mode((GAME_SCREEN_SIZE, GAME_SCREEN_SIZE)) #pygame Surface
    
    # Set start score
    score = 0
    
    # Main loop
    running = True
    
    while running:
        pygame.time.delay(10)
        pygame.time.wait(1000)

        complete = False  
        ax, position_DF = plot_6_circles(RAD, SEP) #radius, separation
        
        centre_x, centre_y, theorem = selection(position_DF)        
        pgx, pgy, pgr = transform_circle_to_pygame(centre_x, centre_y, ax)
        text_surface = my_font.render(theorem, False, (0, 0, 0))
        
        graph_image = pygame.image.load('PLOT.png') #loads the graph as an image
        graph_image = pygame.transform.scale(graph_image, (GAME_SCREEN_SIZE, GAME_SCREEN_SIZE))
        
        while not complete and running:
            screen.blit(graph_image, (0, 0))
            screen.blit(text_surface, (0, 40))
            draw_score(screen, str(score), 5,5)
        
            # Exit button
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                   running = False 
                   
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    button = circle_button(screen, pgx, pgy, pgr)
                    if button.draw_button(mouse_pos):
                        score += 1
                        complete = True
                    else:
                        score -= 1
                        complete = True
                    
            pygame.display.update()
    
    pygame.quit()   
    
if __name__ == '__main__':
    main()

