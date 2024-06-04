#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:57:43 2024

@author: marleymoore

Python script to run a game using Pygame to be used in teaching the concepts of
inequalities to secondary-school aged students. To play the game, the student
must click the correct part of the graph each round, which satisfies two 
randomly generated inequalities, tallying up the score.
"""

#%% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import path as mpath
import pygame
import random

#%% GLOBALS
SCREEN_SIZE = 800 #screen height and width
W = SCREEN_SIZE 
H = SCREEN_SIZE 

# Define the colours of the game
BUTTON_COL = (255,0,0) 
BUTTON_FLASH_COL = (0,255,0)
BUTTON_HOVER_COL = (0,0,255)
NO_VALID_REGION_COL = (135,206,235)

DPI = 200
FIGSIZE = (10,10)

# Set working directory
WDR = '/Users/marleymoore/Desktop/Jobs/MATHS_GAMES/INEQUALITIES'

#%% GRAPHING
def generator(lower, upper):
    """
    Parameters
    ----------
    lower : int
        Lower bound for RNG.
    upper : int
        Upper bound for RNG.

    Yields
    ------
    int
        While being called, yields random integers between the bounds.
    """
    
    while True:
        yield random.randint(lower, upper)

def random_inequality():
    """
    Returns
    -------
    coefficients_array : arr
        array containing 3 randomly generated numbers to be used as the 
        coefficients of the inequality.
    symbol : int
        Decides whether less than or greater than. 0 is <, 1 is >.
    """
    coefficients_array = []
    for n in range(3):
        number = generator(-10, 10)
        coefficients_array.append(next(number))
    
    #we never want to produce an inequality with zero dimensions
    if coefficients_array[0] == 0 and coefficients_array[1] == 0:
        coefficients_array[0] = 1
    
    symbol = random.randint(0,1) #0 for <, 1 for >
    
    return coefficients_array, symbol

def linear_equation(coefficients):
    """
    Solves ax + by = c for the x and y values, given the coefficients.
    The x values are usually integer steps from -10 to 10.

    Parameters
    ----------
    coefficients : arr of length 3
        Contains the coefficients of an inequality.

    Returns
    -------
    x_list : arr
        list of x coordinates for plot.
    y_list : arr
        list of y coordinates for plot.
    """
    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]
    
    y_list = []
    x_list = []
    
    if b!=0:
        x_list = np.arange(-10,11)
        for x in x_list:
            y = (c - a*x)/b
            y_list.append(y)
    
    else:
        y_list = np.arange(-10,11)
        assert a!=0
        for y in y_list:
            x = c/a
            x_list.append(x)
    
    return x_list, y_list

def inequality_text(coefficients, sign):
    """
    Produces the text for the plot.
    
    Parameters
    ----------
    coefficients : arr of length 3
        
    sign : int

    Returns
    -------
    text : str
        Text to be plotted on the graph.
    """

    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]
    
    if sign == 0:
        if a == 0:
            assert b!=0
            if c == 0:
                text = 'y < 0'
            else:
                text = 'y < {0:.1g}'.format(c/b)
        
        else:
            if b == 0:
                assert a!=0
                if c == 0:
                    text = 'x < 0'
                else:
                    text = 'x < {0:.1g}'.format(c/a)
            
            else:
                if c == 0:
                    text = '{}y < {}x'.format(b,-a)
                else:
                    if c<0:
                        text = '{}y < {}x {}'.format(b,-a,c)
                    else:
                        text = '{}y < {}x + {}'.format(b,-a,c)
    
    else:
        assert sign == 1
        if a == 0:
            assert b!=0
            if c == 0:
                text = 'y > 0'
            else:
                text = 'y > {0:.1g}'.format(c/b)
        
        else:
            if b == 0:
                assert a!=0
                if c == 0:
                    text = 'x > 0'
                else:
                    text = 'x > {0:.1g}'.format(c/a)
            
            else:
                if c == 0:
                    text = '{}y > {}x'.format(b,-a)
                else:
                    if c<0:
                        text = '{}y > {}x {}'.format(b,-a,c)
                    else:
                        text = '{}y > {}x + {}'.format(b,-a,c)
            
    return text

def plot_graphs(x_list_a, y_list_a, x_list_b, y_list_b, coefficients_a,
                coefficients_b, sign_a, sign_b):
    """
    Plots two inequalities on a graph, with description text, saves the fig to 
    working directory.

    Parameters
    ----------
    x_list_a : arr
        list of x coordinates for first inequality.
    y_list_a : arr
        list of y coordinates for first inequality.
    x_list_b : arr
        list of x coordinates for second inequality.
    y_list_b : arr
        list of y coordinates for second inequality.
    coefficients_a : arr
        arr of length 3, which contains the coefficients of the first 
        inequality.
    coefficients_b : arr
        arr of length 3, which contains the coefficients of the second 
        inequality.
    sign_a : int
    sign_b : int
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(xmin=-10, xmax=10)
    ax.set_ylim(ymin=-10, ymax=10)
    ax.xaxis.grid(which='major')
    ax.yaxis.grid(which='major')
    
    txt_a = inequality_text(coefficients_a, sign_a) 
    txt_b = inequality_text(coefficients_b, sign_b) 
    
    # Plot first inequality
    ax.plot(x_list_a, y_list_a, 'b', linewidth=4)
    
    # Plot second inequality
    ax.plot(x_list_b, y_list_b, 'b', linewidth=4)
    
    # Plot text
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    ax.text(-9.6, 9, txt_a, fontsize=20, bbox=props)
    ax.text(-9.6, 7.5, txt_b, fontsize=20, bbox=props)
    

    # Plot the main axes
    ax.plot(np.arange(-10,11), np.zeros(21), 'k', linewidth=2)
    ax.plot(np.zeros(21), np.arange(-10,11), 'k', linewidth=2)
    
    plt.savefig(WDR + '/PLOT.png')
    
#%% ESTABLISHING COORDINATES OF VALID REGION
def valid_region_coordinates(coefficients, sign):
    """
    Finds the coordinates of the vertices of the polygon that comprises the 
    valid region of the graph defined by one inequality

    Parameters
    ----------
    coefficients : arr
        the coefficients of one inequality.
    sign : int

    Returns
    -------
    tuple
        Contains the coordinates of the vertices of the valid region.

    """
    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]
    
    if b!=0:
        gradient = -1*a/b
    else:
        gradient = -5.67e5
    
    #coordinates of the plot corners
    top_right = (10, 10)
    bottom_right = (10, -10)
    bottom_left = (-10, -10)
    top_left = (-10, 10)
    
    #coordinates of where the line crosses plot edge
    if b != 0:
        if abs((c-10*a)/b)<=10:
            y_right = (c-10*a)/b
        
        elif abs((c-10*a)/b)>10:
            y_right = np.nan
        
        if abs((c+10*a)/b)<=10:
            y_left = (c+10*a)/b
        
        elif abs((c+10*a)/b)>10:
            y_left = np.nan
    
    elif b == 0:
        y_right = np.nan
        y_left = np.nan
    
    if a != 0:
        if abs((c-10*b)/a)<=10:
            x_top = (c-10*b)/a
        
        elif abs((c-10*b)/a)>10:
            x_top = np.nan
            
        if abs((c+10*b)/a)<=10:
            x_bottom = (c+10*b)/a
        
        elif abs((c+10*b)/a)>10:
            x_bottom = np.nan
            
    elif a == 0:
        x_top = np.nan
        x_bottom = np.nan
        
    top_crossing = (x_top, 10)
    bottom_crossing = (x_bottom, -10)
    right_crossing = (10, y_right)
    left_crossing = (-10, y_left)

    #returning the correct set of coordinates
    if sign == 0:
        if np.isnan(y_right) and np.isnan(y_left): #only top and bottom 
            if gradient > 0:
                return (top_crossing, bottom_crossing, bottom_right, top_right)
            else:
                return (top_crossing, bottom_crossing, bottom_left, top_left) 

        elif np.isnan(y_right) and np.isnan(x_top): #only left and bottom
            return (left_crossing, bottom_crossing, bottom_left)
        
        elif np.isnan(y_right) and np.isnan(x_bottom): #only left and top
            return (left_crossing, top_crossing, top_right, bottom_right, bottom_left)
        
        elif np.isnan(y_left) and np.isnan(x_top): #only right and bottom
            return (right_crossing, bottom_right, bottom_crossing)
        
        elif np.isnan(y_left) and np.isnan(x_bottom): #only right and top
            return (right_crossing, top_crossing, top_left, bottom_left, bottom_right)
        
        elif np.isnan(x_top) and np.isnan(x_bottom): #only left and right
            return (left_crossing, right_crossing, bottom_right, bottom_left)
        
        #only no right-intersect
        elif np.isnan(y_right) and not np.isnan(y_left) and not np.isnan(x_top) and not np.isnan(x_bottom):
            if gradient > 0:
                return (left_crossing, top_crossing, bottom_crossing, top_right, bottom_right) 
            else:
                return (left_crossing, bottom_crossing, bottom_left)
        
        #only no left-intersect
        elif not np.isnan(y_right) and np.isnan(y_left) and not np.isnan(x_top) and not np.isnan(x_bottom): 
            if gradient > 0:
                return (right_crossing, top_crossing, bottom_crossing, bottom_right)
            else:
                return (right_crossing, bottom_crossing, top_crossing, top_left, bottom_left)
        
        #only no top-intersect
        elif not np.isnan(y_right) and not np.isnan(y_left) and np.isnan(x_top) and not np.isnan(x_bottom): 
            if gradient > 0:
                return (left_crossing, bottom_crossing, right_crossing, bottom_right)
            else:
                return (left_crossing, bottom_crossing, right_crossing, bottom_left)
        
        #only no bottom-intersect
        elif not np.isnan(y_right) and not np.isnan(y_left) and not np.isnan(x_top) and np.isnan(x_bottom):
            if gradient > 0:
                return (left_crossing, top_crossing, bottom_right, bottom_left)
            else:
                return (right_crossing, bottom_right, bottom_left, top_crossing)   
        
        else:
            if gradient > 0:
                return (left_crossing, right_crossing, bottom_right)
            
            else:
                return (left_crossing, right_crossing, bottom_left)
        
    elif sign == 1:
        if np.isnan(y_right) and np.isnan(y_left): #only top and bottom 
            if gradient > 0:
                return (top_crossing, bottom_crossing, bottom_left, top_left) 
            else:
                return (top_crossing, bottom_crossing, bottom_right, top_right)

        elif np.isnan(y_right) and np.isnan(x_top): #only left and bottom
            return (left_crossing, bottom_crossing, bottom_right, top_right, top_left)
        
        elif np.isnan(y_right) and np.isnan(x_bottom): #only left and top
            return (left_crossing, top_crossing, top_left)
        
        elif np.isnan(y_left) and np.isnan(x_top): #only right and bottom
            return (right_crossing, bottom_crossing, bottom_left, top_left, top_right)
        
        elif np.isnan(y_left) and np.isnan(x_bottom): #only right and top
            return (right_crossing, top_crossing, top_right)
        
        elif np.isnan(x_top) and np.isnan(x_bottom): #only left and right
            return (left_crossing, right_crossing, top_right, top_left)
        
        #only no right-intersect
        elif np.isnan(y_right) and not np.isnan(y_left) and not np.isnan(x_top) and not np.isnan(x_bottom):
            if gradient > 0:
                return (left_crossing, top_crossing, top_left) 
            else:
                return (left_crossing, bottom_crossing, bottom_right, top_right)
        
        #only no left-intersect
        elif not np.isnan(y_right) and np.isnan(y_left) and not np.isnan(x_top) and not np.isnan(x_bottom): 
            if gradient > 0:
                return (right_crossing, bottom_crossing, bottom_left, top_left)
            else:
                return (right_crossing, top_crossing, top_right)
        
        #only no top-intersect
        elif not np.isnan(y_right) and not np.isnan(y_left) and np.isnan(x_top) and not np.isnan(x_bottom): 
            if gradient > 0:
                return (bottom_crossing, right_crossing, top_right, top_left)
            else:
                return (bottom_crossing, left_crossing, top_left, top_right)
        
        #only no bottom-intersect
        elif not np.isnan(y_right) and not np.isnan(y_left) and not np.isnan(x_top) and np.isnan(x_bottom):
            if gradient > 0:
                return (left_crossing, right_crossing, top_left)
            else:
                return (left_crossing, right_crossing, top_right)
        
        else:
            if gradient > 0:
                return (top_crossing, bottom_crossing, top_left)
            else:
                return (top_crossing, bottom_crossing, top_right)

def unique_coordinates(coordinates):
    """
    Takes in the vertices from valid_region_coordinates() and refines them to 
    only distinct coordinates (some were duplicates).

    Parameters
    ----------
    coordinates : tuple

    Returns
    -------
    refined_coordinates : numpy.ndarray
    """
    coordinates = np.array(coordinates)
    coordinates = [list(i) for i in coordinates]
    _, idx = np.unique(coordinates,axis=0,return_index=True)
    
    df = pd.DataFrame(_, idx, ('x','y'))
    df = df.sort_index(axis=0)
    
    refined_coordinates = df.to_numpy()
    
    return refined_coordinates

def within_bounds(coordinates, input_x, input_y):
    """
    Checks whether some input x,y coordinates are within the region enclosed by
    the valid region vertices in 'coordinates'. The coordinates input into this
    function must be in order around the shape, making a path of the perimeter!

    Parameters
    ----------
    coordinates : np.ndarray
        coordinates of the valid region.
    input_x : int
        x coordinate to check.
    input_y : int
        y coordinate to check.

    Returns
    -------
    bool
        true or false, whether the input coordinates are within the path.

    """
    #add the first coodinate to the end of the list to make a closed region
    coordinates = coordinates + (coordinates[0],)
    coordinates = np.asarray(coordinates)
    
    polygon = mpath.Path(coordinates, closed=True)
    point = (input_x, input_y)
   
    return polygon.contains_point(point)

def transform_coords_to_pygame(coordinates):
    """
    Pygame coordinates start from (0,0) in the top left. Bottom right is (W,H)
    Matplotlib pixels start from (0,0) in the bottom left. Top right is (W,H).
    Converts the x,y coordinates from the graph into pixel coordinates to be
    used on the Pygame screen.

    Parameters
    ----------
    coordinates : np.ndarray
    
    Returns
    -------
    coordinates : np.ndarray
    """
    #ax.bbox.extents = array([ 250.,  250., 1800., 1760.]) from matplotlib fig    
    ax_bbox_extents = [ 250,  250, 1800, 1760]
    x_scale = (ax_bbox_extents[2]-ax_bbox_extents[0]) / (DPI*FIGSIZE[0])
    y_scale = (ax_bbox_extents[3]-ax_bbox_extents[1]) / (DPI*FIGSIZE[0])
    
    x_offset = W * (ax_bbox_extents[0] / (DPI*FIGSIZE[0]))
    y_offset = H * (ax_bbox_extents[1] / (DPI*FIGSIZE[0]))
    
    for i in range(len(coordinates)):
        coordinates[i][0] *= SCREEN_SIZE*x_scale/20
        coordinates[i][1] *= SCREEN_SIZE*y_scale/20
    
    for i in range(len(coordinates)):
        coordinates[i][0] += (SCREEN_SIZE*x_scale/2 + x_offset)  #shift x coordinates
        coordinates[i][1] *=-1
        coordinates[i][1] += (SCREEN_SIZE*y_scale/2 + y_offset - 3)  #shift y coordinates
    
    return coordinates

def valid_region_on_graph(coordinates_a, coordinates_b):
    """
    Checks to see if there is a valid region on the plot.

    Parameters
    ----------
    coordinates_a : ndarray
    coordinates_b : ndarray

    Returns
    -------
    bool
        true or false, is there a valid region on the graph where the two paths
        overlap?.
    """

    coordinates_a = coordinates_a + (coordinates_a[0],)
    coordinates_a = np.asarray(coordinates_a)
    
    coordinates_b = coordinates_b + (coordinates_b[0],)
    coordinates_b = np.asarray(coordinates_b)
    
    polygon_a = mpath.Path(coordinates_a, closed=True)
    polygon_b = mpath.Path(coordinates_b, closed=True)
    
    return mpath.Path.intersects_path(polygon_a, polygon_b)

#%% OBJECTS
class button():
    def __init__(self, surface):
        self.surface = surface
  
    def draw_no_valid_region_button(self, coordinates_rect, mouse_pos):
        """
        Draws the button for "no valid region" which is to be pressable when 
        the valid_region_on_graph() function returns False.

        Parameters
        ----------
        coordinates_rect : tuple
            rectangular coordinates: (x,y,w,h)
        mouse_pos : tuple
            x,y.

        Returns
        -------
        bool

        """
        coordinates_rect = pygame.Rect(coordinates_rect)
        
        font = pygame.font.SysFont('Bauhaus 93', 20)
        text = font.render('No valid region', 1, (0, 0, 0))
        
        #update the colours to RGB-A
        # button_colour_alpha = BUTTON_COL + (50,)
        flash_colour_alpha = BUTTON_FLASH_COL + (50,)
        hover_colour_alpha = BUTTON_HOVER_COL + (50,)
        
        alpha_surface = pygame.Surface(pygame.Rect(coordinates_rect).size, pygame.SRCALPHA)
        
        # If you click when not over this button, return false
        if not coordinates_rect.collidepoint(mouse_pos):
            return False
        
        else:
            pygame.draw.rect(alpha_surface, flash_colour_alpha, alpha_surface.get_rect())
            self.surface.blit(alpha_surface, coordinates_rect)
            self.surface.blit(text, (coordinates_rect[0] + (coordinates_rect[2]/2 - text.get_width()/2),
                                     coordinates_rect[1] + (coordinates_rect[3]/2 - text.get_height()/2)))
            return True
        
    def draw_valid_region_button(self, coordinates_a, coordinates_b, mouse_pos):
        """
        Draws the button for the valid region, which is to be clickable when 
        the valid_region_on_graph() function returns True.

        Parameters
        ----------
        coordinates_a : arr
            coordinates of the first valid region.
        coordinates_b : arr
            coordinates of the second valid region.
        mouse_pos : tuple
            x,y.

        Returns
        -------
        bool
            If True, the mouse was hovering over the correct region. If False,
            mouse was clicked above incorrect region.

        """
        #update the colours to RGB-A
        button_colour_alpha = BUTTON_COL + (50,)
        flash_colour_alpha = BUTTON_FLASH_COL + (50,)
        hover_colour_alpha = BUTTON_HOVER_COL + (50,)
        
        alx, aly = zip(*coordinates_a)
        amin_x, amin_y, amax_x, amax_y = min(alx), min(aly), max(alx), max(aly)
        target_rect_a = pygame.Rect(amin_x, amin_y, amax_x - amin_x, amax_y - amin_y)
        alpha_surface_a = pygame.Surface(target_rect_a.size, pygame.SRCALPHA)
        
        blx, bly = zip(*coordinates_b)
        bmin_x, bmin_y, bmax_x, bmax_y = min(blx), min(bly), max(blx), max(bly)
        target_rect_b = pygame.Rect(bmin_x, bmin_y, bmax_x - bmin_x, bmax_y - bmin_y)
        alpha_surface_b = pygame.Surface(target_rect_b.size, pygame.SRCALPHA)
        
        if not within_bounds(coordinates_a, mouse_pos[0], mouse_pos[1]) or not within_bounds(coordinates_b, mouse_position[0], mouse_position[1]):
            pygame.draw.polygon(alpha_surface_a, button_colour_alpha,
                                [(x - amin_x, y - amin_y) for x, y in coordinates_a])
            self.surface.blit(alpha_surface_a, target_rect_a)
            pygame.draw.polygon(alpha_surface_b, button_colour_alpha,
                                [(x - bmin_x, y - bmin_y) for x, y in coordinates_b])
            self.surface.blit(alpha_surface_b, target_rect_b)
  
            return False
                
        else:
            pygame.draw.polygon(alpha_surface_a, flash_colour_alpha,
                                [(x - amin_x, y - amin_y) for x, y in coordinates_a])
            self.surface.blit(alpha_surface_a, target_rect_a)
            pygame.draw.polygon(alpha_surface_b, flash_colour_alpha,
                                [(x - bmin_x, y - bmin_y) for x, y in coordinates_b])
            self.surface.blit(alpha_surface_b, target_rect_b)
            return True
        
    def draw_answer(self, coordinates_a, coordinates_b):
        hover_colour_alpha = BUTTON_HOVER_COL + (50,)

        alx, aly = zip(*coordinates_a)
        amin_x, amin_y, amax_x, amax_y = min(alx), min(aly), max(alx), max(aly)
        target_rect_a = pygame.Rect(amin_x, amin_y, amax_x - amin_x, amax_y - amin_y)
        alpha_surface_a = pygame.Surface(target_rect_a.size, pygame.SRCALPHA)
        
        blx, bly = zip(*coordinates_b)
        bmin_x, bmin_y, bmax_x, bmax_y = min(blx), min(bly), max(blx), max(bly)
        target_rect_b = pygame.Rect(bmin_x, bmin_y, bmax_x - bmin_x, bmax_y - bmin_y)
        alpha_surface_b = pygame.Surface(target_rect_b.size, pygame.SRCALPHA)
        pygame.draw.polygon(alpha_surface_a, hover_colour_alpha,
                            [(x - amin_x, y - amin_y) for x, y in coordinates_a])
        self.surface.blit(alpha_surface_a, target_rect_a)
        pygame.draw.polygon(alpha_surface_b, hover_colour_alpha,
                            [(x - bmin_x, y - bmin_y) for x, y in coordinates_b])
        self.surface.blit(alpha_surface_b, target_rect_b)
        
        
#%% FUNCTIONS FOR GAME CODE
def graphs():
    """
    Finds the coordinates of the vertices of both valid regions for the pygame 
    screen.

    Returns
    -------
    coordinates_a : tuple
    coordinates_b : tuple

    """
    # Create random inequality
    coefs_a, symbol_a = random_inequality()
    coefs_b, symbol_b = random_inequality()

    # Plot the graph
    xsa, ysa = linear_equation(coefs_a)
    xsb, ysb = linear_equation(coefs_b)
    
    plot_graphs(xsa,ysa, xsb, ysb, coefs_a, coefs_b, symbol_a, symbol_b)
    
    # Find coordinates of vertices that define valid region
    coordinates_a = valid_region_coordinates(coefs_a, symbol_a)
    coordinates_a = unique_coordinates(coordinates_a)
    coordinates_a = transform_coords_to_pygame(coordinates_a)
    coordinates_a = tuple(map(tuple, coordinates_a))
    
    coordinates_b = valid_region_coordinates(coefs_b, symbol_b)
    coordinates_b = unique_coordinates(coordinates_b)
    coordinates_b = transform_coords_to_pygame(coordinates_b)
    coordinates_b = tuple(map(tuple, coordinates_b))
    
    return coordinates_a, coordinates_b

def draw_score(surface, text, x, y):
    """
    Draws the score counter to the screen

    Parameters
    ----------
    surface : pygame.Surface
        screen to blit to.
    text : str
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
    
def draw_no_valid_region_rect(surface, coordinates_rect):
    """
    Draws the image of the button "no valid region" button which should not be
    clickable most of the time.

    Parameters
    ----------
    surface : pygame.Surface
        screen to blit to.
    coordinates_rect : tuple
        contains the coordinates of the button (x,y,w,h).

    Returns
    -------
    None.

    """
    coordinates_rect = pygame.Rect(coordinates_rect)
    
    font = pygame.font.SysFont('Bauhaus 93', 20)
    text = font.render('No valid region', 1, (0, 0, 0))
    
    alpha_surface = pygame.Surface(pygame.Rect(coordinates_rect).size, pygame.SRCALPHA)
    
    pygame.draw.rect(alpha_surface, NO_VALID_REGION_COL, alpha_surface.get_rect())
    surface.blit(alpha_surface, coordinates_rect)
    surface.blit(text, (coordinates_rect[0] + (coordinates_rect[2]/2 - text.get_width()/2),
                             coordinates_rect[1] + (coordinates_rect[3]/2 - text.get_height()/2)))
    
    

#%% GAME
# Initiate pygame
pygame.init() 
# clock = pygame.time.Clock()

# Establish the screen 
pygame.display.set_caption("INEQUALITIES") #title of screen
screen = pygame.display.set_mode((W, H)) #pygame Surface
no_valid_button_rect = (570,735,150,30)
valid_button = button(screen) #initiate button class

# Set start score
score = 0

# Main loop
running = True

while running:
    pygame.time.delay(10)
    complete = False 
    # Plot a random inequality and calculate the vertices of the valid region
    coords_a, coords_b = graphs()
    
    # Check to see if there's a valid region; the "valid polygons" intersect
    is_valid = valid_region_on_graph(coords_a, coords_b)
    graph_image = pygame.image.load('PLOT.png') #loads the graph as an image
    graph_image = pygame.transform.scale(graph_image, (800, 800))

    
    while not complete and running:
        # Set screen and score
        screen.blit(graph_image, (0, 0))
        draw_score(screen, str(score), 5,5)
        # Draw the permanent image of the 'no valid region' sign
        draw_no_valid_region_rect(screen, no_valid_button_rect)
        # valid_button.draw_answer(coords_a, coords_b) #TODO remove this 
        
        # Exit button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               running = False 
               
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos() 
                
                # Draw the valid region buttons including the 'no valid region' button
                if is_valid:
                    if valid_button.draw_valid_region_button(coords_a, coords_b, mouse_position):
                        score += 1
                        complete = True
                    
                    else:
                        score -= 1
                        complete = True
                
                else:
                    if valid_button.draw_no_valid_region_button(no_valid_button_rect, mouse_position):
                        score += 1
                        complete = True
                    
                    else:
                        score -= 1
                        complete = True
        

              
        pygame.display.update()

pygame.quit()    
