#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:30:03 2024

@author: marleymoore

There are 5 major circle theorems as part of the GCSE syllabus:
    - Angle in a semicircle is 90º
    - Angle at centre is twice the angle at circumference
    - Opposite angles in a cyclic quadrilateral sum to 180º
    - Angles in the same segment are equal
    - Alternate segment theorem
    
in conjunction with standard geometric rules:
    - Base angles of an isosceles are equal
    - Alternate angles are equal
    - Angles on a straight line sum to 180º

"""
#%% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

#%% FUNCTIONS
def invalid_theta_about_point(theta, width):
    """
    Given a theta and a width either side of this angle that is 'invalid' to 
    another point, this function returns a Pandas dataframe containing the
    ranges of invalid theta.

    Parameters
    ----------
    theta : int
        angle of point in degrees clockwise from 12 O'clock.
    width : int
        width of invalid region in degrees (minimum separation of points).

    Returns
    -------
    invalid_DF : pd.DataFrame
        contains ranges of invalid theta coordinates.

    """
    # Establish dataframe 
    invalid_DF = pd.DataFrame(columns = ['lower', 'upper'])
    
    # Input bounds
    if (theta - width) < 0:
        invalid_DF.loc[0,'lower'] = (theta - width)%360
        invalid_DF.loc[0,'upper'] = 360
        invalid_DF.loc[1,'lower'] = 0
        invalid_DF.loc[1,'upper'] = (theta + width)
    
    elif (theta + width) > 360:
        invalid_DF.loc[0,'lower'] = (theta - width)
        invalid_DF.loc[0,'upper'] = 360
        invalid_DF.loc[1,'lower'] = 0
        invalid_DF.loc[1,'upper'] = (theta + width)%360
    
    else:
        invalid_DF.loc[0,'lower'] = (theta - width)
        invalid_DF.loc[0,'upper'] = (theta + width)
        
    return invalid_DF

def valid_angles(invalid_DF):
    """
    Provided a dataframe containing all the invalid ranges for theta, this 
    function returns a list of all remaining valid theta.

    Parameters
    ----------
    invalid_DF : pd.DataFrame
        contains ranges of invalid theta coordinates.

    Returns
    -------
    valid : list
        list of valid theta.

    """
    # Pull all the invalid ranges out of dataframe
    ranges = []
    for i in range(len(invalid_DF)):
        lower = invalid_DF.loc[i][0]
        upper = invalid_DF.loc[i][1]
        rangei = np.arange(lower, upper+1)
        ranges.append(rangei)
    
    # Delete duplicate theta
    invalid = np.unique(np.concatenate(ranges).ravel())
    
    # Find which degrees are not within the invalid list
    valid = list(set(np.arange(361)).difference(set(invalid)))

    return valid

def bearing(A,B):
    """
    Calculates the bearing in degrees from point A to point B (i.e. bearing of
    B from A), where A and B are given as vectoral cartesian coordinates.
 
    Parameters
    ----------
    A : np.array
        coordinates of point A.
    B : np.array
        coordinates of point B.

    Returns
    -------
    bearing : float
        bearing of B from A in degrees

    """
    # Ensure the points are arrays
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Calculate vector AB
    AB = B-A
    ABx = AB[0]
    ABy = AB[1]
    
    # North, East, South, West
    if ABx == 0 and ABy > 0:
        theta = 0
    
    elif ABx == 0 and ABy < 0:
        theta = 180
    
    elif ABy == 0 and ABx > 0:
        theta = 90
    
    elif ABy == 0 and ABx < 0:
        theta = 270
    
    elif ABx == 0 and ABy == 0:
        raise ValueError('The two points used are the same')
    
    else:
        if ABx > 0 and ABy > 0: #north east
            theta = 90 - np.rad2deg(np.arctan(ABy/ABx))
            
        elif ABx > 0 and ABy < 0: #south east
            theta = 90 + np.rad2deg(np.arctan(ABy/ABx))
            
        elif ABx < 0 and ABy < 0: #south west
            theta = 360 - (90 + np.rad2deg(np.arctan(ABy/ABx)))
        
        elif ABx < 0 and ABy > 0: #north west
            theta = 360 - (90 - np.rad2deg(np.arctan(ABy/ABx)))
        
    return theta
            
#%% OBJECTS
class circle():
    """
    Circle object for plotting a circle containing a circle theorems problem
    to a figure to be loaded to a pygame window. It should be initialised with
    the variables:
        coordinates of centre as (x,y) integers
        radius of circle as a float
        colour to plot the circle
        
    """
    def __init__(self, x, y, R):
        self.x = x #centre
        self.y = y #centre
        self.R = R #radius
        self.D = 2*R #diameter
    
    def polar_to_cartesian(self, theta):
        """
        Converts a given polar coordinate on the circle circumference into 
        Cartesian coordinates.

        Parameters
        ----------
        theta : int
            angle in degrees clockwise starting at 12 O'clock

        Returns
        -------
        x1 : float
            x-coordinate of point
        y1 : float
            y-coordinate of point

        """
        x1 = self.x + self.R * np.cos(np.deg2rad(theta))
        y1 = self.y + self.R * np.sin(np.deg2rad(theta))
        
        return x1, y1
    
    def tangent_at_point(self, theta):
        """
        Imagine a line BAC that runs tangent to the circle at point A. This 
        function returns the coordinates of endpoints B and C.

        Parameters
        ----------
        theta : int
            angle in degrees clockwise starting at 12 O'clock

        Returns
        -------
        A : numpy.array
            coordinates of endpoint A
        B : numpy.array
            coordinates of endpoint B
            
        """
        x_pos, y_pos = self.polar_to_cartesian(theta)
        m = - (x_pos - self.x) / (y_pos - self.y)
        # c = y_pos - m * x_pos
        # import pdb
        # pdb.set_trace()
        A = np.array([x_pos, y_pos])
        vector = np.array([1, m])
        unit_vector = vector / np.linalg.norm(vector)
        
        AB = (self.R / 2) * unit_vector
        B = A + AB
        
        AC = (self.R / 2) * -unit_vector
        C = A + AC
    
        return B, C
        
    # Circle Theorems
    
    def angle_in_semicircle(self, theta_start):
        """
        Given a start position, this function gives the coordinates of the
        vertices of a right-angled triangle defined by the "angle in a
        semicircle is 90 degrees" circle theorem. These vertices are then to be
        used by matplotlib.patches.polygon to add a patch of the shape onto the
        plot. 
        
        This function also ensures that vertices are never too close together
        by asserting invalid regions about existing points.
        
        Parameters
        ----------
        theta_start : int
            theta seed
        
        Returns
        -------
        (N,2) array
            array containing vertices for patch
            
        """    
        theta_start = theta_start%360 #incase larger angle is given
        theta_end = (theta_start + 180)%360
        
        # Define valid region
        w = 30 # width of invalid region in degrees
        
        # Find the invalid theta about each existing point
        invalid_DF_start = invalid_theta_about_point(theta_start, w)
        invalid_DF_end = invalid_theta_about_point(theta_end, w)
        
        invalid_DF = pd.concat([invalid_DF_start, invalid_DF_end],
                               ignore_index=True)
        
        valid_theta = valid_angles(invalid_DF)
        
        # Pick a third coordinate
        theta3 = random.sample(valid_theta,1)[0]
        
        # Convert all angles to Cartesian coordinates
        x1, y1 = self.polar_to_cartesian(theta_start)
        x2, y2 = self.polar_to_cartesian(theta_end)
        x3, y3 = self.polar_to_cartesian(theta3)
        
        diameter_start = (x1, y1)
        diameter_end = (x2, y2)
        
        coordinates = (diameter_start, diameter_end, (x3,y3))
        coordinates = np.asarray(coordinates)
        
        # Find the coordinates of a square that represents the right angle
        A = np.array([x1,y1]) #diam start
        B = np.array([x2,y2]) #diam end
        C = np.array([x3,y3]) #right angle point
        
        CA = A-C
        CB = B-C
        
        CD = (CA/np.linalg.norm(CA))*(self.R)/6
        CF = np.linalg.norm(CD)*(CB / np.linalg.norm(CB))
        
        D = C + CD #CDEF defines right-angle box
        F = C + CF
        
        E = C + CF + CD
        
        right_angle_coords = (C,D,E,F)
        right_angle_coords = np.asarray(right_angle_coords)
        
        return coordinates, right_angle_coords
    
    def isosceles(self, theta_start):
        """
        Returns the vertices of an isosceles triangle to be plotted as a plt patch.
        Again, the vertices must not be too similar in position so that it is
        obviously a triangle. This function must produce two randomly generated
        positions that are not close to each other on the x-axis, and do not lie
        on a diameter-like line, so that it is obviously an isosceles. 
        
        Parameters
        ----------
        theta_start : int
            theta seed
        
        Returns
        -------
        (N,2) array
            array containing vertices for patch
    
        """
        theta_start = theta_start%360 #incase larger angle is given
        theta_end = (theta_start + 180)%360
        
        # Define invalid region
        w = 25 # width of invalid region in degrees
        
        # Find the invalid theta about each existing point
        invalid_DF_start = invalid_theta_about_point(theta_start, w)
        invalid_DF_end = invalid_theta_about_point(theta_end, w)

        invalid_DF = pd.concat([invalid_DF_start, invalid_DF_end],
                               ignore_index=True)
        
        valid_theta = valid_angles(invalid_DF)
        
        # Pick a second coordinate
        theta2 = random.sample(valid_theta,1)[0]
        
        # Convert all angles to Cartesian coordinates
        x1, y1 = self.polar_to_cartesian(theta_start)
        x2, y2 = self.polar_to_cartesian(theta2)
        
        coordinates = ((x1,y1), (x2,y2), (self.x,self.y))
        coordinates = np.asarray(coordinates)
        
        # Dataframe for angle patches
        angle_DF = pd.DataFrame(columns = ['coordinates', 'bearing_1', 'bearing_2'],
                                index = (['A', 'B']))
        
        # Wedges
        A = np.array([x1,y1])
        B = np.array([self.x,self.y])
        C = np.array([x2,y2])
        
        # Wedge at A
        theta_B_from_A = bearing(A,B)
        theta_C_from_A = bearing(A,C)
        
        # Wedge at B
        theta_A_from_B = bearing(B,A)
        theta_C_from_B = bearing(B,C)

        angle_DF.loc['A','coordinates'] = A
        angle_DF.loc['A','bearing_1'] = theta_B_from_A #first patch
        angle_DF.loc['A','bearing_2'] = theta_C_from_A
        
        angle_DF.loc['B','coordinates'] = B
        angle_DF.loc['B','bearing_1'] = theta_A_from_B #second patch
        angle_DF.loc['B','bearing_2'] = theta_C_from_B
        
        return coordinates, angle_DF
    
    def angle_at_centre(self, theta_start):
        """
        Takes in the start positions and generates random coordinates that 
        define the angle at centre is twice the angle at the circumference
        circle theorem. Invalid region width must be less than 45.
        
        Parameters
        ----------
        theta_start : int
            theta seed
        
        Returns
        -------
        (N,2) array
            array containing vertices for patch
    
        """
        # Define opposite point
        theta_opposite = (theta_start + 180)%360
        
        # Define invalid region about each point
        w = 40 #invalid region width
        
        # Find the invalid theta about each existing point
        invalid_DF_start = invalid_theta_about_point(theta_start, w)
        invalid_DF_opposite = invalid_theta_about_point(theta_opposite, w)
        
        invalid_DF = pd.concat([invalid_DF_start, invalid_DF_opposite],
                               ignore_index=True)
    
        valid_theta = valid_angles(invalid_DF)
        
        # Pick a second coordinate
        theta2 = random.sample(valid_theta,1)[0]
        
        # Update the invalid region
        theta_opposite_2 = (theta2 + 180)%360
        
        invalid_DF_2 = invalid_theta_about_point(theta2, w)
        invalid_DF_opposite_2 = invalid_theta_about_point(theta_opposite_2, w)
        
        invalid_DF_new = pd.concat([invalid_DF, invalid_DF_2, invalid_DF_opposite_2],
                                   ignore_index=True)
        
        valid_theta_new = valid_angles(invalid_DF_new)
        
        # Pick a third coordinate
        theta3 = random.sample(valid_theta_new,1)[0]
    
        # Convert all angles to Cartesian coordinates
        x_start, y_start = self.polar_to_cartesian(theta_start)
        x2, y2           = self.polar_to_cartesian(theta2)
        x3, y3           = self.polar_to_cartesian(theta3)
        
        coordinates = ((x_start, y_start), (x2,y2), (self.x,self.y), (x3,y3))
        coordinates = np.asarray(coordinates)
        
        return coordinates
        
    def cyclic_quadrilateral(self, theta_start):
        """
        Given a start coordinate, three other coordinates are generated and 
        returned as an array, ensuring that no two points are close together.

        Parameters
        ----------
        theta_start : int
            theta seed
        
        Returns
        -------
        (N,2) array
            array containing vertices for patch
    
        """
        # Define invalid region about each point
        w = 44 #invalid region width
        
        # Find the valid theta for second point
        invalid_DF_start = invalid_theta_about_point(theta_start, w)
        valid_theta = valid_angles(invalid_DF_start)
        
        # Pick a second coordinate
        theta2 = random.sample(valid_theta,1)[0]
        
        # Update the valid region
        invalid_DF_2 = invalid_theta_about_point(theta2, w)
        invalid_DF = pd.concat([invalid_DF_start, invalid_DF_2], ignore_index=True)
        valid_theta = valid_angles(invalid_DF)
        
        # Pick a third coordinate
        theta3 = random.sample(valid_theta,1)[0]
        
        # Update the valid region
        invalid_DF_3 = invalid_theta_about_point(theta3, w)
        invalid_DF = pd.concat([invalid_DF, invalid_DF_3], ignore_index=True)
        valid_theta = valid_angles(invalid_DF)
        
        # Pick a fourth coordinate
        theta4 = random.sample(valid_theta,1)[0]
        
        # Order the theta in increasing value
        thetas = np.sort([theta_start, theta2, theta3, theta4])
        
        # Convert all angles to Cartesian coordinates
        x1, y1 = self.polar_to_cartesian(thetas[0])
        x2, y2 = self.polar_to_cartesian(thetas[1])
        x3, y3 = self.polar_to_cartesian(thetas[2])
        x4, y4 = self.polar_to_cartesian(thetas[3])
        
        coordinates = ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
        coordinates = np.asarray(coordinates)
        
        return coordinates
        
    def same_segment_theorem(self, theta_start):
        """
        Given a starting coordinate, this function returns the coordinates of
        four vertices that define the same segment theorem.

        Parameters
        ----------
        theta_start : int
            theta seed
        
        Returns
        -------
        (N,2) array
            array containing vertices for patch

        """
        # Define invalid region about each point
        w = 44 #invalid region width
        
        # Find the valid theta for second point
        invalid_DF_start = invalid_theta_about_point(theta_start, w)
        valid_theta = valid_angles(invalid_DF_start)
        
        # Pick a second coordinate
        theta2 = random.sample(valid_theta,1)[0]
        
        # Update the valid region
        invalid_DF_2 = invalid_theta_about_point(theta2, w)
        invalid_DF = pd.concat([invalid_DF_start, invalid_DF_2], ignore_index=True)
        valid_theta = valid_angles(invalid_DF)
        
        # Pick a third coordinate
        theta3 = random.sample(valid_theta,1)[0]
        
        # Update the valid region
        invalid_DF_3 = invalid_theta_about_point(theta3, w)
        invalid_DF = pd.concat([invalid_DF, invalid_DF_3], ignore_index=True)
        valid_theta = valid_angles(invalid_DF)
        
        # Pick a fourth coordinate
        theta4 = random.sample(valid_theta,1)[0]
        
        # Order the theta in increasing value
        thetas = np.sort([theta_start, theta2, theta3, theta4])
        
        # Convert all angles to Cartesian coordinates
        x1, y1 = self.polar_to_cartesian(thetas[0])
        x2, y2 = self.polar_to_cartesian(thetas[1])
        x3, y3 = self.polar_to_cartesian(thetas[2])
        x4, y4 = self.polar_to_cartesian(thetas[3])
        
        coordinates = ((x1,y1), (x3,y3), (x2,y2), (x4,y4))
        coordinates = np.asarray(coordinates)
        
        return coordinates
    
    def alternate_segment_theorem(self, theta_start):
        """
        Given a starting coordinate, this function returns the coordinates of
        four vertices that define the alternate segment theorem.

        Parameters
        ----------
        theta_start : int
            theta seed
        
        Returns
        -------
        (N,2) array
            array containing vertices for patch

        """
        # Define invalid region about each point
        w = 85 #invalid region width
        
        # Find the valid theta for second point
        invalid_DF = invalid_theta_about_point(theta_start, w)
        valid_theta = valid_angles(invalid_DF)
        
        # Pick a second coordinate
        theta2 = random.sample(valid_theta,1)[0]
        
        # Update the valid region
        invalid_DF_2 = invalid_theta_about_point(theta2, w)
        invalid_DF = pd.concat([invalid_DF, invalid_DF_2], ignore_index=True)
        valid_theta = valid_angles(invalid_DF)
        
        # Pick a third coordinate
        theta3 = random.sample(valid_theta,1)[0]
        
        # Convert all angles to Cartesian coordinates
        x1, y1 = self.polar_to_cartesian(theta_start)
        x2, y2 = self.polar_to_cartesian(theta2)
        x3, y3 = self.polar_to_cartesian(theta3)
        
        coordinates = ((x1,y1), (x2,y2), (x3,y3))
        coordinates = np.asarray(coordinates)
        
        return coordinates
        
#%% PLOT CHECK

def plot():
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ax.set_xlim(xmin=0, xmax=30)
    ax.set_ylim(ymin=0, ymax=30)
    
    circ1 = circle(15,15,10) #establish circle with size and position
    circ_patch = mpatches.Circle((circ1.x,circ1.y),circ1.R, color='k', fill=False)
    ax.add_patch(circ_patch)   
    
    seed_theta = random.randint(0,360) #choose a random seed
    
    # semi_circ_coords, right_angled_coords = circ1.angle_in_semicircle(seed_theta)
    # semi = mpatches.Polygon(semi_circ_coords, color='b', fill=False)
    # right_angle = mpatches.Polygon(right_angled_coords, facecolor=(0.7,0.7,0.7),
    #                                edgecolor=(0.5,0,0), fill=True)
    # ax.add_patch(semi)
    # ax.add_patch(right_angle)
    
    isos_coords, angle_DF = circ1.isosceles(seed_theta)
    # Plot the circle theorem patch
    isos = mpatches.Polygon(isos_coords, color='r', fill=False) 
    ax.add_patch(isos)
    # Plot the angle patches
    for i in range(len(angle_DF)):
        angle_patch = mpatches.Wedge((angle_DF.loc['A']['coordinates'][0],
                                      angle_DF.loc['A']['coordinates'][1]), circ1.R/8,
                                     angle_DF.loc[i][0], angle_DF.loc[i][1])
        ax.add_patch(angle_patch)
        

    # angle_1 = mpatches.Polygon(angle_1_coords, facecolor=(0.7,0.7,0.7),
    #                            edgecolor=(0.5,0,0), fill=True)
    # angle_2 = mpatches.Polygon(angle_2_coords, facecolor=(0.7,0.7,0.7),
    #                            edgecolor=(0.5,0,0), fill=True)

    
    # at_centre_coords = circ1.angle_at_centre(seed_theta)
    # arrow = mpatches.Polygon(at_centre_coords, color='b', fill=False)
    # ax.add_patch(arrow)
    
    # cyc_quad_coords = circ1.cyclic_quadrilateral(seed_theta)
    # quad = mpatches.Polygon(cyc_quad_coords, color='r', fill=False)
    # ax.add_patch(quad)
    
    # same_seg_coords = circ1.same_segment_theorem(seed_theta)
    # same = mpatches.Polygon(same_seg_coords, color='r', fill=False)
    # ax.add_patch(same)
    
    # alt_seg_coords = circ1.alternate_segment_theorem(seed_theta)
    # alt = mpatches.Polygon(alt_seg_coords, color='r', fill=False)
    # ax.add_patch(alt)
    # b, c = circ1.tangent_at_point(seed_theta)
    # ax.plot((b[0],c[0]),(b[1],c[1]), ls='-')
    # ax.axline((0,c), color='k', slope=m)
    
    plt.show()

plot()
