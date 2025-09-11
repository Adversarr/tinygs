#!/usr/bin/env python3
"""
Point cloud visualization script using PyVTK.
This script reads point cloud data from a text file and visualizes it in 3D.
"""

import vtk
import numpy as np
import argparse
import sys
import os

def read_point_cloud(filename):
    """
    Read point cloud data from a text file using numpy.
    
    Args:
        filename (str): Path to the input file
        
    Returns:
        numpy.ndarray: Array of shape (N, 3) containing x, y, z coordinates
    """
    try:
        # Load data using numpy
        points = np.loadtxt(filename)
        
        # Ensure we have the right shape (N, 3)
        if points.ndim == 1:
            # If only one point, reshape to (1, 3)
            points = points.reshape(1, -1)
        
        # Check that we have 3 coordinates per point
        if points.shape[1] != 3:
            print(f"Error: Expected 3 coordinates per point, got {points.shape[1]}")
            sys.exit(1)
            
        return points
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data format in {filename}.")
        print(f"Details: {e}")
        sys.exit(1)

def create_vtk_point_cloud(points):
    """
    Create a VTK point cloud from a numpy array of points.
    
    Args:
        points (numpy.ndarray): Array of shape (N, 3) containing x, y, z coordinates
        
    Returns:
        vtkPolyData: VTK polydata containing the points
    """
    # Create points
    vtk_points = vtk.vtkPoints()
    
    # Add all points at once
    vtk_points.SetNumberOfPoints(len(points))
    for i, (x, y, z) in enumerate(points):
        vtk_points.SetPoint(i, x, y, z)
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    
    # Create vertices
    vertices = vtk.vtkCellArray()
    for i in range(len(points)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    
    polydata.SetVerts(vertices)
    
    return polydata

def visualize_point_cloud(polydata):
    """
    Visualize the point cloud using VTK.
    
    Args:
        polydata (vtkPolyData): VTK polydata containing the points
    """
    # Create mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(2)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    render_window.SetWindowName("Point Cloud Visualization")
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Set up interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    
    # Start visualization
    render_window.Render()
    interactor.Start()

def main():
    """
    Main function to parse arguments, read point cloud data and visualize it.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize 3D point cloud data using PyVTK")
    parser.add_argument("filename", nargs="?", default="means.txt",
                        help="Path to the point cloud data file (default: means.txt)")
    
    # Parse arguments
    args = parser.parse_args()
    filename = args.filename
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        sys.exit(1)
    
    # Read point cloud data
    print(f"Reading point cloud data from {filename}...")
    points = read_point_cloud(filename)
    print(f"Loaded {len(points)} points.")
    
    # Create VTK point cloud
    print("Creating VTK point cloud...")
    polydata = create_vtk_point_cloud(points)
    
    # Visualize point cloud
    print("Visualizing point cloud...")
    visualize_point_cloud(polydata)

if __name__ == "__main__":
    main()