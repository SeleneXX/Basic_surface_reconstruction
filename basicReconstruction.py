import argparse
import numpy as np
import plotly
import plotly.figure_factory as ff
from skimage import measure
from knnsearch import knnsearch
import math


def createGrid(points, resolution=64):
    """
    constructs a 3D grid containing the point cloud
    each grid point will store the implicit function value
    Args:
        points: 3D points of the point cloud
        resolution: grid resolution i.e., grid will be NxNxN where N=resolution
                    set N=16 for quick debugging, use *N=64* for reporting results
    Returns: 
        X,Y,Z coordinates of grid vertices     
        max and min dimensions of the bounding box of the point cloud                 
    """
    max_dimensions = np.max(points,axis=0) # largest x, largest y, largest z coordinates among all surface points
    min_dimensions = np.min(points,axis=0) # smallest x, smallest y, smallest z coordinates among all surface points    
    bounding_box_dimensions = max_dimensions - min_dimensions # com6pute the bounding box dimensions of the point cloud
    max_dimensions = max_dimensions + bounding_box_dimensions/10  # extend bounding box to fit surface (if it slightly extends beyond the point cloud)
    min_dimensions = min_dimensions - bounding_box_dimensions/10
    X, Y, Z = np.meshgrid( np.linspace(min_dimensions[0], max_dimensions[0], resolution),
                           np.linspace(min_dimensions[1], max_dimensions[1], resolution),
                           np.linspace(min_dimensions[2], max_dimensions[2], resolution) )    
    
    return X, Y, Z, max_dimensions, min_dimensions

def sphere(center, R, X, Y, Z):
    """
    constructs an implicit function of a sphere sampled at grid coordinates X,Y,Z
    Args:
        center: 3D location of the sphere center
        R     : radius of the sphere
        X,Y,Z coordinates of grid vertices                      
    Returns: 
        IF    : implicit function of the sphere sampled at the grid points
    """    
    IF = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2 - R ** 2 
    return IF

def showMeshReconstruction(IF):
    """
    calls marching cubes on the input implicit function sampled in the 3D grid
    and shows the reconstruction mesh
    Args:
        IF    : implicit function sampled at the grid points
    """    
    verts, simplices, normals, values = measure.marching_cubes(IF, 0)
    x, y, z = zip(*verts)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=simplices,
                            title="Isosurface")
    plotly.offline.plot(fig)

def mlsReconstruction(points, normals, X, Y, Z):
    """
    surface reconstruction with an implicit function f(x,y,z) representing
    MLS distance to the tangent plane of the input surface points 
    The method shows reconstructed mesh
    Args:
        input: filename of a point cloud    
    Returns:
        IF    : implicit function sampled at the grid points
    """

    # idx stores the index to the nearest surface point for each grid point in Q.
    # we use provided knnsearch function        
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    R = points
    K = 20
    idx, Di = knnsearch(Q, R, K)

    ################################################
    # <================START CODE<================>
    ################################################
     
    # replace this random implicit function with your MLS implementation!
    _, D = knnsearch(R, R, 1)
    B2 = np.average(D) * 4
    print(math.sqrt(B2))
    IF = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    idx = idx.reshape((X.shape[0], X.shape[1], X.shape[2], K))
    Di = Di.reshape((X.shape[0], X.shape[1], X.shape[2], K))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                arraypi = np.array([points[c, :] for c in idx[i,j,k,:]])
                arrayp = np.array([[X[i,j,k], Y[i,j,k], Z[i,j,k]] for _ in range(K)])
                arrayni = np.array([normals[c, :] for c in idx[i,j,k,:]]).transpose()
                diff = arrayp - arraypi
                arraydp = diff.dot(arrayni).diagonal()
                distance = Di[i, j, k, :]
                phi = np.exp(- distance / B2)
                fp = phi.dot(arraydp)/np.sum(phi)
                IF[i, j, k] = fp
    IF = IF.transpose()
    ################################################
    # <================END CODE<================>
    ################################################

    return IF 


def naiveReconstruction(points, normals, X, Y, Z):
    """
    surface reconstruction with an implicit function f(x,y,z) representing
    signed distance to the tangent plane of the surface point nearest to each 
    point (x,y,z)
    Args:
        input: filename of a point cloud    
    Returns:
        IF    : implicit function sampled at the grid points
    """

    # idx stores the index to the nearest surface point for each grid point.
    # we use provided knnsearch function
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    print(Q.shape)
    R = points
    K = 1
    idx, _ = knnsearch(Q, R, K)


    ################################################
    # <================START CODE<================>
    ################################################

    # replace this random implicit function with your naive surface reconstruction implementation!
    IF = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    idx = idx.reshape((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                pj = points[idx[i,j,k], :]
                p = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                nj = normals[idx[i,j,k], :]
                fp = nj.dot(p - pj)
                IF[i,  j, k] = fp
    IF = IF.transpose()
    ################################################
    # <================END CODE<================>
    ################################################

    return IF 




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic surface reconstruction')
    parser.add_argument('--file', type=str, default = "sphere.pts", help='input point cloud filename')
    parser.add_argument('--method', type=str, default = "mls",\
                        help='method to use: mls (Moving Least Squares), naive (naive reconstruction), sphere (just shows a sphere)')
    args = parser.parse_args()

    #load the point cloud
    data = np.loadtxt(args.file)
    points = data[:,:3]
    normals = data[:,3:]

    # create grid whose vertices will be used to sample the implicit function
    X,Y,Z,max_dimensions,min_dimensions = createGrid(points, 64)

    if args.method == 'mls':
        print(f'Running Moving Least Squares reconstruction on {args.file}')
        IF = mlsReconstruction(points, normals, X, Y, Z)
    elif args.method == 'naive':
        print(f'Running naive reconstruction on {args.file}')
        IF = naiveReconstruction(points, normals, X, Y, Z)
    else:
        # toy implicit function of a sphere - replace this code with the correct
        # implicit function based on your input point cloud!!!
        print(f'Replacing point cloud {args.file} with a sphere!')
        IF = mlsReconstruction(points, normals, X, Y, Z)

    showMeshReconstruction(IF)