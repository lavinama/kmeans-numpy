import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import random
import time

global show_plots
show_plots = False

def generate_data():
    # Size of dataset to be generated. The final size is 4 * data_size
    data_size = 1000
    num_iters = 50
    num_clusters = 4

    # sample from Gaussians 
    data1 = np.random.normal((5, 5, 5), (4, 4, 4), (data_size,3))
    data2 = np.random.normal((4, 20, 20), (3,3,3), (data_size, 3))
    data3 = np.random.normal((25, 20, 5), (5, 5, 5), (data_size,3))
    data4 = np.random.normal((30, 30, 30), (5, 5, 5), (data_size,3))

    # Combine the data to create the final dataset
    data = np.concatenate((data1,data2, data3, data4), axis = 0)

    # Shuffle the data
    np.random.shuffle(data)
    
    if show_plots:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.scatter(data[:,0], data[:,1], data[:,2], s= 0.5)
        plt.show()
    return data, num_iters, num_clusters

def init_centroids(data):
    # Set random seed for reproducibility 
    np.random.seed(0)
    # Initialise centroids
    centroids = data[random.sample(range(data.shape[0]), 4)]
    # Create a list to store which centroid is assigned to each dataset
    assigned_centroids = np.zeros(len(data), dtype = np.int32)
    return centroids, assigned_centroids

def compute_l2_distance(x, centroid):
    # Compute the difference, following by raising to power 2 and summing
    dist = ((x - centroid) ** 2).sum(axis = x.ndim - 1)
    return dist

def get_closest_centroid(x, centroids):
    # Loop over each centroid and compute the distance from data point.
    dist = compute_l2_distance(x, centroids)
    # Get the index of the centroid with the smallest distance to the data point 
    closest_centroid_index =  np.argmin(dist, axis=1)
    return closest_centroid_index  

def compute_sse(data, centroids, assigned_centroids):
    # Initialise SSE 
    sse = 0
    # Compute SSE    
    sse = compute_l2_distance(data, centroids[assigned_centroids]).sum() / len(data)
    return sse

def visualise_clusters(data, centroids, assigned_centroids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for c in range(len(centroids)):
            cluster_members = [data[i] for i in range(len(data)) if assigned_centroids[i] == c]    
            cluster_members = np.array(cluster_members)
            
            ax.scatter(cluster_members[:,0], cluster_members[:,1], cluster_members[:,2], s= 0.5)
    
def main():
    data, num_iters, num_clusters = generate_data()
    centroids, assigned_centroids = init_centroids(data)

    # Number of dimensions in centroid
    num_centroid_dims = data.shape[1]

    # List to store SSE for each iteration 
    sse_list = []

    # Loop over iterations
    for n in range(num_iters):
        # Get closest centroids to each data point
        assigned_centroids = get_closest_centroid(data[:, None, :], centroids[None,:, :])
        # Compute new centroids
        for c in range(centroids.shape[1]):
            # Get data points belonging to each cluster 
            cluster_members = data[assigned_centroids == c]
            # Compute the mean of the clusters
            cluster_members = cluster_members.mean(axis = 0)
            # Update the centroids
            centroids[c] = cluster_members
        # Compute the SSE for the iteration
        sse = compute_sse(data, centroids, assigned_centroids)
        sse_list.append(sse)

    if show_plots:
        visualise_clusters(data, centroids, assigned_centroids)
        plt.figure()
        plt.xlabel("Iterations")
        plt.ylabel("SSE")
        plt.plot(range(len(sse_list)), sse_list)
        plt.show()


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print("Time Elapsed Per Loop Vectorised {:.3f}".format((tic - toc)/ 50))
