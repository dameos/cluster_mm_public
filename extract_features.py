from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


# String path to image
image_path = sys.argv[1]
# Number of clusters Only for testing

# Load image 
image = cv2.imread(image_path)

r = 300 / image.shape[1]
dim = (300, int(image.shape[0] * r))
 
# perform the actual resizing of the image and show it
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


def convert_image_to_LAB(image):
    # Change image color space
    imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    imageLAB_shape = imageLAB.shape
    # Empty Zeros 
    zeros = np.zeros((imageLAB_shape[0]*imageLAB_shape[1], 2))
    index_zeros = 0
    for i in range(imageLAB_shape[0]):
        for j in range(imageLAB_shape[1]):
            zeros[index_zeros] = np.delete(imageLAB[i][j],0)
            index_zeros = index_zeros + 1
    return zeros
    

def paint_image(k_means, centers, image):
    # Convert to LAB
    imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Get shape 
    imageLAB_shape = imageLAB.shape
    for i in range(imageLAB_shape[0]):
        for j in range(imageLAB_shape[1]):
            features = np.delete(imageLAB[i][j],0)
            # Use the color of centroid
            prediction = k_means.predict([features])
            # Add illumination to pic 
            true_colors = centers[prediction[0]]
            imageLAB[i][j][0] = 128.
            imageLAB[i][j][1] = true_colors[0]
            imageLAB[i][j][2] = true_colors[1]
    # Convert image to RGB
    imageRGB = cv2.cvtColor(imageLAB, cv2.COLOR_LAB2BGR)
    cv2.imwrite('pene.png',imageRGB)
    cv2.imshow("Image clustered", imageRGB)
    cv2.waitKey(0)

def get_best_k_calinski(image_AB, show_chart):
    distortions = []
    K = range(2,7)
    max1 = 10
    k = 0
    for k1 in K:
        k_mean_model = KMeans(n_clusters=k1, random_state=1).fit(image_AB)
        labels = k_mean_model.labels_
        value = sum(np.min(cdist(image_AB, k_mean_model.cluster_centers_, 'euclidean'), axis=1)) / image_AB.shape[0]
        if value < max1:
            max1 = value
            k = k1
        distortions.append(value)
    if show_chart == True:
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Calinski Harabaz score mehtod showing optimal k')
        plt.show()
    return k

# Convert image to lab and remove L
image_AB = convert_image_to_LAB(image)
k = get_best_k_calinski(image_AB, True)
print(k)
k_means = KMeans(n_clusters=k, random_state=1).fit(image_AB)
centers = k_means.cluster_centers_
paint_image(k_means, centers, image)
