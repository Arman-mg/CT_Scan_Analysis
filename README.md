# **CT-Scan Analysis for Lung Masking and Infection Identification**

## **Introduction**
This repository contains Python code that uses machine learning and image processing techniques to perform various operations on CT scan images. The operations include loading and visualizing the CT scan images, creating a histogram of the scan, and using K-Means for clustering to identify different regions of the image.

The primary function of this code is to identify the lungs in a CT scan and further detect Ground Glass Opacities (GGO), a common feature in lung infections such as COVID-19. This is accomplished by applying K-Means clustering and DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

## **Dependencies**
The script requires the following Python libraries:
* NumPy
* NiBabel
* Matplotlib
* scikit-learn
* itertools

You can install them using pip:

pip install numpy nibabel matplotlib scikit-learn

## **Usage**

The script operates on a single slice of a CT scan, specified by the index. It reads .nii files (Neuroimaging Informatics Technology Initiative) which is a standard format for storing medical imaging data.
Functions

    read_nii(filepath): Reads the CT scan from the file and stores it in an array.
    hist(): Plots a histogram of CT scan values.
    Kmeans(Ncluster, ifind): Applies K-means clustering on the CT scan and displays the original and quantized images.
    find_lungs(eps, min_samples): Applies DBSCAN clustering on the CT scan and displays the identified left and right lung masks.
    final_lung_masks(): Improves the lung masks and displays the final left and right lung masks.
    find_GGO(): Applies a filter on the CT scan to identify Ground Glass Opacities (GGO) and displays the original image with GGO.
    infection_meas(): Measures the extent of infection in the lungs.

Please note that all images are displayed using matplotlib's imshow function.