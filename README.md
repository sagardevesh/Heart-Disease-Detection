# Heart-Disease-Prediction
This project focusses on prediction of heart disease in a person based on the input features given in the heart disease dataset. 

**Task 1:**
**Dataset Description:**

The dataset we are using for this assignment is the heart disease dataset from Kaggle. The dataset dates back to 1988. It consists of 13 input features, and a target feature called ‘target’. Here is the link to the dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data

Below is the brief explanation of each feature:

1.	Age: This feature represents the age of the patient.
2.	Sex: Sex of the patient consisting of 2 classes – Male (represented by ‘1’) and Female (represented by ‘0’).
3.	cp: ‘cp’ is an abbreviation for chest pain and it represents the type of chest pain (1 = typical angina, 2 = atypical angina, 3 = non — anginal pain, 4 = asymptotic.
4.	trestbps: This feature represents the resting blood pressure in mmHg (millimetres of mercury).
5.	chol: Represents Serum cholestoral in mg/dl. 
6.	fbs: Represents Fating Blood Sugar (1 = fasting blood sugar > 120mg/dl, 0 = otherwise)
7.	restecg: Resting Electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy).
8.	thalach: Represents maximum heart rate achieved by the patient. 
9.	exang: Represents exercise induced angina (1 = yes, 0 = no).  
10.	oldpeak: ST depression induced by exercise relative to rest (‘ST’ relates to positions on the ECG plot.
11.	slope: represents the slope of the peak exercise ST segment (0= downsloping; 1= flat; 2= upsloping)
12.	ca: Represents the number of major vessels (0-3)
13.	thal: Represents a blood disorder called thalassemia. (0 = Normal, 1 = fixed defect (no blood flow in some part of the heart), 2 = normal blood flow, 3 = reversible defect (a blood flow is observed but it is not normal)
14.	target: Diagnosis of heart disease (0 = Absent, 1 = Present).
The purpose of using this dataset is to predict whether a person is suffering from a heart disease or not with the help of all the input features. 
Pre-processing:

As far as pre-processing is concerned, the dataset was already pre-processed to some extent. The categorical features were already label encoded. The dataset did not consist of any NULL values. However, there were a lot of duplicate rows, precisely 723 out of 1025 rows were duplicates. After eliminating all the duplicate rows, we were left with 302 records that were to be fed into the models. 

The next thing I did was the segregation of the input and the target variables into two separate dataframes. I created a new dataframe ‘x’ consisting of all the 13 input features, and another dataframe ‘y’ consisting of the target feature ‘target’.

The next task was to check whether all the input features were relevant enough in making the target predictions. For this purpose, I used the Pearson Correlation Coefficient method to perform feature selection. This method recommended to drop the ‘slope’ feature from the input dataframe as it was highly correlated with other input features. Hence, I went ahead and dropped this feature. 

**Task 2:**
The aim in task 2 was to implement K-means algorithm to get the clusters of patients. To begin with, we have to get the optimal number of clusters for representation. For this, I used the elbow method which showed that the optimal number of clusters is 2. Below is the figure for elbow method showing the optimal number of clusters for our dataset. 

![image](https://github.com/sagardevesh/Heart-Disease-Detection/assets/25725480/1b85b5c7-99e1-44b3-871a-6fcc258e4028)

 
To get the optimal number of clusters, I also implemented the Silhouette measure which measures how similar an object is to its own cluster compared to other clusters. It basically measures the goodness of the clustering technique. The higher the score, the better is the clustering technique. I calculated silhouette scores individually for number of clusters = 2,3,4 and 5. I got the highest silhouette score of 0.389 for number of clusters=2, further confirming the results provided by the elbow method. 

For comparative visual analysis for determining the number of clusters using Silhouette measure, I used ‘SilhouetteVisualizer’ module from the visual analysis tool ‘Yellowbrick’. Below is the visualisation, which clearly shows the highest Silhouette score for number of clusters = 2. 

![image](https://github.com/sagardevesh/Heart-Disease-Detection/assets/25725480/b24a81d0-0891-4ab0-9376-c8c33b6c7c63)

 
Once I got the optimum number of clusters=2, the next task was to implement K-Means algorithm with k=2, and get the labels. 

Next, I used Principal Component Analysis (PCA) to display the clusters in a reduced 2D dimension. Although, the assignment's requirement is to illustrate the clusters in 2D dimension, I have also plotted a ‘Cumulative explained variance V/S Number of components’ graph that helps us choose the right number of components. The below graph shows the amount of variance depending on the number of components selected. A general rule of thumb is to preserve around 80% of the variance. Hence, in our case, we can go ahead and select the number of components to be 2 as it has the variance of 0.90 (Slightly above 0.80).

![image](https://github.com/sagardevesh/Heart-Disease-Detection/assets/25725480/ec0fa633-72df-44a5-934a-9b0fc66e1777)

 
Once I got to know the number of components, I implemented PCA with n_components=2, and fit it on the input dataframe x_1. I used the pca.transform method to get the pca scores, which I used to fit the K-Means algorithm model. I created 2 new columns and named it ‘Component 1’ and ‘Component 2’ respectively, and stored the scores_pca values under them. Next I used the kmeans_pca.labels_ method to get the labels of the model, and stored it in a new column named ‘K-Means PCA labels’.

The column ‘K-Means PCA labels’ indicates the final labels after PCA was used, and has 2 labels ‘0’ and ‘1’, where ‘0’ represents ‘No Heart Disease’ and ‘1’ represents ‘Heart Disease’. Once I got the labels, I visualised the clusters using scatter plot, and marked their centroids as well. 

i.	Describe how the k-means algorithm works.
K-means algorithm is an unsupervised clustering algorithm which classifies the unlabelled data into a specified number of clusters. Before implementing this algorithm, we need to find the optimal number of clusters that we need to obtain. This can be done either by using the elbow method, or the Silhouette measure. We then select random datapoints and define them as centroids for each cluster, since the exact centre of the datapoints is unknown. In our case, since we have 2 clusters, we will select 2 random centroids. In the next step, the k-means algorithm assigns each datapoint to its nearest cluster. It chooses the cluster for data points where the distance between the data point and the centroid is minimum. Once all the data points are assigned to their respective clusters, the centroid of each cluster is re-computed by calculating the average of all the datapoints of that cluster, post which we get new centroids. 
Once it gets the new centroids, it computes the distance of each datapoint to each centroid, and assigns that datapoint to the cluster where the distance between the datapoint and the centroid is minimum. This process continues until optimal centroids have been found, and the assignment of each datapoint to the correct clusters are not changing anymore.

**Task 3:**
The next task was to implement the Naïve Bayes and the Decision Tree models. Prior to that, I split the original dataset into train and test data with a 70-30 weightage.  After I implemented the Naïve Bayes classifier, I got an accuracy score of 83.51% for it. 

Similarly, I implemented the Decision Tree classifier with the same train and test data, and got an accuracy score of 73.62%, which is slightly less when compared to the accuracy of the Naïve Bayes model. Additionally, I also went ahead to plot the decision tree to visualise the results.

**Statistical Significance test:**
To compare the performances of the 2 models, I used 5x2cv paired t test. Under this, I assumed the significance threshold to be 0.05 (alpha). The null hypothesis is that both the algorithms perform equally well and are not significantly different. If the p-value is more than the significance threshold, it would mean that we cannot reject the null hypothesis of the 2 algorithms being significantly different. Whereas, if the p-value is less than the significance threshold, it would imply that we can reject the null hypothesis and that would mean that the two models are significantly different. In our case, we got a p-value of 0.015, which is less than the significance threshold of 0.05, hence we could conclude that the Naïve Bayes and the Decision Tree models are significantly different. 

**To run the project:-**

Steps to run on local machine/jupyter notebook:
To run this assignment on your local machine, you need the following software installed.

Anaconda Environment
Python

To install Python, download it from the following link and install it on your local machine:

https://www.python.org/downloads/

To install Anaconda, download it from the following link and install it on your local machine:

https://www.anaconda.com/products/distribution

After installing the required software, clone the repo, and run the following command for opening the code in jupyter notebook.

jupyter notebook Heart_disease_prediction.ipynb
This command will open up a browser window with jupyter notebook loading the project code.

You need to upload the dataset from the git repo to your jupyter notebook environment.

Furthermore, 

1. Install the necessary modules such as pandas, matplotlib, seaborn, numpy, KMeans(from sklearn),  PCA (from sklearn).
2. Use pip install to install the above dependencies

