# Exploring Risk Factors for Cardiovascular Disease
This project analyzes data from the "Exploring Risk Factors for Cardiovascular Disease" dataset, which contains information about patients and whether or not they have cardiovascular disease. The project uses Python and several data analysis and machine learning libraries to explore the dataset and build a predictive model for cardiovascular disease.

## Getting Started
### Prerequisites:
To run this code, you will need the following libraries:
- numpy
- pandas
- seaborn
- matplotlib
- sklearn

You can install them using pip.

### Clone the repository:
```shell
$ git clone https://github.com/Tayyab885/Risk-Factors-For-Cardiovascular-Heart-Disease.git
```

### Dataset:
The dataset used in this project is available in the repo. The dataset has 13 columns:
- id - unique identifier for each patient
- age - age of the patient in days
- gender - gender of the patient (1 = female, 2 = male)
- height - height of the patient in cm
- weight - weight of the patient in kg
- ap_hi - systolic blood pressure
- ap_lo - diastolic blood pressure
- cholesterol - cholesterol level (1 = normal, 2 = above normal, 3 = well above normal)
- gluc - glucose level (1 = normal, 2 = above normal, 3 = well above normal)
- smoke - whether or not the patient is a smoker (0 = no, 1 = yes)
- alco - whether or not the patient drinks alcohol (0 = no, 1 = yes)
- active - whether or not the patient is physically active (0 = no, 1 = yes)
- cardio - whether or not the patient has cardiovascular disease (0 = no, 1 = yes)

### Exploratory Data Analysis:
The first step is to explore the data and understand the relationships between the variables. The df.info() and df.describe() methods provide information about the data types and statistics of the variables. The output of these methods shows that there are no missing values in the dataset.
To gain more insights about the data, we also create some visualizations. We use matplotlib and seaborn libraries to plot various types of graphs.
#### Categorical Variables:
We create a bar chart to visualize the distribution of each categorical variable in the dataset.
#### Continuous Variables:
We use a histogram to visualize the distribution of each continuous variable in the dataset.
#### Barplots with respect to Target Feature:
We create a bar chart to visualize the distribution of each categorical variable with respect to the target feature.
These visualizations help us to understand the relationship between the independent and target variables, which can be useful in selecting features for our machine learning models.

### Data Preprocessing:
- The dataset is then preprocessed to prepare it for machine learning. The train_test_split() function is used to split the data into training and testing sets.
- The StandardScaler() function is used to scale the data.

### Machine Learning:
The next step is to build a machine learning model to predict whether or not a patient has cardiovascular disease. In this project, an XGBoost Classifier is used. The model is trained on the training set and tested on the testing set. The RandomizedSearchCV method is used for hyperparameter tuning to optimize the model's performance.

### Results:
The accuracy of the model is 74%. The confusion matrix and classification report show the precision, recall, and F1-score of the model.

### Conclusion:
In this project, we explored the risk factors for cardiovascular disease and built a machine learning model using XGBoost Classifier to predict whether or not a patient has cardiovascular disease. The results show that the model has an accuracy of 74% and can be used to predict cardiovascular disease.
