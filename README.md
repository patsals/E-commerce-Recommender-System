# E-commerce-Recommender-System

## Overview
The goal of this project was to compare different approaches to engineering a recommender system. For this project specifically, we looked at different collaborative filtering models implemented using Sci-kit Learn, H2O ML, and tensorflow in order to come up with a solution for predicting whether or not a user will purchase/put an item into their cart.

## Data
The dataset utilized for this project was obtained from Kaggle (https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store). It contains about 5 months worth of data collected from an online e-commerce website(name is anonymous) by an open source customer data platform. It has 885,129 entries, each corresponding to a user’s interaction on the platform with one of the product listings. 

<img src="data.png"  width="80%">

## Exploratory Data Analysis
### Preliminary Data Quality Checks
It was immediately discovered that there were plenty of problems with our dataset upon investigation:
- **Data types**: incorrect data types for 'event_time', 'user_id', 'product_id', and 'category_id'
- **Duplicate entries**: out of the 885,219 rows we found that there were 655 duplicate rows and they were removed
- **Null values**: present in 'category_code', 'brand', and 'user_session'; it was decided to impute missing values for 'category_code' and 'brand' and ignore the nulls in the 'user_session' column since we have no additional context and it doesn't effect the scope of the project
- **Outlier values**: Although there were some products that did have outlier prices signficantly larger compared to alternatives, these prices were consistent across all of their interaction appearances which lead me to believe they were intentional.
- **Temporal consistency**: according to the description of the dataset, it should contain about 5 months worth of data and after investigating further it appears that this is the case 

### Further Exploration Using Visuals
Since the scope of this project is to develop a recommender system to predict whether or not a user will purse a product(interaction is labeled as either 'cart' or 'purchase', not 'view' only), it is important to identify any underlying correlations between the different labels and other attributes.

<img src="label_distribution.png"  width="60%">
We can see that nearly 89.67% of our data is labeled as 'view' but in order to correctly establish any underlying patterns in the data, we must find the set of products that were __only viewed__ and never actually purchased or put into a cart. 
After doing so, we get the following two sets of proucts segmented:
- Number of products that were labeled as 'purchase' or 'cart': 9,837
- Number of products that were only ever labeled as 'view': 43,616

Now investigating these groups further:

<img src="brand_popularity_distribution.png"  width="60%">
Plotting the 30 biggest brands(determined by the number of unique products sold), we can see that only 5 brands have less than 50% of their products labeled as either 'purchase' or 'cart'.

<img src="interaction_distribution.png"  width="60%">
After examining the number of interactions between the two groups, products taht were labeled as either 'purchase' or 'cart' tended to have a higher number of interactions. This is demonstrated by how wide the whiskers span on the boxplot as well as the overall spread of the data.

<img src="price_distribution.png"  width="60%">
If we consider the different distributions between products that were labeled as either 'puchase' or 'cart' and products that were only ever labeled as 'view', we can see that the distribution for the prior spans more wider. Like before, this is also demonstrated by how wide the whiskers span on the boxplot as well as the overall spread of the data.

<img src="category_popularity_distribution.png"  width="60%">
Looking at the 30 most popular product categories(determined by the number of unique products labeled in that category), we can see that there are some categories with less than 50% of their products labeled as either 'purchase' or 'cart'. If we assumed category to have no effect on whether or not a product is purchased or viewed we should expect each bar to be around the red dotted line, but according to our dataset it doesn't seem like this is the case.






## Models
### Logistic Regression model (Sci-kit Learn)
This model serves as a our baseline since it is a binary predictor with simple implementation and easily tunability. 

### AutoML model (H2o)
Here I wanted to try using H2o's automl model because I wanted to try experimenting it for the first time, especially since it claims to product high-performing machine learning models with little to no tuning required. The models I let it evaluate were Generalized Linear Model, Distributed Random Forest, and Stacked Ensemble; out of all of the models Stacked Ensemble ended up being chosen as the highest performer.

### Collaborative Filtering Model (Tensorflow)
Compared to the other models, I decided to train a neural network with two layers, one for users and one for products, with 50 embedding dimensions in order to model the complex behavior of user interactions. Using this approach, only a large enough dataset consisting of user,product interactions were required to train the neural network and make predictions.

## Feature Engineering (Logistic Regression and AutoML only)
As a result of our exploratory data analysis and intuition regarding consumer behavior, additional features were created to train our model. These features will only be utilized for our baseline Logisitic Regression model and for our AutoML model since our Collaborative Filtering Tensorflow model will utilize an alternative approach. 

Here are the following features engineered:
- most_alike_customer_similarity_score: the greatest Jaccard similarity score between the current user and all other users that did purchase that product
- is_popular_brand: a binary value indicating whether or not if the product's brand was in the top 50% percentile of brands with the most interactions
- is_popular_product: a binary value indicating whether or not if the product had more interactions than compared to the rest of the 50% of the population
- category_code one-hot encoded: one-hot encoding of the product's category code
- brand one-hot encoded: one-hot encoding of the product's brand


## Evaluation Metrics
- Accuracy: percentage of predictions that were actually correct
- F1-score: harmonic mean between precision and recall
- ROC-auc score: receiver operating characteristic, area under the curve which tells us how well a model is able to perform between two classes
  
## Results
- Logistic Regression model(after performing cross-validation grid search):
  
|            | Accuracy | F1-score | ROC-AUC  |
| ---------- | -------- | -------- | -------- |
| Train      | 0.6739   | 0.6299   | -        |
| Valid      | 0.6509   | 0.6150   | -        |
| Test       | 0.6254   | 0.5924   | 0.66     |

- AutoML model(Stacked Ensemble):
  
|            | Accuracy | F1-score | ROC-AUC |
| ---------- | -------- | -------- | -------- |
| Train      | 0.6123   | 0.6904   | -        |
| Valid      | 0.5877   | 0.6785   | -        |
| Test       | 0.5620   | 0.6607   | 0.66     |

- Collaborative Filtering Tensorflow model:
  
|            | Accuracy | F1-score | ROC-AUC |
| ---------- | -------- | -------- | -------- |
| Train      | 0.9981   | 0.9982   | -        |
| Valid      | 0.7299   | 0.7365   | -        |
| Test       | 0.7084   | 0.7229   | 0.79     |

<img src="rocauc_curve.png"  width="80%">

After training and tuning all of our models, it appears that the Collaborative Filtering Tensorflow model was overall the most successful. Across all of the training, validation, and testing sets it obtained the highest accuracy and F1-score. In addition, its performance demmonstrated on the ROC-AUC curve highlights its ability to make accurate positive predictions.


## Final thoughts
Although the purpose of this project was to create a recommender system and all I did here was train a binary predictor, the next steps required to fully implement it would be fairly straight forward. All I would need to do is create an API endpoint that takes in the current user and a list of products. The list would most likely contain the entire store's database of product ids and using our tensorflow model we can return the top N products according to the largest probabilities. While this is most likely how it would work for the storefront page when a user initially logs in, I believe it can easily be altered to recommend products based on the on the user is currently viewing as well.

In conclusion, it was interesting to evaluate the different models and observe how effective a more-complicated model can be compared to other simple implementations. Even though I thought I performed sufficient feature engineering, the neural network being able to identify the underlying latent factors and relationships allowed it to perform the best -- even when it is only trained on (user,pair) interactions dataset.



