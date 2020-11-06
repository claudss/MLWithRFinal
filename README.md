# Machine Learning Using R Final Project
Final project for Machine Learning Using R course at UCLA Extension, completed as part of Data Science certification. 

## Table of Contents
* [Summary](#summary)
* [Data](#data)
* [Exploring Models](#exploring-models)
* [Results](#results)
* [Conclusions](#conclusions)

## Summary
In this project, I analyzed a dataset collected on voting information from the US Presidential Primary election in 2016, compiled by Kaggle co-founder Ben Hamner, which can be found online [here](https://www.kaggle.com/benhamner/2016-us-election/). My goal was to explore machine learning models to find which would best be able to predict a candidate winning in an election, based on demographic data. This is a classification problem, as it is attempting to predict a label- in this case, the name of a candidate that will win in a given county. However, it could also alternatively have been framed as a regression problem, attempting to predict the number of votes a candidate will gain.


## Data
Per county, this dataset contains: population estimates, the percentage of the population occupied by various ethnicities or certain age brackets, housing status, education status, income level, number of firms in the county, and what proportion of those are attributed to owners of certain ethnicities. While more obscure factors also appeared (e.g. median time to travel to work), I wanted to focus on attributes that looked more directly relevant to what could influence a person's vote, which immediately reduced the amount of possible predictors for models from 51 to 29.

While the original dataset does contain county information for all 50 states, exploration was narrowed down to the major West Coast states- California, Oregon, and Washington. Since the dataset contained some slight inconsistencies that were difficult to standardize into its general format, such as data for the state of Kansas being counted by Congressional district rather than by county, those three states were selected for the sake of time and the manageable scale of the project. Given the heavy political leanings of the region, the candidate options I examined were the two major Democratic candidates of the time, Bernie Sanders and Hillary Clinton. 

The demographic and voting data I left almost completely untouched after my initial cutting down of considered features, as explained previously. However, I did find out through manual trial and error during coding that the variable referred to as nthawown, which indicated the amount of firms owned by native Hawaiians/Pacific Islanders, had to be removed from consideration- unfortunately, it was so sparse and almost entirely filled with zeroes that including it immediately broke any models I attempted to use. 

To start off, I visualized the demographic data using R library `ggplot2`.

![Demographics](https://github.com/claudss/MLWithRFinal/blob/main/Pictures/LargeGraphs1.png)

These three plots have an x axis representing the percentage of a county's population recorded as white, and a y axis representing the percentage of a county's population that was recorded as Hispanic/Latino, Black, and Asian respectively- the more "major" demographics typically courted by American political campaigns. Red dots represent counties that Bernie Sanders won, and blue dots represent counties that Hillary Clinton won. These demonstrate that predominantly white counties tended to go for Clinton or Sanders with relatively equal frequency. Counties with high Asian populations were rather sparse, but in them Clinton was more likely to win. There was a striking difference in the plots of Hispanic/Latino and Black populations, particularly the middle plot that has white population as the x axis and Black population as the y axis. A Clinton win was nearly guaranteed in counties that had majority Black population recorded in the data, with a similar trend occurring to a less drastic degree in the first plot of Hispanic/Latino population as opposed to white population.


## Exploring Models
I tested a variety of different machine learning approaches on the data: ridge regression, lasso regression, linear regression, and random forest. Ridge regression is an upgrade to standard linear regression that is useful when there is multicollinearity between variables, i.e. they have some degree of interdependence. This demographic data fits that criteria, especially due to intersecting statistics that involve both racial and economic measures of the population. Lasso regression happens to be an upgrade to ridge regression, but with the key difference that it will set coefficients to 0 under certain circumstances, effectively performing variable selection as part of its process- this was of interest to me, as it could help to whittle down a large predictor pool to something more concise.

Results were analyzed through standard methods with all predictors used, and through a fine-tuning/subset method- for ridge and lasso regression, this meant fine-tuning by tenfold cross-validation, and for linear regression and random forest this meant using the subset of predictors deemed most relevant by lasso regression. Effectiveness was examined via confusion matrices and ROC curves. For example, here are the confusion matrices of standard and fine-tuned ridge regression:

![Ridge1](https://github.com/claudss/MLWithRFinal/blob/main/Pictures/Ridge_Confusion.png) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ![Ridge2](https://github.com/claudss/MLWithRFinal/blob/main/Pictures/RidgeFinetuned_Confusion.png)


As an additional example, this is the ROC curve comparing the effectiveness of both (with standard ridge regression represented with a black line, and fine-tuned ridge regression with a red line):

![RidgeROC](https://github.com/claudss/MLWithRFinal/blob/main/Pictures/Comparison_RidgeROC.png)

## Results
This table shows the final measures taken on each model type:
| Algorithm |	Train Error Rate |	Test Error Rate	| AUC |
| :---: | :---: | :---: | :---: |
|Ridge Regression | 47.3% |	45.3% |	0.79 |
|Tuned Ridge Regression | 24.6% | 24.6% | 0.87 |
|Lasso Regression | 56.8% | 54.9% | 0.5 |
|Tuned Lasso Regression | 24.9% | 23.5% | 0.87 |
|Logistic Regression | 22.2% | 22.2% | 0.88 |
|Logistic Regression (Subset) | 22.4% | 22.4% | 0.87 |
|Random Forest | N/A | 21.5% | 0.89 |
|Random Forest (Subset) | N/A | 21.3% | 0.89 |

Note: For the random forest, it is important to note that train error is a mostly unhelpful statistic. Due to how R grows random forests, if you grow at the default/recommended settings, it will basically always operate perfectly on the training set. Not only did I observe this as I ran my code, I was lucky enough to discover a forum post by the actual admin of the randomForest package explaining this, which can be found [here](https://r.789695.n4.nabble.com/Random-Forest-AUC-td3006649.html#a3008074). I did log the OOB error generated with the model, but it is not the same as an actual training error stat, so I did not count it as such.


## Conclusions
Random forest (with subset of predictors) looked most promising out of all the options, as it is able to generalize to a test set with almost 80% accuracy. From a more general angle, random forest is also good at making sense out of what seems a messy mix of different factors, and even works in a way similar to the ideal of voting- through it, R grows an ensemble of decision trees whose choices each play their own role in the model's final selections. The immediate next step after a smaller-scale endeavor like this would be to expand out from more than just the West Coast, and also to look at things in terms of more than just one party. This could happen not only by moving to examine the data of every state available in this dataset, but also by cordoning off specific hotspots for party loyalty and analyzing the demographics and likelihoods there- such as, for example, analyzing the key demographics in the "Rust Belt" area of northern states, which contains some of the important election "swing states" like Ohio and Michigan. Because such states tend to have close contests between winning candidates of both parties during both the primary and full presidential elections, exploring what demographics contribute to accurate predictions of winners there (across party lines) would be the ideal future choice. Location-locked predictions can be lined up together to better understand trends in key regions, and how to cater to those trends.
