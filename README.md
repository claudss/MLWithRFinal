# Machine Learning Using R Final Project
Final project for Machine Learning Using R course at UCLA Extension, completed as part of Data Science certification. 

## Table of Contents

## Summary
In this project, I analyzed a dataset collected on voting information from the US Presidential Primary election in 2016, compiled by Kaggle co-founder Ben Hamner- it can be found online [here](https://www.kaggle.com/benhamner/2016-us-election/). My goal was to explore machine learning models to find which would best be able to predict which candidate would win in an election, based on demographic data.


## Data
Per county, this dataset contains: population estimates, the percentage of the population occupied by various ethnicities or certain age brackets, housing status, education status, income level, number of firms in the county, and what proportion of those are attributed to owners of certain ethnicities. While more obscure factors also appeared (e.g. median time to travel to work), I wanted to focus on attributes that looked more directly relevant to what could influence a person's vote, which immediately reduced the amount of possible predictors for models from 51 to 29.

While the original dataset does contain county information for all 50 states, exploration was narrowed down to the major West Coast states- California, Oregon, and Washington. Since the dataset contained some slight inconsistencies that were difficult to standardize into its general format, such as data for the state of Kansas being counted by Congressional district rather than by county, those three states were selected for the sake of time and the manageable scale of the project. Given the heavy political leanings of the region, the candidate options I examined were the two major Democratic candidates of the time, Bernie Sanders and Hillary Clinton. 

The demographic and voting data I left almost completely untouched after my initial cutting down of considered features, as explained previously. However, I did find out through manual trial and error during coding that the variable referred to as nthawown, which indicated the amount of firms owned by native Hawaiians/Pacific Islanders, had to be removed from consideration- unfortunately, it was so sparse and almost entirely filled with zeroes that including it immediately broke any models I attempted to use. 

To start off, I visualized the demographic data using R library `ggplot2`.

(IMAGE)

These three plots have an x axis representing the percentage of a county's population recorded as white, and a y axis representing the percentage of a county's population that was recorded as Hispanic/Latino, Black, and Asian respectively- the more "major" demographics typically courted by American political campaigns. Red dots represent counties that Bernie Sanders won, and blue dots represent counties that Hillary Clinton won. These demonstrate that predominantly white counties tended to go for Clinton or Sanders with relatively equal frequency. Counties with high Asian populations were rather sparse, but in them Clinton was more likely to win. There was a striking difference in the plots of Hispanic/Latino and Black populations, particularly the middle plot that has white population as the x axis and Black population as the y axis. A Clinton win was nearly guaranteed in counties that had majority Black population recorded in the data, with a similar trend occurring to a less drastic degree in the first plot of Hispanic/Latino population as opposed to white population.


## Exploring Models
I tested a variety of different models on the data: ridge regression, lasso regression, linear regression, and random forest. I also introduced the extra measure of best subset selection to investigate how much of a difference may have occurred.



## Conclusions
