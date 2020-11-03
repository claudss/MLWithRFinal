# CODE BEGINS
#Note: it looks like running incrementally model by model/step by step (steps as outlined through comments), rather than all at once, should yield results that will match the report exactly
setwd("D:/Desktop/CSX_ALL/R/R_FinalProject/")
# predictions: democratic winners in the 2016 US presidential primary
RNGkind(sample.kind = "Rounding")
library(dplyr) # setup
library(ggplot2)
library(plotly)
library(grid)
library(gridExtra)
library(DT)
library(GGally)
library(randomForest)
library(ROCR)
library(glmnet)
library(pROC)

# read in both data sheets
primary <- read.csv("KaggleData/primary_results.csv", stringsAsFactors = FALSE) 
demographics <- read.csv("KaggleData/county_facts.csv", stringsAsFactors = FALSE)

alldata = TRUE # boolean: do we consider all states? set as TRUE for now.

# here, we filter the votes dataset so we just see stats for Democratic candidates
votes <- primary %>%  
  filter(party == "Democrat") %>% 
  group_by(state_abbreviation, county) %>% 
  summarize(winner = candidate[which.max(fraction_votes)],
            Vote = max(fraction_votes),
            votes = max(votes))

# here, we take into account every demographic column in the demographic dataset (county_facts)
# we also filter this to 3 select states 
demographics %<>% 
  select(state_abbreviation = state_abbreviation, county = area_name, pop14=PST045214, agel5=AGE135214, agel18 = AGE295214, age65=AGE775214, belowpovertyline=PVY020213, totalfirms=SBO001207,area=LND110210,
         female=SEX255214, white=RHI125214, black_afro=RHI225214, american_indian=RHI325214, asian=RHI425214, nathaw=RHI525214, tworace=RHI625214,blkowned=SBO315207,amerindown=SBO115207,womenowned=SBO015207,
         hislat=RHI725214, highschool=EDU635213, bachelor = EDU685213, veteran=VET605213, homeowner=HSG445213, medianvalhouse =HSG495213, personphh=HSD310213, percapitaincome=INC910213, medianincome=INC110213, asianowned=SBO215207, nathawown=SBO515207, hisowned=SBO415207, density = POP060210) %>% 
  mutate(county = gsub(" County", "", county))

# condition in the event filtering is desired
if (alldata == FALSE) {
  demographics %<>% 
    filter(state_abbreviation %in% c("CA", "OR", "WA"))
}

# get our full dataset by combining everything using helpful SQL-esque "inner join" command
votes <- inner_join(votes, demographics, by = c("state_abbreviation","county")) 
# double check if there are any candidates who won other than the two we expected (there aren't)
unique(votes$winner) 
# view an easy table of our values, just to look over things ourselves
datatable(votes, class = 'compact')

# IMPORTANT SECTION: ESTABLISH TRAIN/TEST SETS
train_index <- sample(1:nrow(votes), 0.8 * nrow(votes))
test_index <- setdiff(1:nrow(votes), train_index)
#x_train <- votes[train_index, -3]
votestrain <- votes[train_index, -c(1:2)] # remove non-numeric columns of state/county names, these trip up models
votestrain <- subset(votestrain, select=-c(nathawown, Vote, votes)) # i had to remove column "nathawown" from consideration. the reason: it was sadly too filled with zeroes and threw off the models
votestrain$winner = as.factor(votestrain$winner) # set non-numeric "winner" column to be a factor so we can perform easy comparison
# remove items same as above
votestest <- votes[test_index, -c(1:2)]
votestest <- subset(votestest, select=-c(nathawown, Vote, votes))
votestest$winner = as.factor(votestest$winner)

rocplot=function(pred, truth, ...) { # create function to plot an ROC curve
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf, ...)
}


#counties by winner, percentage of white vs hispanic/latino population per candidate 
qplot(x =  white, y = hislat, data = votes, 
               color = winner, size = Vote)

# counties by winner, percentage of white vs black population per candidate
qplot(x =  white, y = black_afro, data = votes, 
               color = winner, size = Vote)

# counties by winner, percentage of white vs asian population per candidate
qplot(x =  white, y = asian, data = votes, 
      color = winner, size = Vote)



# ----- MODEL 1: RIDGE REGRESSION -----
set.seed(210)
x_train = model.matrix(winner~.,votestrain)
y_train = votestrain$winner
x_test = model.matrix(winner~.,votestest)
y_test = votestest$winner
grid=10^seq(10,-2, length =100)

set.seed(210)
ridge.mod=glmnet(x_train,y_train,alpha=0,lambda=grid, family="binomial", thresh=1e-12)
ridge.trainpr=predict(ridge.mod,s=4,newx=x_train) # get predictions for our test set
ridge.trainpred <- ifelse(ridge.trainpr > 0.5, "Hillary Clinton", "Bernie Sanders")
mean(y_train!=ridge.trainpred) 

set.seed(210)
ridge.probs=predict(ridge.mod,s=4,newx=x_test) # get predictions for our test set
ridge.pred <- ifelse(ridge.probs > 0.5, "Hillary Clinton", "Bernie Sanders")
mean(y_test!=ridge.pred) # average error rate of about 50%... not great

table(ridge.trainpred, y_train)
table(ridge.pred, y_test)


# attempt to tune
set.seed(210)
cv.ridge=cv.glmnet(x_train,y_train, family="binomial",alpha=0) # use 10-fold cross validation to choose our tuning param lambda
plot(cv.ridge)
bestlambda = cv.ridge$lambda.min
bestlambda


# let's try using our new tuned lambda value!
set.seed(210)
ridge.trainpr2=predict(ridge.mod,s=bestlambda,newx=x_train) # get predictions for our test set
ridge.trainpred2 <- ifelse(ridge.trainpr2 > 0.5, "Hillary Clinton", "Bernie Sanders")
mean(y_train!=ridge.trainpred2) 

set.seed(210)
ridge.newprobs=predict(ridge.mod,s=bestlambda,newx=x_test) # get predictions for our test set
ridge.newpred <- ifelse(ridge.newprobs > 0.5, "Hillary Clinton", "Bernie Sanders")
mean(y_test!=ridge.newpred) # accuracy has gone up! about 75% accuracy now. but, can we do better?

table(ridge.trainpred2, y_train)
table(ridge.newpred, y_test)


# ROC and AUC acquiring section
set.seed(210)
ridge.rocprobs=predict(ridge.mod,s=4,newx=x_test, type="response")
ridge.rocpred <- prediction(ridge.rocprobs, y_test)
ridgeprf <- performance(ridge.rocpred, measure = "tpr", x.measure = "fpr")

set.seed(210)
ridge.rocprobs2=predict(ridge.mod,s=bestlambda,newx=x_test, type="response")
ridge.rocpred2 <- prediction(ridge.rocprobs2, y_test)
ridgeprf2 <- performance(ridge.rocpred2, measure = "tpr", x.measure = "fpr")

plot(ridgeprf)
plot(ridgeprf2, col="red", add=TRUE)

set.seed(210)
auc_ridge1 <- performance(ridge.rocpred, measure = "auc")
auc_ridge1 <- auc_ridge1@y.values[[1]]
set.seed(210)
auc_ridge2 <- performance(ridge.rocpred2, measure = "auc")
auc_ridge2 <- auc_ridge2@y.values[[1]]

auc_ridge1
auc_ridge2


# ----- MODEL 3: LASSO REGRESSION -----
# let's start out same as above
set.seed(210)
lasso.mod=glmnet(x_train,y_train,alpha=1,lambda=grid, family="binomial")
lasso.trainpr=predict(lasso.mod,s=5,newx=x_train) # get train scores
lasso.trainpred <- ifelse(lasso.trainpr > 0.5, "Hillary Clinton", "Bernie Sanders")
mean(y_train!=lasso.trainpred) # average accuracy of about 43%... not a great start either!

set.seed(210)
lasso.probs=predict(lasso.mod,s=4,newx=x_test) # get predictions for our test set
lasso.pred <- ifelse(lasso.probs > 0.5, "Hillary Clinton", "Bernie Sanders")
mean(y_test!=lasso.pred) # average accuracy of about 47%

table(lasso.trainpred, y_train)
table(lasso.pred, y_test)


# attempt to tune
set.seed(210)
cv.lasso=cv.glmnet(x_train,y_train, family="binomial",alpha=1) # cross-validate again
plot(cv.lasso)
l_bestlambda = cv.lasso$lambda.min
l_bestlambda

set.seed(210)
lasso.trainpr2=predict(lasso.mod,s=l_bestlambda,newx=x_train) # get predictions for our test set
lasso.trainpred2 <- ifelse(lasso.trainpr2 > 0.5, "Hillary Clinton", "Bernie Sanders")
mean(y_train!=lasso.trainpred2) 

set.seed(210)
lasso.newprobs=predict(lasso.mod,s=l_bestlambda,newx=x_test) # get predictions for our test set
lasso.newpred <- ifelse(lasso.newprobs > 0.5, "Hillary Clinton", "Bernie Sanders")
mean(y_test!=lasso.newpred) # accuracy has gone up! about 70% accuracy once more.
lasso.coefs = predict(lasso.mod,s=l_bestlambda,type="coefficients") 
lasso.coefs

table(lasso.trainpred2, y_train)
table(lasso.newpred, y_test)


# ROC and AUC acquiring section
set.seed(210)
lasso.rocprobs=predict(lasso.mod,s=4,newx=x_test, type="response")
lasso.rocpred <- prediction(lasso.rocprobs, y_test)
lassoprf <- performance(lasso.rocpred, measure = "tpr", x.measure = "fpr")

set.seed(210)
lasso.rocprobs2=predict(lasso.mod,s=l_bestlambda,newx=x_test, type="response")
lasso.rocpred2 <- prediction(lasso.rocprobs2, y_test)
lassoprf2 <- performance(lasso.rocpred2, measure = "tpr", x.measure = "fpr")

plot(lassoprf)
plot(lassoprf2, col="red", add=TRUE)

auc_lasso1 <- performance(lasso.rocpred, measure = "auc")
auc_lasso1 <- auc_lasso1@y.values[[1]]
auc_lasso2 <- performance(lasso.rocpred2, measure = "auc")
auc_lasso2 <- auc_lasso2@y.values[[1]]

auc_lasso1
auc_lasso2


# ----- MODEL 3: LOGISTIC REGRESSION -----
set.seed(210)
glm.fitall <- glm(winner ~ ., family="binomial", votestrain)
summary(glm.fitall)

# training score
set.seed(210)
glm.trainprobs = predict(glm.fitall, votestrain, type="response")
glm.trainpred=rep("Bernie Sanders",nrow(votestrain))
glm.trainpred[glm.trainprobs >.5]="Hillary Clinton"
mean(glm.trainpred!=votestrain$winner)

# testing score
set.seed(210)
glm.probs = predict(glm.fitall, votestest, type="response")
glm.pred=rep("Bernie Sanders",nrow(votestest))
glm.pred[glm.probs >.5]="Hillary Clinton"
mean(glm.pred!=votestest$winner)

table(glm.trainpred, y_train)
table(glm.pred, y_test)


# let's try something new: use only the variables shown in lasso regression. does this help things?
set.seed(210)
glm.fit2 <- glm(winner~agel18+age65+female+white+black_afro+
                  asian+tworace+hislat+highschool+bachelor+veteran+
                  medianvalhouse+personphh+percapitaincome+asianowned+
                  hisowned, family="binomial", votestrain)
summary(glm.fit2)

# training score
set.seed(210)
glm.trainprobs2 = predict(glm.fit2, votestrain, type="response")
glm.trainpred2=rep("Bernie Sanders",nrow(votestrain))
glm.trainpred2[glm.trainprobs2 >.5]="Hillary Clinton"
mean(glm.trainpred2!=votestrain$winner)

set.seed(210)
glm.probs2 = predict(glm.fit2, votestest, type="response")
glm.pred2=rep("Bernie Sanders",nrow(votestest))
glm.pred2[glm.probs2 >.5]="Hillary Clinton"
mean(glm.pred2!=votestest$winner) # as it turns out, the result isn't much different.

table(glm.trainpred2, y_train)
table(glm.pred2, y_test)


# ROC and AUC acquiring section
set.seed(210)
lin.rocpred <- prediction(glm.probs, y_test)
linprf <- performance(lin.rocpred, measure = "tpr", x.measure = "fpr")

set.seed(210)
lin.rocpred2 <- prediction(glm.probs2, y_test)
linprf2 <- performance(lin.rocpred2, measure = "tpr", x.measure = "fpr")

plot(linprf)
plot(linprf2, col="red", add=TRUE)

auc_logit1 <- performance(lin.rocpred, measure = "auc")
auc_logit1 <- auc_logit1@y.values[[1]]
auc_logit2 <- performance(lin.rocpred2, measure = "auc")
auc_logit2 <- auc_logit2@y.values[[1]]

auc_logit1
auc_logit2


# ----- MODEL 4: RANDOM FOREST -----
# i decided to try two different random forest methods here.
# one using every variable, and one using only the variables outlined by lasso regression... 
# will there be a difference?
set.seed(210)
rfmodel1 <- randomForest(winner~., votestrain, mtry=15, importance=TRUE)
rf.pred <- predict(rfmodel1, x_test)
print(rfmodel1) 
table(rf.pred, y_test)
mean(rf.pred!=votestest$winner)

set.seed(210)
rfmodel2 <- randomForest(winner~agel18+age65+female+white+black_afro+
                           asian+tworace+hislat+highschool+bachelor+veteran+
                           medianvalhouse+personphh+percapitaincome+asianowned+
                           hisowned, data=votestrain, importance=TRUE)
pred2 <- predict(rfmodel2, x_test)
print(rfmodel2) 
table(pred2, y_test)
mean(pred2!=votestest$winner)

# ROC and AUC acquiring section
set.seed(210)
rf.rocpred <- predict(rfmodel1, x_test, type="prob")
foo = rf.rocpred[,2]
names(foo) = c()
set.seed(210)
rf.rocpredic = prediction(foo, y_test)
rfprf <- performance(rf.rocpredic, measure = "tpr", x.measure = "fpr")

set.seed(210)
rf.rocpred2 <- predict(rfmodel2, x_test, type="prob")
foo2 = rf.rocpred2[,2]
names(foo2) = c()
set.seed(210)
rf.rocpredic2 = prediction(foo2, y_test)
rfprf2 <- performance(rf.rocpredic2, measure = "tpr", x.measure = "fpr")

plot(rfprf)
plot(rfprf2, col="red", add=TRUE)

set.seed(210)
auc_rf1 <- performance(rf.rocpredic, measure = "auc")
auc_rf1 <- auc_rf1@y.values[[1]]
set.seed(210)
auc_rf2 <- performance(rf.rocpredic2, measure = "auc")
auc_rf2 <- auc_rf2@y.values[[1]]

auc_rf1
auc_rf2
