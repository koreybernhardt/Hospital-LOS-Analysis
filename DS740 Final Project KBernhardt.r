################################################
## D740 Final R Submission for Korey Bernhardt
## Submitted 12/15/2020
################################################

################################################
###Load Libraries and Data
################################################
library(arules)
library(arulesViz)
library(car)
library(pROC)
library(boot)

los = read.csv("train.csv") #only train data has Stay data, which is needed for supervised learning methods
los = na.omit(los) #Available Beds has 113 NA values

summary(los)

################################################
#Association Rules 
################################################

los.ar = los

#create high los variable for stays over and under 30 days
los.ar$los.high = with(los.ar, ifelse((Stay == "0-10" | Stay == "11-20" | Stay == "21-30"), 0, 1))

attach(los.ar)

#discretize continuous predictors by qartile
summary(Available.Extra.Rooms.in.Hospital)
Available.Extra.Rooms.in.Hospital.disc = discretize(Available.Extra.Rooms.in.Hospital, "fixed", breaks=c(0, 2, 4, 24), ordered=T)        
summary(los$Visitors.with.Patient)
Visitors.with.Patient.disc = discretize(Visitors.with.Patient, "fixed", breaks=c(0, 2, 4, 32), ordered=T) 
summary(los$Admission_Deposit)
Admission_Deposit.disc = discretize(Admission_Deposit, "fixed", breaks=c(1800, 4186, 5409, 11008), ordered=T) 

#make factors
Hospital_type_code= as.factor (Hospital_type_code)
Hospital_region_code= as.factor (Hospital_region_code)
Available.Extra.Rooms.in.Hospital= as.factor (Available.Extra.Rooms.in.Hospital.disc)
Department= as.factor (Department)
Ward_Type= as.factor (Ward_Type)
Ward_Facility_Code= as.factor (Ward_Facility_Code)
Bed.Grade= as.factor (Bed.Grade)
Type.of.Admission= as.factor (Type.of.Admission)
Severity.of.Illness= as.factor (Severity.of.Illness)
Visitors.with.Patient= as.factor (Visitors.with.Patient.disc)
Age= as.factor (Age)
Admission_Deposit= as.factor (Admission_Deposit.disc)
los.high = as.factor(los.high)

#Create data frame for association rules modeling
los.df= data.frame(Hospital_type_code,Hospital_region_code,Available.Extra.Rooms.in.Hospital,Department,Ward_Type,
                   Ward_Facility_Code,Bed.Grade,Type.of.Admission,Severity.of.Illness,Visitors.with.Patient,Age,
                   Admission_Deposit,los.high)


#Create association rules
los.rules = apriori(los.df, parameter = list(support = .03, confidence = 0.5))
summary(los.rules)

#This was used, with Stay included in the data frame, to analyze where each separate stay classification
#was a consequent.  They didn't appear separately, which led to creating the high stay variable
# inspect(subset(los.rules, subset = rhs %ain% c("Stay=41-50")))

#Find subset of rules where length of stay is greater than 30 days
high.los.rules = apriori(los.df, parameter = list(support = .03, confidence = 0.5),
                    appearance = list(rhs = c("los.high=1"), default = "lhs") )
#Show top 10 by lift
arules::inspect(head(high.los.rules, n = 10, by = "lift")) 

#Remove redundant rules and show top 10 by lift
high.nonRedundant = which(interestMeasure(high.los.rules,
                                     measure = "improvement",
                                     quality_measure = "confidence") >= 0)
summary( high.los.rules[high.nonRedundant] )
high.los.rules.nr = high.los.rules[high.nonRedundant] 
arules::inspect(head(high.los.rules.nr, n=10, by="lift"))


subrules.high = head(high.los.rules.nr, n = 10, by = "lift")
plot(subrules.high, method="grouped")


################################################
#Logistic Regression
################################################

#Note that I changed to using Logistic Regression instead of ANN because I was interested
#in seeing if there was a way to combine this with the Association Rules, for example did 
#the predictors showing up repeatedly in the top 10 association rules also create a logistic
#model that was better than other.

los.lr= subset(los, select = -c(1,2,4,11,12))
los.lr$los.high = with(los.lr, ifelse((Stay == "0-10" | Stay == "11-20" | Stay == "21-30"), 0, 1))

attach(los.lr)

#Examining model options
#This model uses predictors that occured repeatedly in the top 10 association rules
fit1 = glm(los.high ~ Admission_Deposit + Type.of.Admission + Ward_Type + Available.Extra.Rooms.in.Hospital + Department + Visitors.with.Patient,data=los.lr,family=binomial)
summary(fit1)
#AIC: 329229
#CVError: 0.1705133
#Area under the curve: 0.7766

#This model is the fulle model excluding predictors that resulted in a high VIF
fit2 = glm(los.high ~.-Hospital_type_code-Ward_Facility_Code-Stay-Hospital_region_code,data=los.lr,family=binomial)
summary(fit2)
#AIC: 325844
#CVError: 0.1685354
#Area under the curve: 0.7859

#This model uses predictors that seem to be common and potentially indicative of length of stay.  
#I wanted to see if adding the additional data available improved the model
fit3 = glm(los.high ~ Type.of.Admission + Severity.of.Illness + Department + Age,data=los.lr,family=binomial)
summary(fit3)
#AIC: 417560
#CVError: 0.2363541
#Area under the curve: 0.5807

#Full model, without high VIF predictors, is best model of these 3.  Model using predictors based on 
#association rules performs better than "common" predictors.


#Cross Validation on all of the models
fit1 = glm(los.high ~ Admission_Deposit + Type.of.Admission + Ward_Type + Available.Extra.Rooms.in.Hospital + Department + Visitors.with.Patient,data=los.lr,family=binomial)
CVModel1 = cv.glm(los.lr,fit1,K=10)
CVErrorModel1 = CVModel1$delta[1]

fit2 = glm(los.high ~.-Hospital_type_code-Ward_Facility_Code-Stay-Hospital_region_code,data=los.lr,family=binomial)
CVModel2 = cv.glm(los.lr,fit2,K=10)
CVErrorModel2 = CVModel2$delta[1]

fit3 = glm(los.high ~ Type.of.Admission + Severity.of.Illness + Department + Age,data=los.lr,family=binomial)
CVModel3 = cv.glm(los.lr,fit3,K=10)
CVErrorModel3 = CVModel3$delta[1]

CVErrorModel1; CVErrorModel2; CVErrorModel3

#Predictions and AUC
set.seed(321, sample.kind = "Rounding")

n = dim(los.lr)[1] 
k=10 
groups = c(rep(1:k,floor(n/k)),1:(n-floor(n/k)*k))
cvgroups = sample(groups,n)
predictvals1 = rep(-1,n)
predictvals2 = rep(-1,n)
predictvals3 = rep(-1,n)

for(i in 1:k){
  groupi = (cvgroups==i)
  
  fit1 = glm(los.high ~ Admission_Deposit + Type.of.Admission + Ward_Type + Available.Extra.Rooms.in.Hospital + Visitors.with.Patient,data=los.lr[!groupi,],family=binomial)
  predictvals1[groupi]=predict(fit1,los.lr[groupi,],type = "response")

  fit2 = glm(los.high ~.-Hospital_type_code-Ward_Facility_Code-Stay-Hospital_region_code,data=los.lr[!groupi,],family=binomial)
  predictvals2[groupi]=predict(fit2,los.lr[groupi,],type = "response")

  fit3 = glm(los.high ~ Type.of.Admission + Severity.of.Illness + Department + Age,data=los.lr[!groupi,],family=binomial)
  predictvals3[groupi]=predict(fit3,los.lr[groupi,],type = "response")
}

myroc1 = roc(response=los.lr$los.high,predictor=predictvals1)
myroc2 = roc(response=los.lr$los.high,predictor=predictvals2)
myroc3 = roc(response=los.lr$los.high,predictor=predictvals3)
plot.roc(myroc1, main = "ROC for AR based Model", print.auc= TRUE)
plot.roc(myroc2, main = "ROC for Full Model", print.auc= TRUE)
plot.roc(myroc3, main = "ROC for Common Predictors Model", print.auc= TRUE)
auc(myroc1)
auc(myroc2)
auc(myroc3)

#Confusion matrices
T1 = table(predictvals1 >0.5,los.lr$los.high);T1
T2 = table(predictvals2 >0.5,los.lr$los.high);T2
T3 = table(predictvals3 >0.5,los.lr$los.high);T3

#Error Rate from these predictions
cv1 = (T1[1,2] + T1[2,1])/n; cv1
cv2 = (T2[1,2] + T2[2,1])/n; cv2
cv3 = (T3[1,2] + T3[2,1])/n; cv3


############################################################
#Double Cross Validation - for Logistic Regression Models
############################################################

dim(los.lr)
n = dim(los.lr)[1]
names(los.lr) 

# specify models to consider
#Logistic Models
LogRegModel1 = (los.high ~ Admission_Deposit + Type.of.Admission + Ward_Type + Available.Extra.Rooms.in.Hospital + Department + Visitors.with.Patient)
LogRegModel2 = (los.high ~.-Hospital_type_code-Ward_Facility_Code-Stay-Hospital_region_code)
LogRegModel3 = (los.high ~ Type.of.Admission + Severity.of.Illness + Department + Age)
allLogRegModels = list(LogRegModel1,LogRegModel2,LogRegModel3)	
nLogRegModels = length(allLogRegModels)
nmodels = nLogRegModels

###################################################################
##### Double cross-validation for modeling-process assessment #####				 
###################################################################

##### model assessment OUTER shell #####
fulldata.out = los.lr
k.out = 10 
n.out = dim(fulldata.out)[1]
#define the cross-validation splits 
groups.out = c(rep(1:k.out,floor(n.out/k.out))); if(floor(n.out/k.out) != (n.out/k.out)) groups.out = c(groups.out, 1:(n.out%%k.out))
set.seed(123, sample.kind = "Rounding")
cvgroups.out = sample(groups.out,n.out)  #orders randomly, with seed (8) 

# set up storage for predicted values from the double-cross-validation
allpredictedCV.out = rep(NA,n.out)
# set up storage to see what models are "best" on the inner loops
allbestmodels = rep(NA,k.out)

# loop through outer splits
for (j in 1:k.out)  {  #be careful not to re-use loop indices
  groupj.out = (cvgroups.out == j)
  traindata.out = los.lr[!groupj.out,]
  trainx.out = model.matrix(los.high~.,data=traindata.out)[,-(14)]
  trainy.out = traindata.out[,14]
  validdata.out = los.lr[groupj.out,]
  validx.out = model.matrix(los.high~.,data=validdata.out)[,-(14)]
  validy.out = validdata.out[,14]
  
  ### entire model-fitting process ###
  fulldata.in = as.data.frame(traindata.out)
  ###	:	:	:	:	:	:	:  ###
  ###########################
  ## Full modeling process ##
  ###########################
  
  # we begin setting up the model-fitting process to use notation that will be
  # useful later, "in"side a validation
  n.in = dim(fulldata.in)[1]
  x.in = model.matrix(los.high~.,data=fulldata.in)[,-(2)]
  y.in = fulldata.in[,1]
  # number folds and groups for (inner) cross-validation for model-selection
  k.in = 10 
  #produce list of group labels
  groups.in = c(rep(1:k.in,floor(n.in/k.in))); if(floor(n.in/k.in) != (n.in/k.in)) groups.in = c(groups.in, 1:(n.in%%k.in))
  cvgroups.in = sample(groups.in,n.in)  #orders randomly, with seed (8) 
  # table(cvgroups.in)  # check correct distribution
  allmodelCV.in = rep(NA,nmodels) #place-holder for results
  
    ##### cross-validation for model selection ##### 
  
  # # compute and store the CV(10) values
  allpredictedCV.in = rep(-1,nLogRegModels)
  
  for (m in 1:nLogRegModels) {
    glmmodel = glm(formula = allLogRegModels[[m]],data=fulldata.in,family=binomial)
    glmcv.in=cv.glm(fulldata.in,glmmodel,K=10)
    allpredictedCV.in[m]=glmcv.in$delta[1]
  }
  allpredictedCV.in
  
  allmodelCV.in[(1:nLogRegModels)] = allpredictedCV.in
  
  bestmodel.in = (1:nLogRegModels)[order(allmodelCV.in)[1]]  # actual selection
  # state which is best model and minimum CV(10) value
  bestmodel.in; min(allmodelCV.in)
  
  set.seed(123, sample.kind = "Rounding")
  #choose 1/3rd of data for training
  train = sample(1:282413,94138,replace=F)
  train.in = fulldata.in[train,]
  test.in = fulldata.in[-train,]
  
  ### finally, fit the best model to the full (available) data ###
  bestfit = glm(formula = allLogRegModels[[bestmodel.in]],data=fulldata.in,family=binomial)  # fit on all available data
  bestcoef = coef(bestfit)
 
   #############################
  ## End of modeling process ##
  #############################
  
  
  ###   :	:	:	:	:	:	:  ###
  ### resulting in bestmodel.in ###
  
  allbestmodels[j] = bestmodel.in
  
  allpredictedCV.out[groupj.out] = predict(bestfit,validdata.out,type = "response")

}

# for curiosity, we can see the models that were "best" on each of the inner splits
allbestmodels

#assessment
T1 = table(allpredictedCV.out >0.5,fulldata.out$los.high);T1

#Error Rate
cv1 = (T1[1,2] + T1[2,1])/n;cv1
#0.2151036


