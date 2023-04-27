rm(list=ls())
library(dplyr)
library(moments)
library(rio)
d=import("TelcoChurn.xlsx")

## Pre-processing

table(d$Churn)  ## This is unbalanced data set with yes=1869 and no=5174

colSums(is.na(d))  ## there are 11 null values in TotalCharges

d=d[complete.cases(d), ]  ## dropping those rows with null values

d$SeniorCitizen=ifelse(d$SeniorCitizen==1,'Yes','No')
table(d$SeniorCitizen)

table(d$MultipleLines)
d$MultipleLines=ifelse(d$MultipleLines=='Yes','Yes','No')
table(d$MultipleLines)

table(d$OnlineSecurity)
d$OnlineSecurity=ifelse(d$OnlineSecurity=='Yes','Yes','No')
table(d$OnlineSecurity)

table(d$OnlineBackup)
d$OnlineBackup=ifelse(d$OnlineBackup=='Yes','Yes','No')

table(d$DeviceProtection)
d$DeviceProtection=ifelse(d$DeviceProtection=='Yes','Yes','No')

table(d$TechSupport)
d$TechSupport=ifelse(d$TechSupport=='Yes','Yes','No')

table(d$StreamingMovies)
d$StreamingMovies=ifelse(d$StreamingMovies=='Yes','Yes','No')

table(d$StreamingTV)
d$StreamingTV=ifelse(d$StreamingTV=='Yes','Yes','No')

d$Churn=ifelse(d$Churn=='Yes',1,0)

## Factor variables
col <- c('gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService',
         'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
         'PaperlessBilling','PaymentMethod','Churn')
d[col] <- lapply(d[col], factor)
str(d)


## checking correlation

dummy=select(d,TotalCharges,MonthlyCharges,tenure)
cor(dummy)  ## High correlation between Total charges and tenure so dropping total charges
library(PerformanceAnalytics)
cor_df <- d[, sapply(d, is.numeric)]
chart.Correlation(cor_df)

## subsetting the data based on telephone,internet and both users

phone=subset(d,d$PhoneService=='Yes' & d$InternetService=='No')
internet=subset(d,d$InternetService!='No' & d$PhoneService=='No')
both=subset(d,d$InternetService!='No' & d$PhoneService!='No')

str(phone)
str(internet)
str(both)

table(phone$Churn)
table(internet$Churn)
table(both$Churn)

## ------------Only Phone churn Model:----------------------

set.seed(1024)

trainphone=sample(1:nrow(phone), size=round(0.75*nrow(phone)), replace=FALSE)
trainp=phone[trainphone,]
testp =phone[-trainphone,]
dim(trainp)
dim(testp)

## Model


p1=glm(Churn~SeniorCitizen+Dependents+MultipleLines+tenure*gender+Contract+PaymentMethod+MonthlyCharges+tenure+PaperlessBilling,family=binomial (link="logit"), data=trainp)
summary(p1)

testp_x=testp[ , c('gender','SeniorCitizen','Dependents','tenure','MultipleLines','Contract','PaymentMethod','MonthlyCharges','PaperlessBilling')]
predlogitphone <-predict(p1, newdata=testp_x, type="response")
predlogitphone <- ifelse(predlogitphone>0.15, 1, 0)
predlogitphone

## Confusion Matrix
cf1=table(testp$Churn, predlogitphone)

library(ROCR)
phonepred <- prediction(predlogitphone, testp$Churn)
prf <- performance(phonepred, measure="tpr", x.measure="fpr")
plot(prf) 

aucphone <- performance(phonepred, measure="auc")
aucphone <- aucphone@y.values[[1]]
aucphone 

## Assumptions:

## Independence:
library(lmtest)
dwtest(p1)

library(car)
vif(p1)



##-------------Only Internet Churn Model-----------------------

set.seed(1024)

traininternet=sample(1:nrow(internet), size=round(0.75*nrow(internet)), replace=FALSE)
traini=internet[traininternet,]
testi =internet[-traininternet,]
dim(traini)
dim(testi)

## Model----

i1=glm(Churn~SeniorCitizen+Dependents+tenure*gender+OnlineBackup
+OnlineSecurity+DeviceProtection+TechSupport+StreamingMovies
+Contract+PaperlessBilling+PaymentMethod+MonthlyCharges,family=binomial (link="logit"), data=traini)
summary(i1)


testi_x=testi[ , c('gender','SeniorCitizen','Partner','Dependents','tenure','OnlineBackup',
                   'OnlineSecurity','DeviceProtection','TechSupport','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges')]
predlogitinternet <-predict(i1, newdata=testi_x, type="response")
predlogitinternet <- ifelse(predlogitinternet>0.3, 1, 0)
table(predlogitinternet)

## Confusion Matrix
cf2=table(testi$Churn, predlogitinternet)

## AUC and ROC

library(ROCR)
internetpred <- prediction(predlogitinternet, testi$Churn)
pri <- performance(internetpred, measure="tpr", x.measure="fpr")
plot(pri) 

aucinternet <- performance(internetpred, measure="auc")
aucinternet <- aucinternet@y.values[[1]]
aucinternet 

## Assumptions:

## Independence:
library(lmtest)
dwtest(i1)

library(car)
vif(i1)


##-------Both Internet and Phone Churn Models---------
set.seed(1024)

trainboth=sample(1:nrow(both), size=round(0.75*nrow(both)), replace=FALSE)
trainb=both[trainboth,]
testb =both[-trainboth,]
dim(trainb)
dim(testb)

## ---Model-----

## Model----

b1=glm(Churn~gender*tenure+SeniorCitizen+Dependents+MultipleLines+OnlineBackup+OnlineSecurity+
         DeviceProtection+TechSupport+StreamingMovies+Contract+PaperlessBilling+
         PaymentMethod+MonthlyCharges,family=binomial (link="logit"), data=trainb)
summary(b1)


testb_x=testb[ , c('gender','SeniorCitizen','Partner','Dependents','tenure','MultipleLines','OnlineBackup',
                   'OnlineSecurity','DeviceProtection','TechSupport','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges')]
predlogitboth <-predict(b1, newdata=testb_x, type="response")
predlogitboth <- ifelse(predlogitboth>0.3, 1, 0)
table(predlogitboth)

## Confusion Matrix
cf3=table(testb$Churn, predlogitboth)

## AUC and ROC

library(ROCR)
bothpred <- prediction(predlogitboth, testb$Churn)
bri <- performance(bothpred, measure="tpr", x.measure="fpr")
plot(bri) 

aucboth <- performance(bothpred, measure="auc")
aucboth <- aucboth@y.values[[1]]
aucboth 

## Assumptions:

## Independence:
library(lmtest)
dwtest(b1)

library(car)
vif(b1)


###--------------Stargazer output-------------------
library(stargazer)
stargazer(p1,i1,b1,type='text',single.row=TRUE)

##------Interpretations in ODDS ratio--------
e1=exp(p1$coef)
exp(cbind(OddsRatio = coef(b1), confint(b1)))
e2=exp(i1$coef)
e3=exp(b1$coef)



