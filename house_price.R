setwd("C://Users/Zixin Zeng/PycharmProjects/webdatamining/house")
library(readxl)
data<-read_excel("processed_data.xlsx")
names(data)<-c("link","name_of_website","nbhood","address","city","district","rent_price","shi","wei","ting","area","dir","floor","total_floor","balcony","uploader","bed","closet","sofa","TV","fridge","washer","cooler","water","wifi","gas","heater","lng","lat","dist_school","num_school","dist_hosp","num_hosp")
library(lme4)
data$city<-factor(data$city)
data$district<-factor(data$district)
data$dir<-factor(data$dir)
data$balcony<-factor(data$balcony)
data$uploader<-factor(data$uploader)
data$bed<-factor(data$bed)
data$closet<-factor(data$closet)
data$sofa<-factor(data$sofa)
data$TV<-factor(data$TV)
data$fridge<-factor(data$fridge)
data$washer<-factor(data$washer)
data$cooler<-factor(data$cooler)
data$water<-factor(data$water)
data$wifi<-factor(data$wifi)
data$gas<-factor(data$gas)
data$heater<-factor(data$heater)
data$floor<-factor(data$floor)
data$dist_school[data$dist_school==-1]<-4000
data$dist_hosp[data$dist_hosp==-1]<-5000
data$dist_school<-rescale(data$dist_school)
data$dist_hosp<-rescale(data$dist_hosp)
library(corrplot)
library(gplots)
corrplot.mixed(corr=cor(data[,c(7:11,30:33)],use="complete.obs"),upper="ellipse",tl.pos="lt",upper.col=colorpanel(50,"red","gray60","blue4"))

#2 packages
library('MuMIn')
library("lmerTest")
#model
fit <- lmer(formula=rent_price ~ shi+wei+ting+area+dir+uploader+balcony+bed+closet+sofa+TV+fridge+washer+cooler+water+wifi+gas+heater+dist_school+num_school+dist_hosp+num_hosp + (1|city/district) +floor*total_floor,data=data)
#evaluation
r.squaredGLMM(fit)#see R2
anova(fit)
summary(fit)
library(car)
vif(fit)


