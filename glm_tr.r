library("lme4")
library("car")
library("emmeans")


# Selection of time window decoding method.
classification_mode = "incremental" # "incremental", "sliding"
if (classification_mode == "incremental") {
cl_str = "tr"
} else if (classification_mode == "sliding") {
cl_str = "sl"
}


# Model 1: max decoding accuracy per decoding feature.
time_res_max_dec_score <- read.csv(sprintf("~/Codes/time_res_%s_class_res_nfs_max_score.csv", cl_str), header=TRUE, stringsAsFactors=FALSE)

time_res_max_dec_score$Feature<-as.factor(time_res_max_dec_score$Feature)
time_res_max_dec_score$Dataset<-as.factor(time_res_max_dec_score$Dataset)
time_res_max_dec_score$Subject<-as.factor(time_res_max_dec_score$Subject)

model_score <- lmer(Accuracy ~ Feature + (1|Dataset/Subject), data=time_res_max_dec_score, weights=time_res_max_dec_score$Trials, control=lmerControl(optimizer="bobyqa"))

summary(model_score)
Anova(model_score)
emmeans(model_score, pairwise ~ Feature)


# Model 2: time to max decoding accuracy per decoding feature.
time_res_max_dec_time <- read.csv(sprintf("~/Codes/time_res_%s_class_res_nfs_max_time.csv", cl_str), header=TRUE, stringsAsFactors=FALSE)

time_res_max_dec_time$Feature<-as.factor(time_res_max_dec_time$Feature)
time_res_max_dec_time$Dataset<-as.factor(time_res_max_dec_time$Dataset)
time_res_max_dec_time$Subject<-as.factor(time_res_max_dec_time$Subject)

model_time <- lmer(Time ~ Feature + (1|Dataset/Subject), data=time_res_max_dec_time, weights=time_res_max_dec_time$Trials, control=lmerControl(optimizer="bobyqa"))

summary(model_time)
Anova(model_time)
emmeans(model_time, pairwise ~ Feature)


# Model 3: MAX ITR across all decoding features.
time_res_max_dec_itr <- read.csv(sprintf("~/Codes/time_res_%s_class_res_nfs_max_itr.csv", cl_str), header=TRUE, stringsAsFactors=FALSE)

time_res_max_dec_itr$Feature<-as.factor(time_res_max_dec_itr$Feature)
time_res_max_dec_itr$Dataset<-as.factor(time_res_max_dec_itr$Dataset)
time_res_max_dec_itr$Subject<-as.factor(time_res_max_dec_itr$Subject)

model_itr <- lmer(Accuracy ~ Feature + (1|Dataset/Subject), data=time_res_max_dec_itr, weights=time_res_max_dec_itr$Trials, control=lmerControl(optimizer="bobyqa"))

summary(model_itr)
Anova(model_itr)
emmeans(model_itr, pairwise ~ Feature)


# Model 4: time to max ITR per decoding feature.
time_res_max_dec_itr_time <- read.csv(sprintf("~/Codes/time_res_%s_class_res_nfs_max_itr_time.csv", cl_str), header=TRUE, stringsAsFactors=FALSE)

time_res_max_dec_itr_time$Feature<-as.factor(time_res_max_dec_itr_time$Feature)
time_res_max_dec_itr_time$Dataset<-as.factor(time_res_max_dec_itr_time$Dataset)
time_res_max_dec_itr_time$Subject<-as.factor(time_res_max_dec_itr_time$Subject)

model_itr_time <- lmer(Time ~ Feature + (1|Dataset/Subject), data=time_res_max_dec_itr_time, weights=time_res_max_dec_itr_time$Trials, control=lmerControl(optimizer="bobyqa"))

summary(model_itr_time)
Anova(model_itr_time)
emmeans(model_itr_time, pairwise ~ Feature)