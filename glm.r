library("lme4")
library("lmtest")
library("phia")
library("car")
library("emmeans")

# First model: burst waveforms vs power, without aperiodic activity subtraction
class_res_1_nfs <- read.csv("~/Codes/bebop_test/class_res_1_all__nfs_sel.csv", header=TRUE, stringsAsFactors=FALSE)

class_res_1_nfs$Feature<-as.factor(class_res_1_nfs$Feature)
class_res_1_nfs$Dataset<-as.factor(class_res_1_nfs$Dataset)
class_res_1_nfs$Subject<-as.factor(class_res_1_nfs$Subject)

model_1_nfs <- glmer(Accuracy ~ Feature + (1|Dataset/Subject), data=class_res_1_nfs, family=binomial(), weights=class_res_1_nfs$Trials)

summary(model_1_nfs)
Anova(model_1_nfs)
emmeans(model_1_nfs, pairwise ~ Feature)
contrast_results_1_nfs <- emmeans(model_1_nfs, pairwise ~ Feature)
eff_size(contrast_results_1_nfs, sigma=sigma(model_1_nfs), edf=Inf)


# Second model: burst waveforms  vs rest of burst representations, without aperiodic activity subtraction
class_res_2_nfs <- read.csv("~/Codes/bebop_test/class_res_2_all__nfs_sel.csv", header=TRUE, stringsAsFactors=FALSE)

class_res_2_nfs$Feature<-as.factor(class_res_2_nfs$Feature)
class_res_2_nfs$Dataset<-as.factor(class_res_2_nfs$Dataset)
class_res_2_nfs$Subject<-as.factor(class_res_2_nfs$Subject)

model_2_nfs <- glmer(Accuracy ~ Feature + (1|Dataset/Subject), data=class_res_2_nfs, family=binomial(), weights=class_res_2_nfs$Trials)

summary(model_2_nfs)
Anova(model_2_nfs)
emmeans(model_2_nfs, pairwise ~ Feature)
contrast_results_2_nfs <- emmeans(model_2_nfs, pairwise ~ Feature)
eff_size(contrast_results_2_nfs, sigma=sigma(model_2_nfs), edf=Inf)