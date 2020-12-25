########################################################################################
#               Assignment 2 : Multivariate Analysis 
#               Aditya Ghosh (M.Stat 1st Year student, I.S.I. Kolkata)
########################################################################################

#---------------------------------------------------------------------------------------
#                 Problem 1: Applying lda, qda to a real-life dataset
#---------------------------------------------------------------------------------------

library(MASS)      # required for the functions lda, qda.

# We use a diabetes dataset that measures glucose levels in the blood after fasting (glufast), after a test condition (glutest) as well as steady state plasma glucose (steady) and steady state (insulin) for diabetes, the sixth variable (Group) denotes the states of diabetes mellitus (levels: 1, 2, 3).
diabetes = read.table(url("http://bios221.stanford.edu/data/diabetes.txt"), header = TRUE, row.names = 1)    
head(diabetes)
table(diabetes$Group)

diabetes$Group = 1 + diabetes$Group # just for a better coloring
pairs(diabetes[,2:5], col = diabetes$Group, pch = 19)

attach(diabetes)

plot(glutest ~ insulin, pch = 20, col = Group) # observe that the classes are well-separated
legend("topleft", legend = c("1", "2", "3"), col = 4:2, title = "diabetes level", pch = 19, horiz = T)

table(Group)
set.seed(3)
train = sort(c(sample(which(Group == 2), 16), sample(which(Group == 3), 18), sample(which(Group == 4), 38)))   # selects the training data in a stratified manner

lda.fit = lda(Group ~ insulin + glutest, data = diabetes, subset = train)
train.lda = predict(lda.fit, diabetes[train,])
test.lda = predict(lda.fit, diabetes[-train,])
mean(train.lda$class != Group[train]) # proportions of misclassification for LDA on train data
mean(test.lda$class != Group[-train]) # proportions of misclassification for LDA on test data

qda.fit = qda(Group ~ insulin + glutest, data = diabetes, subset = train)
train.qda = predict(qda.fit, diabetes[train,])
test.qda = predict(qda.fit, diabetes[-train,])
mean(train.qda$class != Group[train]) # proportions of misclassification for QDA on train data
mean(test.qda$class != Group[-train]) # proportions of misclassification for QDA on test data

# A trick to illustrate the classification regions: 
# Draw points uniformly from the plot window, 
# classify them using the same classification rules, 
# and plot them with a smaller point size! 
# We apply this trick below.

par(mar = c(3, 4.2, 3, 2))
layout(matrix(1:4, nrow = 2))
u = runif(70000, min(insulin)-20, max(insulin)+20)
v = runif(70000, min(glutest)-30, max(glutest)+30)
X = data.frame(cbind(u, v))
colnames(X) =  c("insulin", "glutest")

cols =  as.numeric(predict(lda.fit, X)$class) + 1
with(diabetes[train, ], plot(glutest ~ insulin, xlim = range(insulin), ylim = range(glutest), pch = 20, col = Group))
title(main = "LDA on Train data", line = 1)
points(u, v, pch = 20, cex = 0.01, col = cols)

with(diabetes[-train, ], plot(glutest ~ insulin, xlim = range(insulin), ylim = range(glutest), pch = 20, col = Group))
title(main = "LDA on Test data", line = 1)
points(u, v, pch = 20, cex = 0.01, col = cols)

cols = as.numeric(predict(qda.fit, X)$class) + 1
with(diabetes[train, ], plot(glutest ~ insulin, xlim = range(insulin), ylim = range(glutest), pch = 20, col = Group))
title(main = "QDA on Train data", line = 1)
points(u, v, pch = 20, cex = 0.01, col = cols)

with(diabetes[-train, ], plot(glutest ~ insulin, xlim = range(insulin), ylim = range(glutest), pch = 20, col = Group))
title(main = "QDA on Test data", line = 1)
points(u, v, pch = 20, cex = 0.01, col = cols)

Discriminant.Analysis<-function(){
	train = sort(c(sample(which(Group == 2), 16), sample(which(Group == 3), 18), sample(which(Group == 4), 38)))   # selecting train data in a stratified manner

	lda.fit = lda(Group ~ insulin + glutest, data = diabetes, subset = train)
	train.lda = predict(lda.fit, diabetes[train,])
	test.lda = predict(lda.fit, diabetes[-train,])
	l1 <- mean(train.lda$class != Group[train])
	l2 <- mean(test.lda$class != Group[-train])

	qda.fit = qda(Group ~ insulin + glutest, data = diabetes, subset = train)
	train.qda = predict(qda.fit, diabetes[train,])
	test.qda = predict(qda.fit, diabetes[-train,])
	q1 <- mean(train.qda$class != Group[train])
	q2 <- mean(test.qda$class != Group[-train])

	Mis <- matrix(c(l1, l2, q1, q2), ncol = 2, byrow = T)
	rownames(Mis) = c("LDA", "QDA"); colnames(Mis) = c("Train", "Test")
	return(Mis) # returns the proportions of misclassification
}

set.seed(1)
Misclassification = matrix(0, nrow = 2, ncol = 2)
for(i in 1:10) Misclassification = Misclassification + Discriminant.Analysis()
Misclassification/10

#---------------------------------------------------------------------------------------
#                   Problem 2: Applying lda, qda to simulated datasets
#---------------------------------------------------------------------------------------

library(MASS)      # required for the functions lda, qda, mvrnorm.

# Our data consists of n/3 observations from each of the populations N_(mu_i, Sigma_i), i = 1, 2, 3. 
# We select 10% of the data randomly (in a stratified manner) as the training data. 
# For LDA and QDA we estimate the parameters from the train data only. 
# The following function Classify.Simulated simulates the data, applies the 3 classification rules, and reports the proportions of misclassification. It also plots the classification region in a clever manner.

Classify.Simulated <- function(Sigma1, Sigma2, Sigma3, mu1 = c(1, 1), mu2 = c(1, 7), mu3 = c(7, 1), n = 1500){
	X1 = mvrnorm(n = n/3, mu = mu1, Sigma = Sigma1)
	X2 = mvrnorm(n = n/3, mu = mu2, Sigma = Sigma2) 
	X3 = mvrnorm(n = n/3, mu = mu3, Sigma = Sigma3) 
	Y = data.frame(rbind(X1, X2, X3), rep(1:3, each = n/3))
	names(Y) = c("Y1", "Y2", "Group")

	# Optimal classification rule: (common function for OPT rule, LDA, and QDA)
	Classify<-function(x, mu, Sinvs, log.det.S){
		d = NULL
		for(i in 1:3)
			d = c(d, -1/2 * log.det.S[i] -1/2 * t(x) %*% Sinvs[,,i] %*% x + t(x) %*% Sinvs[,,i] %*% mu[,i] - 1/2 * t(mu[,i]) %*% Sinvs[,,i] %*% mu[,i])   
			# Here we have ignored pi_i's because they are taken to be same both for the poplulation, the sample and also for the training data. Had the sample sizes been unequal, we could use sample proportions to estimate the pi_i's and add the term log(pi_i) in d. 
		return(list("group" = which.max(d), "discfun" = d[which.max(d)]))
	}

	# Selecting the training data:
	train = sort(c(sample(which(Y$Group == 1), n/30), sample(which(Y$Group == 2), n/30), sample(which(Y$Group == 3), n/30)))  # selects indices of train data in a stratified manner
with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 20, col = factor(Group)))

	# Some auxiliary things for ease of computation:
	mu = cbind(mu1, mu2, mu3)
	Sinvs = array(data = c(solve(Sigma1), solve(Sigma2), solve(Sigma3)), dim = c(2, 2, 3))
	log.det.S = - apply(Sinvs, 3, function(x) log(det(x)))

	# Sample estimates:
	Xbars = with(Y[train, ], rbind(tapply(Y1, Group, mean), tapply(Y2, Group, mean)))
	Sinvs.sample = Sinvs 
	Svar = matrix(0, nrow = 2, ncol = 2)
	for(i in 1:3){
		Sinvs.sample[,,i] = solve(Stemp <- var(subset(Y[train, ], Group == i)[,1:2]))  # required for qda
		Svar =  Svar + Stemp
	}
	common.Sinv = replicate(3, solve(Svar/3))   # required for lda
	log.dets.sample = - apply(Sinvs.sample, 3, function(x) log(det(x)))
	
	# Applying the optimal rule:
	train.opt <- apply(Y[train, 1:2], 1, function(x) Classify(as.numeric(x[1:2]), mu, Sinvs, log.det.S)$group)
	test.opt <- apply(Y[-train, 1:2], 1, function(x) Classify(as.numeric(x[1:2]), mu, Sinvs, log.det.S)$group)
	
	# Applying the LDA:
	train.lda <- apply(Y[train, 1:2], 1, function(x) Classify(as.numeric(x[1:2]), Xbars, common.Sinv, rep(0, 3))$group)
	test.lda <- apply(Y[-train, 1:2], 1, function(x) Classify(as.numeric(x[1:2]), Xbars, common.Sinv, rep(0, 3))$group)
	
	# Applying the QDA:
	train.qda <- apply(Y[train, 1:2], 1, function(x) Classify(as.numeric(x[1:2]), Xbars, Sinvs.sample, log.dets.sample)$group)
	test.qda <- apply(Y[-train, 1:2], 1, function(x) Classify(as.numeric(x[1:2]), Xbars, Sinvs.sample, log.dets.sample)$group)

	Mis <- matrix(c(mean(train.opt != Y[train, "Group"]), mean(test.opt != Y[-train, "Group"]),
	mean(train.lda != Y[train, "Group"]), mean(test.lda != Y[-train, "Group"]),
	mean(train.qda != Y[train, "Group"]), mean(test.qda != Y[-train, "Group"])), ncol = 2, byrow = T)
	rownames(Mis) = c("OPT", "LDA", "QDA")
	colnames(Mis) = c("Train", "Test")
	
	# Illustrating the classification regions using the same trick as above: 
	par(mar = c(4.5, 4.2, 3, 2), family = "System Font")
	layout(matrix(1:4, nrow = 2))
	u = runif(50000, min(Y[, 1])-0.5, max(Y[, 1])+0.5)
	v = runif(50000, min(Y[, 2])-0.5, max(Y[, 2])+0.5)

	cols = apply(cbind(u, v), 1, function(x) Classify(as.numeric(x[1:2]), Xbars, common.Sinv, rep(0, 3))$group)
	with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 20, col = factor(Group)))
	title(main = "LDA on Train data", line = 1)
	points(u, v, pch = 20, cex = 0.01, col = cols)

	with(Y[-train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 20, cex = 0.5, col = factor(Group)))
	title(main = "LDA on Test data", line = 1)
	points(u, v, pch = 20, cex = 0.01, col = cols)

	cols = apply(cbind(u, v), 1, function(x) Classify(as.numeric(x[1:2]), Xbars, Sinvs.sample, log.dets.sample)$group)
	with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 20, col = factor(Group)))
	title(main = "QDA on Train data", line = 1)
	points(u, v, pch = 20, cex = 0.01, col = cols)

	with(Y[-train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 20, cex = 0.5, col = factor(Group)))
	title(main = "QDA on Test data", line = 1)
	points(u, v, pch = 20, cex = 0.01, col = cols)


	# # Matching outputs of our software with that of the R functions lda and qda:
	
	# lda.fit <- lda(Group ~ Y1 + Y2, data = Y, subset = train)
	# train.lda.R = predict(lda.fit, Y[train,])$class
	# test.lda.R = predict(lda.fit, Y[-train,])$class
	# l1 <- mean(train.lda.R != Y[train, "Group"])
	# l2 <- mean(test.lda.R != Y[-train, "Group"])

	# qda.fit = qda(Group ~ Y1 + Y2, data = Y, subset = train)
	# train.qda.R = predict(qda.fit, Y[train,])$class
	# test.qda.R = predict(qda.fit, Y[-train,])$class
	# q1 <- mean(train.qda.R != Y[train, "Group"])
	# q2 <- mean(test.qda.R != Y[-train, "Group"])
	
	# Mis.R <- matrix(c(l1, l2, q1, q2), ncol = 2, byrow = T)
	# rownames(Mis.R) = c("LDA", "QDA")
	# colnames(Mis.R) = c("Train", "Test")
	# print(Mis.R)  # proportions of misclassification
	# print(dim(Y[train, ]))
	# print(dim(Y[-train, ]))

	# We find that our outputs matches with that of the standard R functions lda and qda.

	return(Mis)
}

## 2 part (i)
mu1 = c(0, 0); mu2 = c(2, -2); mu3 = c(4, 4)
set.seed(1)
Sigma = matrix(c(2, 1, 1, 2), nrow = 2)
Classify.Simulated(Sigma1 = Sigma, Sigma2 = Sigma, Sigma3 = Sigma, mu1, mu2, mu3)

## 2 part (ii)
mu1 = c(3, 3); mu2 = c(-1, 6); mu3 = c(6, -1)
set.seed(1)
Classify.Simulated(Sigma1 = matrix(c(4, 2, 2, 9), nrow = 2), Sigma2 = matrix(c(1, -1, -1, 4), nrow = 2), Sigma3 = matrix(c(4, 1, 1, 1), nrow = 2), mu1, mu2, mu3)

# 2 part (iii)  We take each of the 3 classes to be distributed as a mixture of multivariate normals.
set.seed(1)
n = 1500
Sigma = matrix(c(1, 0, 0, 9), nrow = 2)
mu1 = c(0, 2); mu2 = c(5, 0); mu3 = c(10, 2)

u1 = runif(n/3, 0, 1)
X11 = mvrnorm(n = n/3, mu = mu1, Sigma = Sigma) 
X12 = mvrnorm(n = n/3, mu = mu1 + c(15, 0), Sigma = Sigma) 
X1 = as.numeric(u1 > 1/2) * X11 + as.numeric(u1 < 1/2) * X12

u2 = runif(n/3, 0, 1)
X21 = mvrnorm(n = n/3, mu = mu2, Sigma = Sigma) 
X22 = mvrnorm(n = n/3, mu = -mu2, Sigma = Sigma) 
X2 = as.numeric(u2 > 1/2) * X21 + as.numeric(u2 < 1/2) * X22

u3 = runif(n/3, 0, 1)
X31 = mvrnorm(n = n/3, mu = mu3, Sigma = Sigma) 
X32 = mvrnorm(n = n/3, mu = mu3 + c(10, 0), Sigma = Sigma) 
X3 = as.numeric(u3 > 1/2) * X31 + as.numeric(u3 < 1/2) * X32

Y = data.frame(rbind(X1, X2, X3), rep(1:3, each = n/3))
names(Y) = c("Y1", "Y2", "Group")

# Since the optimal rule implemented in Classify.Simulated function is suited only for data generated from multivariate normal, we have to proceed differently in this part. 
Sigma.inv = solve(Sigma)
my.dmvn<-function(x, mu) 
	return(exp(-1/2 * t(x - mu) %*% Sigma.inv %*% (x - mu))) 
# Note that some terms in the log-likelihood are omitted above, because they do not matter for the comparison purpose since the Sigmas of the multivariate normals are all same here.
# The following function calculates the optimal rule for this problem.
opt.rule<-function(x){
	f1 <- my.dmvn(x, mu1)/2 + my.dmvn(x, mu1 + c(15, 0))/2
	f2 <- my.dmvn(x, mu2)/2 + my.dmvn(x, -mu2)/2
	f3 <- my.dmvn(x, mu3)/2 + my.dmvn(x, mu3 + c(10, 0))/2
	return(which.max(c(f1, f2, f3)))
}
# Also, since we observed that for LDA and QDA, our function Classify.Simulated does produce the same output as the functions lda, qda in R, we can use either of them here, for the purpose of doing LDA and QDA.

# Selecting the training data:
set.seed(3)
train = sort(c(sample(which(Y$Group == 1), n/30), sample(which(Y$Group == 2), n/30), sample(which(Y$Group == 3), n/30)))  # selecting indices of train data in a stratified manner
with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 20, col = factor(Group)))

# Applying the optimal rule:
train.opt <- apply(Y[train, 1:2], 1, function(x) opt.rule(as.numeric(x[1:2])))
test.opt <- apply(Y[-train, 1:2], 1, function(x) opt.rule(as.numeric(x[1:2])))
o1 <- mean(train.opt != Y[train, "Group"])
o2 <- mean(test.opt != Y[-train, "Group"])

# Applying the LDA:
lda.fit <- lda(Group ~ Y1 + Y2, data = Y, subset = train)
train.lda.R = predict(lda.fit, Y[train,])$class
test.lda.R = predict(lda.fit, Y[-train,])$class
l1 <- mean(train.lda.R != Y[train, "Group"])
l2 <- mean(test.lda.R != Y[-train, "Group"])

# Applying the QDA:
qda.fit = qda(Group ~ Y1 + Y2, data = Y, subset = train)
train.qda.R = predict(qda.fit, Y[train,])$class
test.qda.R = predict(qda.fit, Y[-train,])$class
q1 <- mean(train.qda.R != Y[train, "Group"])
q2 <- mean(test.qda.R != Y[-train, "Group"])

# proportions of misclassification
Miss <- matrix(c(o1, o2, l1, l2, q1, q2), ncol = 2, byrow = T)
rownames(Miss) = c("OPT", "LDA", "QDA")
colnames(Miss) = c("Train", "Test")
print(Miss)

# Illustrating the classification regions using the same trick as above: 
par(mar = c(4.5, 4.2, 3, 2), family = "System Font")
layout(matrix(1:6, nrow = 2))
u = runif(70000, min(Y[, 1])-0.5, max(Y[, 1])+0.5)
v = runif(70000, min(Y[, 2])-0.5, max(Y[, 2])+0.5)
X = data.frame(cbind(u, v))
colnames(X) =  c("Y1", "Y2")

cols = as.numeric(predict(lda.fit, X)$class)
with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 19, cex = 0.5, col = factor(Group)))
title(main = "LDA on Train data", line = 1, cex.main = 1.5)
points(u, v, pch = 20, cex = 0.01, col = cols)

with(Y[-train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 19, cex = 0.5, col = factor(Group)))
title(main = "LDA on Test data", line = 1, cex.main = 1.5)
points(u, v, pch = 20, cex = 0.01, col = cols)

cols = as.numeric(predict(qda.fit, X)$class)
with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 19, cex = 0.5, col = factor(Group)))
title(main = "QDA on Train data", line = 1, cex.main = 1.5)
points(u, v, pch = 20, cex = 0.01, col = cols)

with(Y[-train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 19, cex = 0.5, col = factor(Group)))
title(main = "QDA on Test data", line = 1, cex.main = 1.5)
points(u, v, pch = 20, cex = 0.01, col = cols)

cols = apply(cbind(u, v), 1, opt.rule)
with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 19, cex = 0.5, col = factor(Group)))
title(main = "Optimal rule on Train data", line = 1, cex.main = 1.5)
points(u, v, pch = 20, cex = 0.01, col = cols)

with(Y[-train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 19, cex = 0.5, col = factor(Group)))
title(main = "Optimal rule on Test data", line = 1, cex.main = 1.5)
points(u, v, pch = 20, cex = 0.01, col = cols)


########################################################################################

set.seed(3)
Sigma = matrix(c(1, 0, 0, 1), nrow = 2)
n = 3000
x = runif(n, -2, 2); y = runif(n, -2, 2)
Y = cbind(x, y)
color = rep(2, n)
color[y > x^3 - sin(x) - sin(pi*x) + 1] = 1
color[y < x^3 - sin(x) - sin(pi*x) - 1] = 3

Y = data.frame(cbind(Y, color))

names(Y) = c("Y1", "Y2", "Group")



# Selecting the training data:
train = sample(c(T, F), n/5, replace = T)
# train = sort(c(sample(which(Y$Group == 1), n/5), sample(which(Y$Group == 2), n/5), sample(which(Y$Group == 3), n/5)))  # selecting indices of train data in a stratified manner
with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 20, col = factor(Group)))

OPT<-function(x){
	if(x[2] > x[1]^3 - sin(x[1]) - sin(pi*x[1]) + 1) return(1)
	if(x[2] < x[1]^3 - sin(x[1]) - sin(pi*x[1]) - 1) return(3)
	else return(2)
}

train.opt <- apply(Y[train, 1:2], 1, function(x) OPT(as.numeric(x[1:2])))
test.opt <- apply(Y[-train, 1:2], 1, function(x) OPT(as.numeric(x[1:2])))
o1 <- mean(train.opt != Y[train, "Group"])
o2 <- mean(test.opt != Y[-train, "Group"])

lda.fit <- lda(Group ~ Y1 + Y2, data = Y, subset = train)
train.lda.R = predict(lda.fit, Y[train,])$class
test.lda.R = predict(lda.fit, Y[-train,])$class
l1 <- mean(train.lda.R != Y[train, "Group"])
l2 <- mean(test.lda.R != Y[-train, "Group"])

qda.fit = qda(Group ~ Y1 + Y2, data = Y, subset = train)
train.qda.R = predict(qda.fit, Y[train,])$class
test.qda.R = predict(qda.fit, Y[-train,])$class
q1 <- mean(train.qda.R != Y[train, "Group"])
q2 <- mean(test.qda.R != Y[-train, "Group"])

# proportions of misclassification
Miss <- matrix(c(o1, o2, l1, l2, q1, q2), ncol = 2, byrow = T)
rownames(Miss) = c("OPT", "LDA", "QDA")
colnames(Miss) = c("Train", "Test")
Miss



par(mfrow = c(1, 3), family = "System Font")
u = runif(10^5, -2, 2)
v = runif(10^5, -2, 2)
X = data.frame(cbind(u, v))
colnames(X) =  c("Y1", "Y2")

cols = as.numeric(predict(lda.fit, X)$class)
with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 19, cex = 0.5, col = factor(Group)))
title(main = "LDA on Train data", line = 1)
points(u, v, pch = 20, cex = 0.01, col = cols)

cols = as.numeric(predict(qda.fit, X)$class)
with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 19, cex = 0.5, col = factor(Group)))
title(main = "QDA on Train data", line = 1)
points(u, v, pch = 20, cex = 0.01, col = cols)


cols = apply(cbind(u, v), 1, OPT)
with(Y[train, ], plot(Y2 ~ Y1, xlim = range(Y1), ylim = range(Y2), pch = 19, cex = 0.5, col = factor(Group)))
title(main = "OPT on Train data", line = 1)
points(u, v, pch = 20, cex = 0.01, col = cols)
