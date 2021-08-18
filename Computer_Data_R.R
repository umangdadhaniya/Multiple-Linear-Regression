# Load the Cars dataset
md <- read.csv(file.choose())
View(md)
md <- md[ -c(1) ]
?dummy_columns
library(fastDummies)
md <- dummy_cols(md,select_columns = "cd",remove_selected_columns = TRUE, remove_first_dummy = TRUE)
md <- dummy_cols(md,select_columns = "multi",remove_selected_columns = TRUE, remove_first_dummy = TRUE)
md <- dummy_cols(md,select_columns = "premium",remove_selected_columns = TRUE, remove_first_dummy = TRUE)
View(md)
attach(md)

# Normal distribution
qqnorm(speed)
qqline(speed)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(md)

# Scatter plot
plot(speed, price) # Plot relation ships between each X with Y
plot(hd, price)

# Or make a combined plot
pairs(md)   # Scatter plot for all pairs of variables
plot(md)

cor(speed, price)
cor(md) # correlation matrix

# The Linear Model of interest
model.md <- lm(price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes, data = md) # lm(Y ~ X)
summary(model.md)

#model.carV <- lm(MPG ~ VOL)
#summary(model.carV)

#model.carW <- lm(MPG ~ WT)
#summary(model.carW)

#model.carVW <- lm(MPG ~ VOL + WT)
#summary(model.carVW)

#### Scatter plot matrix with Correlations inserted in graph
#install.packages("GGally")
library(GGally)
ggpairs(md)


### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(md)

cor2pcor(cor(md))

# Diagnostic Plots
install.packages(car)
library(car)

plot(model.md)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.md, id.n = 5) # QQ plots of studentized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.md, id.n = 3) # Index Plots of the influence measures
influencePlot(model.md, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 77th observation
model.md1 <- lm(price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes, data = md[-1701, ])
model.md1 <- lm(price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes, data = md[-1441, ])
summary(model.md1)


### Variance Inflation Factors
vif(model.md)  # VIF is > 10 => collinearity

# Regression model to check R^2 on Independent variales
VIFWT <- lm(WT ~ VOL + HP + SP)
VIFVOL <- lm(VOL ~ WT + HP + SP)
VIFHP <- lm(HP ~ VOL + WT + SP)
VIFSP <- lm(SP ~ VOL + HP + WT)

summary(VIFWT)
summary(VIFVOL)
summary(VIFHP)
summary(VIFSP)

# VIF of SP
1/(1-0.95)

#### Added Variable Plots ######
avPlots(model.md, id.n = 2, id.cex = 0.8, col = "red")

# Linear Model without WT
model.final <- lm(price ~ speed + hd + ram + screen + ads + trend + premium_yes, data = md)
summary(model.final)

# Linear model without WT and influential observation
model.final1 <- lm(price ~ speed + hd + ram + screen + ads + trend + premium_yes, data = md[-1441, ])
model.final1 <- lm(price ~ speed + hd + ram + screen + ads + trend + premium_yes, data = md[-1701, ])
summary(model.final1)

# Added Variable Plots
avPlots(model.final1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.final1)

# Evaluation Model Assumptions
plot(model.final1)
plot(model.final1$fitted.values, model.final1$residuals)


qqnorm(model.final1$residuals)
qqline(model.final1$residuals)
  