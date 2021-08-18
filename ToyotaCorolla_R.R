# Load the Cars dataset
tc <- read.csv(file.choose())
View(tc)
tc <- tc[ -c(1,2,5,6,8,10,11,12,15,19:39) ]

View(tc)
attach(tc)

# Normal distribution
qqnorm(Age_08_04)
qqline(Age_08_04)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(tc)

# Scatter plot
plot(KM, Price) # Plot relation ships between each X with Y
plot(HP, Price)

# Or make a combined plot
pairs(tc)   # Scatter plot for all pairs of variables
plot(tc)

cor(KM, Price)
cor(tc) # correlation matrix

# The Linear Model of interest
model.tc <- lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight , data = tc) # lm(Y ~ X)
summary(model.tc)

model.tcV <- lm(Price ~ cc)
summary(model.tcV)

model.tcW <- lm(Price ~ Doors)
summary(model.tcW)

model.tcVW <- lm(Price ~ cc + Doors)
summary(model.tcVW)

#### Scatter plot matrix with Correlations inserted in graph
#install.packages("GGally")
library(GGally)
ggpairs(tc)


### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(tc)

cor2pcor(cor(tc))

# Diagnostic Plots
install.packages(car)
library(car)

plot(model.tc)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.tc, id.n = 5) # QQ plots of studentized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.tc, id.n = 3) # Index Plots of the influence measures
influencePlot(model.tc, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 77th observation
model.tc1 <- lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight , data = tc[-222, ])
model.tc1 <- lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight , data = tc[-81, ])
summary(model.tc1)


### Variance Inflation Factors
vif(model.tc)  # VIF is > 10 => collinearity

# Regression model to check R^2 on Independent variales
#VIFWT <- lm(WT ~ VOL + HP + SP)
#VIFVOL <- lm(VOL ~ WT + HP + SP)
#VIFHP <- lm(HP ~ VOL + WT + SP)
#VIFSP <- lm(SP ~ VOL + HP + WT)

#summary(VIFWT)
#summary(VIFVOL)
#summary(VIFHP)
#summary(VIFSP)

# VIF of SP
#1/(1-0.95)

#### Added Variable Plots ######
avPlots(model.tc, id.n = 2, id.cex = 0.8, col = "red")

# Linear Model without WT
model.final <- lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight , data = tc)
summary(model.final)

# Linear model without WT and influential observation
model.final1 <-  lm(Price ~ Age_08_04 + KM + HP + Gears + Quarterly_Tax + Weight , data = tc[-222, ])
model.final1 <- lm(Price ~ Age_08_04 + KM + HP  + Gears + Quarterly_Tax + Weight , data = tc[-81, ])
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
