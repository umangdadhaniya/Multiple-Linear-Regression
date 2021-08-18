# Load the Cars dataset
company <- read.csv(file.choose())
View(company)
#install.packages("fastDummies")
library(fastDummies)
company <- dummy_cols(company, 
                   select_columns = "state",remove_selected_columns = TRUE)
?fastDummies
colnames(company)[7] <- "state_New"

attach(company)

# Normal distribution
qqnorm(spend)
qqline(spend)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(company)

# Scatter plot
plot(spend, profit) # Plot relation ships between each X with Y
plot(adm, profit)

# Or make a combined plot
pairs(company)   # Scatter plot for all pairs of variables
plot(company)

cor(spend, profit)
cor(company) # correlation matrix

# The Linear Model of interest
model.company <- lm(profit ~ state_New  + adm + mspend + state_Florida + spend , data = company) # lm(Y ~ X)
summary(model.company)

model.carV <- lm(MPG ~ VOL)
summary(model.carV)

model.carW <- lm(MPG ~ WT)
summary(model.carW)

model.carVW <- lm(MPG ~ VOL + WT)
summary(model.carVW)

#### Scatter plot matrix with Correlations inserted in graph
#install.packages("GGally")
library(GGally)
ggpairs(company)


### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(company)

cor2pcor(cor(company))

# Diagnostic Plots
install.packages(car)
library(car)

plot(model.company)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.company, id.n = 5) # QQ plots of studentized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.company, id.n = 3) # Index Plots of the influence measures
influencePlot(model.company, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 77th observation
model.company1 <- lm(profit ~ spend + adm + mspend + state_California +  state_Florida + state_New , data = company[-47, ])
model.company1 <- lm(profit ~ spend + adm + mspend + state_California +  state_Florida + state_New , data = company[-48, ])
model.company1 <- lm(profit ~ spend + adm + mspend + state_California +  state_Florida + state_New , data = company[-50, ])
summary(model.company1)


### Variance Inflation Factors
vif(model.company)  # VIF is > 10 => collinearity

# Regression model to check R^2 on Independent variales
VIFWT <- lm(spend ~ adm + mspend + state_California + state_Florida + state_New)
VIFVOL <- lm(adm ~ spend + mspend + state_California + state_Florida + state_New)
VIFHP <- lm(mspend ~ spend + adm + state_California + state_Florida + state_New)
VIFSP <- lm(state_California ~ spend + adm + mspend + state_Florida + state_New)
VIFnp <- lm(state_Florida ~ spend + adm + mspend + state_California + state_New)
VIFnl <- lm(state_New ~ spend + adm + mspend + state_California + state_Florida)


summary(VIFWT)
summary(VIFVOL)
summary(VIFHP)
summary(VIFSP)
summary(VIFnp)
summary(VIFnl)


#### Added Variable Plots ######
avPlots(model.company, id.n = 2, id.cex = 0.8, col = "red")

# Linear Model without WT
model.final <- lm(profit ~ spend + mspend + state_California, data = company)
summary(model.final)

# Linear model without WT and influential observation
model.final1 <- lm(profit ~ spend + mspend + state_California, data = company[-50, ])
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
