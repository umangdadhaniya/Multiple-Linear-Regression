# Load the Cars dataset
ap <- read.csv(file.choose())
View(ap)
ap <- ap[ -c(10,12) ]

View(ap)
attach(ap)

# Normal distribution
qqnorm(AveragePrice)
qqline(AveragePrice)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(tc)

# Scatter plot
plot(tot_ava1, AveragePrice) # Plot relation ships between each X with Y
plot(tot_ava2, AveragePrice)

# Or make a combined plot
pairs(ap)   # Scatter plot for all pairs of variables
plot(ap)

cor(tot_ava1, AveragePrice)
cor(ap) # correlation matrix

# The Linear Model of interest
model.ap <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge.Bags + year , data = ap) # lm(Y ~ X)
summary(model.ap)

model.apV <-lm(AveragePrice ~ Total_Bags)
summary(model.apV)

model.apW <- lm(AveragePrice ~ Small_Bags)
summary(model.apW)

model.app <- lm(AveragePrice ~ Large_Bags)
summary(model.app)

model.tcVW <- lm(AveragePrice ~ Total_Bags + Small_Bags + Large_Bags + XLarge.Bags)
summary(model.tcVW)

#### Scatter plot matrix with Correlations inserted in graph
#install.packages("GGally")
library(GGally)
ggpairs(ap)


### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(ap)

cor2pcor(cor(ap))

# Diagnostic Plots
install.packages(car)
library(car)

plot(model.ap)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.ap, id.n = 5) # QQ plots of studentized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.ap, id.n = 3) # Index Plots of the influence measures
influencePlot(model.ap, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 77th observation
model.ap1 <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge.Bags + year , data = ap[-14126, ])
model.ap1 <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge.Bags + year , data = ap[-15561, ])
model.ap1 <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge.Bags + year , data = ap[-17429, ])
model.ap1 <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge.Bags + year , data = ap[-17469, ])
summary(model.ap1)


### Variance Inflation Factors
vif(model.ap)  # VIF is > 10 => collinearity

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
avPlots(model.ap, id.n = 2, id.cex = 0.8, col = "red")

# Linear Model without WT
model.final <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + year, data = ap)
summary(model.final)

# Linear model without WT and influential observation
model.final1 <-  lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + year , data = ap[-14126, ])
model.final1 <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + year , data = ap[-15561, ])
model.final1 <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + year , data = ap[-17429, ])
model.final1 <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + year , data = ap[-17469, ])
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
