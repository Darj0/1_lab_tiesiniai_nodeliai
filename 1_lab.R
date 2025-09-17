# ------------------- Bibliotekos -------------------
library(nortest)    # Anderson-Darling testui
library(MASS)       # boxcox, stepAIC
library(car)        # qqPlot
library(corrplot)   # koreliacijos grafikai
library(ggplot2)    # grafikai

# ------------------- 1. Duomenų skaitymas -------------------
data <- read.csv("/Users/darjabaranova/Documents/VU/3_kursas/tiesiniai_metodai/salary_prediction_data.csv",
                 stringsAsFactors = FALSE)

# 2. Kategorinių kintamųjų nustatymas
data$Education <- factor(data$Education)
data$Location <- factor(data$Location)
data$Job_Title <- factor(data$Job_Title)
data$Gender <- factor(data$Gender)
data$Gender_num <- ifelse(data$Gender == "Female", 1, 0)  # kiekybinė lytis koreliacijai

# ------------------- 2. Pradinė kintamojo "Salary" analizė -------------------
summary(data$Salary)


# ------------------- 3. Normalumo testai -------------------
# Kolmogorovo–Smirnovo testas
ks.test(data$Salary, "pnorm", mean = mean(data$Salary), sd = sd(data$Salary))


# Histogramos su tankio linija
options(scipen = 999)
hist(data$Salary, breaks=30, freq=FALSE, col="darkgray",
     main="Atlyginimo histograma", xlab="Atlyginimas", ylab="Tankis")
x_vals <- seq(min(data$Salary), max(data$Salary), length.out=100)
lines(x_vals, dnorm(x_vals, mean=mean(data$Salary), sd=sd(data$Salary)), col="red", lwd=2)


# QQ grafikas
qqPlot(data$Salary, dist = "norm",
       main = "QQ grafikas: atlyginimas",
       xlab = "Teoriniai kvantiliai", ylab = "Stebėti atlyginimai")

# ------------------- 4. Koreliacija (kiekybiniai kintamieji) -------------------
num_vars <- data[, c("Salary", "Experience", "Age", "Gender_num")]
cor_matrix <- cor(num_vars, use="complete.obs")
print(cor_matrix)
corrplot(cor_matrix, method="number", type="upper")


### prideti issilavinima bei pareigas
# ------------------- 5. Stačiakampės diagramos -------------------
mano_tema <- theme_minimal(base_size=14) +
  theme(
    axis.ticks = element_line(color="black"),
    axis.ticks.length = unit(5, "pt"),
    axis.line.x = element_line(color="black"),
    axis.line.y = element_line(color="black"),
    plot.title = element_text(hjust=0.5)
  )

# Išsilavinimas
ggplot(data, aes(x=Education, y=Salary, fill=Education)) +
  geom_boxplot() +
  labs(title="Atlyginimas pagal išsilavinimą", x="Išsilavinimas", y="Atlyginimas (USD)") +
  mano_tema

# Vietovė
ggplot(data, aes(x=Location, y=Salary, fill=Location)) +
  geom_boxplot() +
  labs(title="Atlyginimas pagal vietovę", x="Vietovė", y="Atlyginimas (USD)") +
  mano_tema

# Pareigos
ggplot(data, aes(x=Job_Title, y=Salary, fill=Job_Title)) +
  geom_boxplot() +
  labs(title="Atlyginimas pagal pareigas", x="Pareigos", y="Atlyginimas (USD)") +
  mano_tema

# Lytis
ggplot(data, aes(x=Gender, y=Salary, fill=Gender)) +
  geom_boxplot() +
  labs(title="Atlyginimas pagal lytį", x="Lytis", y="Atlyginimas (USD)") +
  mano_tema

# ------------------- 6. Sklaidos diagramos -------------------
plot(data$Experience, data$Salary,
     col="steelblue", pch=19,
     main="Darbo patirtis ir atlyginimas",
     xlab="Darbo patirtis (m.)", ylab="Atlyginimas (USD)")
abline(lm(Salary ~ Experience, data=data), col="red", lwd=2)
legend("topleft", legend=paste("Koreliacija:", round(cor(data$Experience, data$Salary),2)), bty="n")

plot(data$Age, data$Salary,
     col="forestgreen", pch=19,
     main="Amžius ir atlyginimas",
     xlab="Amžius (m.)", ylab="Atlyginimas (USD)")
abline(lm(Salary ~ Age, data=data), col="red", lwd=2)
legend("topleft", legend=paste("Koreliacija:", round(cor(data$Age, data$Salary),2)), bty="n")

