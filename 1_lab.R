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

hist(data$Salary,
     breaks = 30,
     main = "Pradinis atlyginimo pasiskirstymas",
     xlab = "Atlyginimas",
     col = "lightblue",
     border = "white")

# Kvadratinė šaknis ir logaritminė transformacija
data$Salary_sqrt <- sqrt(data$Salary)
data$Salary_log <- log(data$Salary)

# ------------------- 3. Normalumo testai -------------------
# Kolmogorovo–Smirnovo testas
ks.test(data$Salary, "pnorm", mean = mean(data$Salary), sd = sd(data$Salary))

# Shapiro-Wilk testas
shapiro.test(data$Salary)

# Anderson-Darling testas
ad.test(data$Salary)
ad.test(data$Salary_sqrt)
ad.test(data$Salary_log)

# QQ grafikas
qqPlot(data$Salary, dist = "norm",
       main = "QQ grafikas: atlyginimas",
       xlab = "Teoriniai kvantiliai", ylab = "Stebėti atlyginimai")

# Histogramos su tankio linija
options(scipen = 999)
hist(data$Salary, breaks=30, freq=FALSE, col="darkgray",
     main="Atlyginimo histograma", xlab="Atlyginimas", ylab="Tankis")
x_vals <- seq(min(data$Salary), max(data$Salary), length.out=100)
lines(x_vals, dnorm(x_vals, mean=mean(data$Salary), sd=sd(data$Salary)), col="red", lwd=2)

# ------------------- 4. Koreliacija (kiekybiniai kintamieji) -------------------
num_vars <- data[, c("Salary", "Experience", "Age", "Gender_num")]
cor_matrix <- cor(num_vars, use="complete.obs")
print(cor_matrix)
corrplot(cor_matrix, method="number", type="upper")

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

# ------------------- 7. Pilnas regresijos modelis -------------------
model_full <- lm(Salary ~ Education + Experience + Location + Job_Title + Age + Gender, data=data)
summary(model_full)

# Diagnostikos grafikai
par(mfrow=c(2,2))
plot(model_full)

# ------------------- 8. Kintamųjų atranka pagal AIC -------------------
model_step <- stepAIC(model_full, direction="both", trace=FALSE)
summary(model_step)

par(mfrow=c(2,2))
plot(model_step)

# ------------------- 9. Prognozės pavyzdys -------------------
new_data <- data.frame(
  Education="High School",
  Experience=5,
  Location="Rural",
  Job_Title="Analyst",
  Age=28,
  Gender="Female"
)
predict(model_step, newdata=new_data, interval="prediction")
