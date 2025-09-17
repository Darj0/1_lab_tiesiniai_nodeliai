library(car)
df <- read.csv("C:/Users/meliu/Desktop/TM/salary_prediction_data.csv", header = TRUE)
names(df)
str(df) 


# --------------Įsitikinam, ar duomenys turi normalųjį skirstinį--------------
#1. Nubraižome histogramą ir kvantiliu grafiką
options(scipen = 999) 
hist(df$Salary,
     breaks = 30,         
     col = "darkgray",
     main = "Atlyginimo histograma",
     xlab = "Atlyginimas",
     ylab = "Tankis",
     freq = FALSE)  # freq=FALSE, kad būtų tankio (density) skalė, ne dažniai
x_vals <- seq(min(df$Salary), max(df$Salary), length.out = 100)
y_vals <- dnorm(x_vals, mean = mean(df$Salary), sd = sd(df$Salary))
lines(x_vals, y_vals, col = "red", lwd = 2)

library(car)
qqPlot(df$Salary, 
       dist = "norm",
       main = "QQ atlyginimo grafikas",
       xlab = "Teoriniai kvantiliai",
       ylab = "Stebėti atlyginimai")

#2. Statistiniai testai
y <- df$Salary
mu <- mean(y)
sigma <- sd(y)
# Kolmogorovo–Smirnovo testas 
ks.test(y, "pnorm", mean = mu, sd = sigma)
#p reiksme (0.3418) didesne uz reiksmingumo lygmeni alfa = 0.05 -> neatmetame H0, 
#duomenys yra normaliai pasiskirste.

#Shapiro-Wilk testas
shapiro.test(df$Salary)
#p reiksme = 5.786e-05 -> skirstinys nera normalusis.

#Anderson-Darling testas
library(nortest)
ad.test(df$Salary)
#p reiksme = 0.0001573 -> skirstinys nera normalusis.

# -----------------------Pradinė duomenų analizė-----------------------
# 1. Stebėjimų patikra
dim(df)          # 1000 stebėjimų, 7 kintamieji 
any(is.na(df))   # nėra trūkstamų reikšmių

# 2. Koreliacinė matrica (kiekybiniai kintamieji)
num_vars <- df[, c("Salary", "Age", "Experience")]
cor(num_vars)

library(corrplot)
num_vars <- df[, c("Salary", "Age", "Experience")]
cor_matrix <- cor(num_vars, use = "complete.obs")
corrplot(cor_matrix, method = "number", type = "upper")

# 3. Stačiakampės diagramos
library(ggplot2)

# Bendras temos šablonas 
mano_tema <- theme_minimal(base_size = 14) +
  theme(             
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(5, "pt"),         
    axis.line.x = element_line(color = "black"), 
    axis.line.y = element_line(color = "black"), 
    plot.title = element_text(hjust = 0.5)     
  )

# Išsilavinimas
ggplot(df, aes(x = Education, y = Salary, fill = Education)) +
  geom_boxplot() +
  labs(title = "Atlyginimas pagal išsilavinimą",
       x = "Išsilavinimas", y = "Atlyginimas (JAV doleriai)") +
  mano_tema

# Vietovė
ggplot(df, aes(x = Location, y = Salary, fill = Location)) +
  geom_boxplot() +
  labs(title = "Atlyginimas pagal vietovę",
       x = "Vietovė", y = "Atlyginimas (JAV doleriai)") +
  mano_tema

# Pareigos
ggplot(df, aes(x = Job_Title, y = Salary, fill = Job_Title)) +
  geom_boxplot() +
  labs(title = "Atlyginimas pagal pareigas",
       x = "Pareigos", y = "Atlyginimas (JAV doleriai)") +
  mano_tema

# Lytis
ggplot(df, aes(x = Gender, y = Salary, fill = Gender)) +
  geom_boxplot() +
  labs(title = "Atlyginimas pagal lytį",
       x = "Lytis", y = "Atlyginimas (JAV doleriai)") +
  mano_tema

#4 . Sklaidos diagramos: atlyginimo prieš regresorius
plot(df$Experience, df$Salary,
     col = "steelblue", pch = 19,
     main = "Sklaidos diagrama: darbo patirtis ir atlyginimas",
     xlab = "Darbo patirtis (m.)",
     ylab = "Atlyginimas (JAV doleriai)")
abline(lm(Salary ~ Experience, data = df), col = "red", lwd = 2) #tiesinė regresijos linija
corr_val <- cor(df$Experience, df$Salary)
legend("topleft",
       legend = paste("Koreliacija:", round(corr_val, 2)),
       bty = "n")


plot(df$Age, df$Salary,
     col = "forestgreen", pch = 19,
     main = "Sklaidos diagrama: amžius ir atlyginimas",
     xlab = "Amžius (m.)",
     ylab = "Atlyginimas (JAV doleriai)")
abline(lm(Salary ~ Age, data = df), col = "red", lwd = 2)
corr_val <- cor(df$Age, df$Salary)
legend("topleft",
       legend = paste("Koreliacija:", round(corr_val, 2)),
       bty = "n")




0