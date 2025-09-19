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
data$Education_ord <- as.numeric(factor(data$Education, 
                                        levels = c("High School", "Bachelor", "Master", "PhD"),
                                        ordered = TRUE))

data$Job_Title_ord <- as.numeric(factor(data$Job_Title,
                                        levels = c("Analyst", "Director", "Engineer", "Manager"),
                                        ordered = TRUE))

num_vars <- data[, c("Salary", "Experience", "Age", "Gender_num", "Education_ord", "Job_Title_ord")]
cor_matrix <- cor(num_vars, use = "complete.obs")

vardai_lt <- c(
  "Atlyginimas",
  "Darbo patirtis (m.)",
  "Amžius (m.)",
  "Lytis (moteris = 1)",
  "Išsilavinimas (ord.)",
  "Pareigos (ord.)"
)

colnames(cor_matrix) <- vardai_lt
rownames(cor_matrix) <- vardai_lt

corrplot(cor_matrix, method = "number", type = "upper",
         tl.col = "black", tl.srt =45 , number.cex = 0.9)


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



# 95% PI (mean ± 1.96*SE)
mean_ci95 <- function(x, conf = 0.95) {
  x <- x[!is.na(x)]
  m  <- mean(x)
  se <- sd(x) / sqrt(length(x))
  z  <- qnorm(1 - (1 - conf)/2)
  data.frame(y = m, ymin = m - z*se, ymax = m + z*se)
}

# ── Išsilavinimas
ggplot(data, aes(x = Education, y = Salary)) +
  stat_summary(fun = mean, geom = "col") +
  stat_summary(fun.data = mean_ci95, geom = "errorbar", width = 0.15) +
  labs(title = "Vidurkiai ir 95% PI pagal išsilavinimą",
       x = "Išsilavinimas", y = "Atlyginimas (USD)") +
  mano_tema

# ── Vietovė
ggplot(data, aes(x = Location, y = Salary)) +
  stat_summary(fun = mean, geom = "col") +
  stat_summary(fun.data = mean_ci95, geom = "errorbar", width = 0.15) +
  labs(title = "Vidurkiai ir 95% PI pagal vietovę",
       x = "Vietovė", y = "Atlyginimas (USD)") +
  mano_tema

# ── Pareigos
ggplot(data, aes(x = Job_Title, y = Salary)) +
  stat_summary(fun = mean, geom = "col") +
  stat_summary(fun.data = mean_ci95, geom = "errorbar", width = 0.15) +
  labs(title = "Vidurkiai ir 95% PI pagal pareigas",
       x = "Pareigos", y = "Atlyginimas (USD)") +
  mano_tema

# ── Lytis
ggplot(data, aes(x = Gender, y = Salary)) +
  stat_summary(fun = mean, geom = "col") +
  stat_summary(fun.data = mean_ci95, geom = "errorbar", width = 0.15) +
  labs(title = "Vidurkiai ir 95% PI pagal lytį",
       x = "Lytis", y = "Atlyginimas (USD)") +
  mano_tema


# 95% PI (mean ± 1.96*SE)
mean_ci95 <- function(x, conf = 0.95) {
  x <- x[!is.na(x)]
  m  <- mean(x); se <- sd(x)/sqrt(length(x)); z <- qnorm(1-(1-conf)/2)
  data.frame(y = m, ymin = m - z*se, ymax = m + z*se)
}

# bendras "zoom" į 5–95 % atlyginimų intervalą (kad nebūtų "plokščios" kolonos)
ylims <- as.numeric(quantile(data$Salary, c(0.05, 0.95), na.rm = TRUE))

# ── Išsilavinimas: spalvos pagal kategoriją, rikiuota pagal vidurkį
ggplot(data, aes(x = reorder(Education, Salary, mean), y = Salary, fill = Education)) +
  stat_summary(fun = mean, geom = "col", alpha = 0.9) +
  stat_summary(fun.data = mean_ci95, geom = "errorbar", width = 0.15) +
  scale_fill_brewer(palette = "Set2") +
  coord_cartesian(ylim = ylims) +
  labs(title = "Vidurkiai ir 95% PI pagal išsilavinimą",
       x = "Išsilavinimas", y = "Atlyginimas (USD)") +
  mano_tema +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 15, hjust = 1))

# ── Vietovė: spalvos pagal kategoriją
ggplot(data, aes(x = reorder(Location, Salary, mean), y = Salary, fill = Location)) +
  stat_summary(fun = mean, geom = "col", alpha = 0.9) +
  stat_summary(fun.data = mean_ci95, geom = "errorbar", width = 0.15) +
  scale_fill_brewer(palette = "Pastel1") +
  coord_cartesian(ylim = ylims) +
  labs(title = "Vidurkiai ir 95% PI pagal vietovę",
       x = "Vietovė", y = "Atlyginimas (USD)") +
  mano_tema +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 15, hjust = 1))

# ── Pareigos: daug kategorijų → flip'intas grafikas, kad būtų skaitoma
ggplot(data, aes(x = reorder(Job_Title, Salary, mean), y = Salary, fill = Job_Title)) +
  stat_summary(fun = mean, geom = "col", alpha = 0.9) +
  stat_summary(fun.data = mean_ci95, geom = "errorbar", width = 0.15) +
  scale_fill_brewer(palette = "Set3") +
  coord_cartesian(ylim = ylims) +
  labs(title = "Vidurkiai ir 95% PI pagal pareigas",
       x = "Pareigos", y = "Atlyginimas (USD)") +
  mano_tema +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 30, hjust = 1))

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

