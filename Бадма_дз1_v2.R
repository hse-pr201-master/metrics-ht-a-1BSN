library(tidyverse)
library(lmtest)
library(memisc)
library(car)
library(psych)
library(sjPlot)
library(MASS)
library(caret)
library(mlbench)
library(ggstatsplot)
library(pls)
library(mctest)
library(perturb)
library(sandwich)
set.seed(5)
setwd("~/Documents")

# Задание 1 Подготовка данных
df <- read.csv('forestfires.csv')
df %>% drop_na() # Загружаю данные, фильтрую, логарифимирую
df <- filter(df, area > 0)
describe(df) # Я логарифмирую переменные, потому что они разного масштаба
df[c('ISI')] <- df[c('ISI')] + 1 
df['area'] <- df['area'] + 1 # Поправка, чтобы значения логарифмов были положительные
df <- mutate(df, logisi = log(ISI), logdmc = log(DMC),
             logdc = log(DC), logffmc = log(FFMC)
             , logarea = log(area))
df <- mutate(df, logrh = log(RH))
win <- c('dec', 'jan', 'feb') # Замена месяцев на сезоны
spr <- c('mar', 'apr', 'may')
sumr <- c('jun', 'jul', 'aug')
aut <- c('sep', 'oct', 'nov')
f <- function(x){
  if (x %in% win)
    return ("winter")
  else if (x %in% spr)
    return ("spring")
  else if (x %in% sumr)
    return ("summer")
  else
    return ("autumn")
}
wd <- c('mon', 'tue', 'wed', 'thu', 'fri')
we <- c('sat', 'sun') # Введение категории будни и выходные
f1 <- function(x){
  if (x %in% wd)
    return ("weekday")
  else if (x %in% we)
    return ("weekend")
}
df[3] <- apply(df[3],1, f)
df[4] <- apply(df[4], 1, f1)
df$month <- as.factor(df$month)
df$day <- as.factor(df$day)
df$RH <- NULL
df$ISI <- NULL
#df$area <- NULL
df$DMC <- NULL
df$DC <- NULL # Удаляю старые значения
df$FFMC <- NULL
df <- mutate(df, logtemp = log(temp))
df$temp <- NULL

# Теперь я удалю выбросы из целевой переменной
outlier_values <- boxplot.stats(df$area)$out  # outlier values.
boxplot(df$area, main="Площадь пожаров", boxwex=0.1, ylab='Гектары')
# Выбросы очень заметны, их нужно убрать
outliers <- boxplot(df$area, plot=FALSE)$out
df_clean <- df[-which(df$area %in% outliers),]
boxplot(df_clean$area, main="Площадь пожаров (очищенная)", boxwex=0.1, ylab='Гектары')
# Всё равно достаточно плохо, но уже лучше
df_clean$area <- NULL # Буду предсказывать логарифм площади

# Задание 2 Отбор признаков
full.model <- lm(logarea ~. , data = df_clean)
step.model <- stepAIC(full.model, direction = "both", 
                      trace = FALSE)
summary(step.model) # R^2_adj = 0.09
# Отбор признаков последовательным исключением регрессоров по AIC
# Признаки, которые я буду использовать: day, month, logdmc, logrh, logtemp, использовать их буду в 4 пункте
describe(df_clean[c('logarea', 'month', 'day', # дескриптивные статистики
                    'logdmc', 'logrh', 'logtemp')])[, 3:10]
par(mfrow=c(1,1))
hist(df_clean$logarea, main = 'Целевая переменная', ylab = 'Частотность', xlab='Гектары')
boxplot(df_clean$logarea, main='Ящичковая диаграмма логплощади', boxwex=0.1, ylab='Гектары')

hist(df_clean$logdmc, main = 'Log DMC', ylab = 'Частотность', xlab='Индекс')
boxplot(df_clean$logdmc, main='Ящичковая диаграмма Log DMC', boxwex=0.1, ylab='Индекс')

hist(df_clean$logrh, main = 'Log RH', ylab = 'Частотность', xlab='Индекс')
boxplot(df_clean$logrh, main='Ящичковая диаграмма Log RH', boxwex=0.1, ylab='Индекс')

hist(df_clean$logtemp, main = 'Log Temp', ylab = 'Частотность', xlab='Логградусы')
boxplot(df_clean$logtemp, main='Ящичковая диаграмма Log Temp', boxwex=0.1, ylab='Логградусы')

# Подробнее исследую целевую переменную
plot(df_clean$logarea, type='l', main='Логплощадь', xlab = 'Время', ylab='Площадь')
# Заметно, что около индекса 80 data generating process имеет структурный разрыв
# Поэтому я введу дамми переменную для всех значений до индекса 80
# В самом датасете пожары располагаются в хронологическом порядке, поэтому структурный разрыв имеет смысл
# Возможно, произошла смена руководства парка
dummy <- c(rep(1, 81), rep(0, 152))
df_clean['dummy'] <- as.factor(dummy)
# Таким образом, я прологарифмировал переменные, отобрал их по AIC, удалил выбросы целевой переменной,
# Добавил дамми, преобразовал иные качественные переменные (month, day)

# Задание 3 Мультиколлинеарность
as.data.frame(vif(step.model))$GVIF # Оценка полной модели со всеми переменными на VIF показывает, что мультиколлинеарность
# Слабо выражена: VIF > 8 => мультиколлинеарность => Беспокоиться не о чем
olsrr::ols_eigen_cindex(step.model)['Condition Index'] 
# Если у 2-х и более регрессоров показтель CN > 30 => мультиколлинеарность
# В нашем случае беспокоиться не о чем
# Ссылка: https://cran.r-project.org/web/packages/olsrr/vignettes/regression_diagnostics.html

# Задание 4 Оценка модели
model_1 <- lm(logarea ~ logdmc + logtemp + logrh + day + month +
              dummy, data = df_clean)
summary(model_1) # много незначимых регрессоров, нужна новая спецификация
model_2 <- lm(logarea ~ logdmc + logtemp:logrh + day + month,
              data = df_clean)
summary(model_2) # Стало значительно лучше
model_3 <- lm(logarea ~ logdmc + logtemp:logrh + month + day:dummy,
              data = df_clean) # Я взял произведение logrh и logtemp
summary(model_3) # Потому что это показатели влажности и температуры, которые связаны нелинейно
model_4 <- lm(logarea ~ logdmc + logtemp:logrh + month + day
              -1, data = df_clean) # Все регрессоры, кроме одного, высокозначимы
summary(model_4) # Модель без свободного члена даёт намного лучшие результаты
# Теперь я преобразую целевую переменную методом Бокса-Кокса, чтобы отразить нелинейность
boxcox_model_4 <- boxcox(logarea ~ logdmc + logtemp:logrh + month
                         + day -1, data = df_clean)
lambda <- b$x
lik <- b$y
sorted_bc <- cbind(lambda, lik)[order(-lik),]
head(sorted_bc, n = 10) # Наибольшее значение для лямбды 0.7
model_base <- lm(logarea^(0.7) ~ logdmc + logtemp:logrh + month
                 + day -1, data = df_clean)
summary(model_base) # R^2 adj = 0.9, значимы все регрессоры (кроме 1 на 10% УЗ)
# Регрессия значима в целом, p-value < 2.2e-16
qqPlot(model_base$residuals, ylab = 'Остатки модели', xlab = 'Квантили гаус.распр.')
shapiro.test(model_base$residuals) # На УЗ 5% гипотеза о нормальности остатков не отверг.

# Задание 5 Построение прогнозов
new_data = data.frame(logtemp = median(df_clean$logtemp), 
                      logrh = median(df_clean$logrh),
                      logdmc = median(df_clean$logdmc),
                      month = median(as.integer(df_clean$month)),
                      day = median(as.integer(df_clean$day)))
head(df_clean$day, 10)
head(as.integer(df_clean$day), 10)
new_data$day <- as.factor('weekday')
head(df_clean$month, 10)
head(as.integer(df_clean$month), 10)
new_data$month <- as.factor('summer')
exp(predict(model_base, newdata=new_data)^(1/0.7)) # Точечный прогноз
exp(predict(model_base, newdata = new_data, interval = 'prediction')^(1/0.7))
exp(predict(model_base, newdata = new_data, interval = 'confidence')^(1/0.7))

# Задание 6 Возможная гетероскедастичность
describe(df_clean[c('logtemp', 'logrh', 'logdmc')])$sd
# Гетероскедастичность может быть связана с DMC, наибольшая дисперсия
# Кроме того, DMC -- индекс увлажнённости листьев, что может влиять на площадь пожаров, то есть на остатки регрессии

# Задание 7
gqtest(model_base, order.by = ~logdmc, data = df_clean,  fraction = 0.2)
# Тест Гольдефельда-Квандта не позволяет говорить о наличии гетероскедастичности (Log DMC)
gqtest(model_base, order.by = ~logrh, data = df_clean,  fraction = 0.2)
# Дополнительный тест на гомоскедастичность относительно Log RH (не отвергается)
gqtest(model_base, order.by = ~logtemp, data = df_clean,  fraction = 0.2)
# Дополнительный тест на гомоскедастичность относительно Log Temp (не отвергается на 5%)
plot(x = model_base$model$logdmc, y = model_base$residuals, ylab = 'Остатки регрессии', xlab = 'Log DMC', main = 'Диаграмма рассеяния')
# Разброс примерно постоянный => Гетероскедастичности нет

# Задание 8 Взвешенный МНК
model_wls <- lm(logarea^(0.7) ~ logdmc + logtemp:logrh + month
                 + day -1, data = df_clean, weights=1/logdmc^2)
summary(model_wls) # R^2 adj неожиданно вырос до 0.91, коэффициенты по прежнему значимы
# В общем-то, ничего особенно не изменилось, ведь тесты не указывали на наличие гетероскедастичности

# Задание 9 Робастные ошибки
coeftest(model_base, vcov. = vcovHC(model_base, type = 'HC0'))
coeftest(model_base, vcov. = vcovHC(model_base, type = 'HC3'))
# Никаких существенных различий в значимости переменных нет

# Задание 10 Метод главных компонент
pc_all <- pcr(logarea^(0.7) ~ logdmc +
            month + day + logrh:logtemp + wind -1, scale = TRUE, validation = "CV", data = df_clean)
summary(pc_all) # Две первых главных компоненты объясняют 58.71% дисперсии матрицы регрессоров
df_clean.pca <- prcomp(df_clean[c(1, 2, 5, 6, 7, 8, 9
                                  , 10, 12, 13)], scale. = TRUE)
comp_1 <- df_clean.pca$x[, 1]
comp_2 <- df_clean.pca$x[, 2]
pc_2 <- lm(df_clean$logarea^0.7 ~ pc1 + pc2)
summary(pc_2) # Регрессия ужасного качества, главные компоненты не
# Позволяют моделировать нелинейные взаимосвязи между перменными в этой задаче
# Гипотеза о значимости регрессии в целом отвергается, вторая компонента не значима

# Задание 11 Градиентный спуск
model_ols <- lm(logarea ~ logdmc + logrh + logtemp -1, data = df_clean)
ols <- as.data.frame(model_ols$coefficients) # OLS-модель
colnames(ols) <- 'OLS' # Из модели были удалены бинарные переменные
grad_des <- function(x, y, learning_rate) {
  b <- runif(ncol(x), 0, 1)
  converged = FALSE
  iters <- 0
  b_new <- 0
  while(converged == FALSE) {
    b_new <- b - learning_rate * ( 1 / nrow(x) * t(x) %*% (x %*% b - y))
    delta <- b_new - b
    b <- b_new
    if( abs(delta) <= 1e-5) {
      converged = TRUE
    }
    iters <- iters + 1
    if(iters >= 20000) break
  }
  return(b)
} # Довольно простая и бесхитростная функция
# Минус этой функции в том, что я не очень понимаю, как можно оценить бинарные переменные
gd <- grad_des(as.matrix(df_clean[c('logdmc', 'logrh', 'logtemp'
                              )]),as.matrix(df_clean$logarea), 0.01)
colnames(gd) <- 'Grad Desc'
cbind(ols, gd)



