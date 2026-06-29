library(dplyr)
library(readr)
library(lubridate)
library(tidyr)
library(stringr)
library(readxl) 

#  LOAD DATASETS
jumps <- read_delim("Jumps.csv", delim = ";", show_col_types = FALSE)
wellness <- read_delim("Wellness.csv", delim = ";", show_col_types = FALSE)
ex_train <- read_delim("ExerciseTrainingData.csv", delim = ";", show_col_types = FALSE)
strength <- read_delim("StrengthTraining.csv", delim = ";", show_col_types = FALSE)
plr_train <- read_excel("PlayerTrainingData.xlsx") # Load as Excel

#  FIX DECIMALS IN STRENGTH TRAINING
strength <- strength %>%
  mutate(
    Prct = as.numeric(str_replace(Prct, ",", ".")),
    Weight = as.numeric(str_replace(Weight, ",", "."))
  )

#  STANDARDIZE ALL DATES 
safe_date <- function(x) if(is.character(x)) dmy(x) else as.Date(x)

jumps$Date <- safe_date(jumps$Date)
wellness$Date <- safe_date(wellness$Date)
ex_train$Date <- safe_date(ex_train$Date)
strength$Date <- safe_date(strength$Date)

#  MAP DATES TO PLAYER TRAINING AND COMPUTE INTERNAL LOAD
date_map <- ex_train %>%
  select(TrainingID, Date) %>%
  drop_na() %>%
  distinct()

plr_train <- plr_train %>%
  inner_join(date_map, by = "TrainingID") %>%
  mutate(
    # Force to character to bypass Excel formatting inconsistencies
    dur_char = as.character(Duration),
    
    time_str = str_extract(dur_char, "\\d{2}:\\d{2}:\\d{2}"),
    
    Duration_minutes = period_to_seconds(hms(time_str)) / 60,
    sRPE = RPE * Duration_minutes
  ) %>%
  select(-dur_char, -time_str) 

daily_internal_load <- plr_train %>%
  group_by(Date) %>%
  summarise(Daily_sRPE = sum(sRPE, na.rm = TRUE))

# AGGREGATE DAILY FEATURES
daily_jumps <- jumps %>%
  group_by(Date) %>%
  summarise(
    Jump_Volume = n(),
    Jump_Max = max(HeightInCm, na.rm = TRUE),
    Jump_Mean = mean(HeightInCm, na.rm = TRUE)
  )


daily_strength <- strength %>%
  mutate(Volume_Load = coalesce(Reps * Weight, Reps * Prct)) %>%
  group_by(Date) %>%
  summarise(Daily_Volume_Load = sum(Volume_Load, na.rm = TRUE))

#  SYNTHESIZE TARGET VARIABLE (DAILY WELLNESS)
ostrc_cols <- c("Difficultparticipating", "Reducedtraining", "Affectedperformance", "Symptomscomplaints")

daily_wellness <- wellness %>%
  mutate(across(all_of(ostrc_cols), ~ replace_na(as.numeric(.), 0))) %>%
  mutate(OSTRC_Score = rowSums(select(., all_of(ostrc_cols)))) %>%
  select(Date, OSTRC_Score, `Hours of sleep`, `Sleep quality`)

# FINAL MASTER JOIN
master_df <- daily_wellness %>%
  left_join(daily_jumps, by = "Date") %>%
  left_join(daily_internal_load, by = "Date") %>%
  left_join(daily_strength, by = "Date") %>%
  mutate(across(
    c(Jump_Volume, Jump_Max, Jump_Mean, Daily_sRPE, Daily_Volume_Load), 
    ~ replace_na(., 0)
  ))
############_____________ANALYSIS____________________###############################

library(dlnm)
library(mgcv)
library(splines)
library(ggplot2)

#  ENFORCE CHRONOLOGICAL ORDER
master_df <- master_df %>% arrange(Date)

# DEFINE CROSS-BASIS MATRICES (Bulletproofed for zero-inflated data)

# argvar is set to "lin" (linear dose-response)
# arglag remains "ns" (non-linear temporal decay)

cb_jumps <- crossbasis(master_df$Jump_Volume, lag = 7, 
                       argvar = list(fun = "lin"), 
                       arglag = list(fun = "ns", df = 3))

cb_sRPE <- crossbasis(master_df$Daily_sRPE, lag = 7, 
                      argvar = list(fun = "lin"), 
                      arglag = list(fun = "ns", df = 3))

cb_gym <- crossbasis(master_df$Daily_Volume_Load, lag = 7, 
                     argvar = list(fun = "lin"), 
                     arglag = list(fun = "ns", df = 3))
#  FIT THE MODEL
# We add s(Time_Index) to control for seasonal baseline drift (e.g., end-of-season exhaustion)
master_df$Time_Index <- as.numeric(master_df$Date) - min(as.numeric(master_df$Date)) + 1

model_dlnm <- gam(OSTRC_Score ~ cb_jumps + cb_sRPE + cb_gym + s(Time_Index, k = 10), 
                  family = gaussian(), 
                  data = master_df)
summary(model_dlnm)

#  GENERATE PREDICTIONS FOR THE DASHBOARD
# Initialize empty column
master_df$Predicted_Risk <- NA 

# Map the 78 predictions back to the exact row indices the model used
valid_row_indices <- as.numeric(rownames(model_dlnm$model))
master_df$Predicted_Risk[valid_row_indices] <- predict(model_dlnm, type = "response")

# Define a "Red Zone" threshold (e.g., 80th percentile of historical severity)
risk_threshold <- quantile(master_df$OSTRC_Score, 0.8, na.rm = TRUE)

# PLOT THE CLIENT MOCK-UP
ggplot(master_df, aes(x = Date)) +
  geom_line(aes(y = Predicted_Risk, color = "Lagged Model Prediction (Risk Index)"), linewidth = 1.2) +
  geom_point(aes(y = OSTRC_Score, color = "Actual Reported Symptoms"), alpha = 0.4) +
  geom_hline(yintercept = risk_threshold, linetype = "dashed", color = "red", linewidth = 1) +
  annotate("text", x = min(master_df$Date) + 10, y = risk_threshold + 1, 
           label = "CRITICAL RISK THRESHOLD", color = "red", fontface = "bold") +
  scale_color_manual(values = c("Actual Reported Symptoms" = "grey50", 
                                "Lagged Model Prediction (Risk Index)" = "blue")) +
  labs(title = "Coach's Dashboard: Predictive Patellar Tendinopathy Risk",
       subtitle = "Alert triggers when cumulative 7-day lagged load breaches critical threshold",
       y = "OSTRC Overuse Severity Score", 
       x = "Season Timeline",
       color = "Metric") +
  theme_minimal() +
  theme(legend.position = "bottom")

