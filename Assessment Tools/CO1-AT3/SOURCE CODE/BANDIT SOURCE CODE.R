# ============================================================================
# EXPLORATION vs EXPLOITATION -- MULTI-ARMED BANDIT: COMPLETE R ANALYSIS
# Domain: Online Advertising (Click-Through-Rate Optimization)
# ============================================================================
# This is a SINGLE, COMPLETE, self-contained R script (not split into
# notebook-style cells). Run it top to bottom in RStudio.
#
# Contents, in order:
#   PART 0 -- Setup (packages, font: Cambria)
#   PART 1 -- Environment & Bandit Simulation (produces the dataset)
#   PART 2 -- Summary Table (domain parameters + RL/bandit terms)
#   PART 3 -- Visualizations (distributions, heatmaps, scatter, network map,
#             boxplots, violin plots, pairwise plot, line/bar comparisons)
#   PART 4 -- Overfitting / Underfitting Graphical Analysis (on the CSV data)
#
# Every plot explicitly labels its X-axis and Y-axis, and every letter of
# text in every plot renders in Cambria (see PART 0 for the font-detection
# logic and its fallback behavior).
# ============================================================================

set.seed(42)

# ============================================================================
# PART 0. SETUP -- Packages and Font (Cambria)
# ============================================================================
required_pkgs <- c("ggplot2", "dplyr", "tidyr", "showtext", "sysfonts", "systemfonts",
                   "igraph", "GGally", "scales")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}
library(ggplot2)
library(dplyr)
library(showtext)
library(sysfonts)

# ---- Locate Cambria on this system; fall back gracefully if not found ----
# Cambria ships by default with Windows/Microsoft Office, so on most Windows
# machines this will be found automatically. If it isn't found (e.g. on
# Linux/Mac without Office installed), the script falls back to the closest
# available serif font rather than failing.
cambria_path <- NA_character_
if (requireNamespace("systemfonts", quietly = TRUE)) {
  cambria_path <- tryCatch({
    hit <- systemfonts::match_fonts("Cambria")
    if (!is.na(hit$path) && file.exists(hit$path)) hit$path else NA_character_
  }, error = function(e) NA_character_)
}

if (!is.na(cambria_path)) {
  sysfonts::font_add(family = "Cambria", regular = cambria_path)
  FONT_FAMILY <- "Cambria"
  cat("Cambria font found on this system -- all plots will use Cambria.\n")
} else {
  FONT_FAMILY <- "serif"
  cat("NOTE: Cambria was not found installed on this system, so plots will\n",
      "fall back to the closest available serif font instead. Cambria ships\n",
      "by default with Windows / Microsoft Office -- if you are on Windows\n",
      "with Office installed, it should normally be detected automatically.\n")
}
showtext_auto()
showtext_opts(dpi = 150)

theme_set(
  theme_minimal(base_family = FONT_FAMILY, base_size = 13) +
    theme(
      text = element_text(family = FONT_FAMILY),
      plot.title = element_text(face = "bold", family = FONT_FAMILY, size = 15),
      plot.subtitle = element_text(family = FONT_FAMILY, size = 11),
      axis.title = element_text(face = "bold", family = FONT_FAMILY, size = 12),
      axis.text = element_text(family = FONT_FAMILY, size = 10),
      legend.text = element_text(family = FONT_FAMILY, size = 10),
      legend.title = element_text(family = FONT_FAMILY, size = 11)
    )
)

# ============================================================================
# PART 1. ENVIRONMENT & BANDIT SIMULATION
# ============================================================================
arm_names <- c("Ad_A", "Ad_B", "Ad_C", "Ad_D")
true_ctr  <- c(Ad_A = 0.05, Ad_B = 0.08, Ad_C = 0.12, Ad_D = 0.10)
n_arms    <- length(arm_names)
best_ctr  <- max(true_ctr)
best_arm  <- names(true_ctr)[which.max(true_ctr)]
n_rounds  <- 2000

device_types <- c("Mobile", "Desktop", "Tablet")
times_of_day <- c("Morning", "Afternoon", "Evening")

cat("True CTRs (hidden from every algorithm):\n"); print(true_ctr)
cat(sprintf("Best possible arm: %s (CTR = %.2f)\n\n", best_arm, best_ctr))

simulate_random <- function(n_rounds) {
  log_df <- data.frame(); cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    arm <- sample(arm_names, 1)
    reward <- rbinom(1, 1, true_ctr[arm]); regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward; cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(Round = t, Arm = arm, Reward = reward,
                                       Cumulative_Reward = cum_reward, Regret = regret, Cumulative_Regret = cum_regret,
                                       Algorithm_Used = "Random (Pure Exploration)"))
  }
  log_df
}

simulate_greedy <- function(n_rounds) {
  counts <- setNames(rep(0, n_arms), arm_names); sums <- setNames(rep(0, n_arms), arm_names)
  log_df <- data.frame(); cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    if (t <= n_arms) { arm <- arm_names[t] } else {
      avg_reward <- sums / counts; arm <- names(avg_reward)[which.max(avg_reward)]
    }
    reward <- rbinom(1, 1, true_ctr[arm])
    counts[arm] <- counts[arm] + 1; sums[arm] <- sums[arm] + reward
    regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward; cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(Round = t, Arm = arm, Reward = reward,
                                       Cumulative_Reward = cum_reward, Regret = regret, Cumulative_Regret = cum_regret,
                                       Algorithm_Used = "Greedy (Pure Exploitation)"))
  }
  log_df
}

simulate_epsilon_greedy <- function(n_rounds, epsilon = 0.1) {
  counts <- setNames(rep(0, n_arms), arm_names); sums <- setNames(rep(0, n_arms), arm_names)
  log_df <- data.frame(); cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    if (runif(1) < epsilon || t <= n_arms) { arm <- sample(arm_names, 1) } else {
      avg_reward <- sums / counts; arm <- names(avg_reward)[which.max(avg_reward)]
    }
    reward <- rbinom(1, 1, true_ctr[arm])
    counts[arm] <- counts[arm] + 1; sums[arm] <- sums[arm] + reward
    regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward; cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(Round = t, Arm = arm, Reward = reward,
                                       Cumulative_Reward = cum_reward, Regret = regret, Cumulative_Regret = cum_regret,
                                       Algorithm_Used = "Epsilon-Greedy (eps=0.1)"))
  }
  log_df
}

simulate_ucb1 <- function(n_rounds) {
  counts <- setNames(rep(0, n_arms), arm_names); sums <- setNames(rep(0, n_arms), arm_names)
  log_df <- data.frame(); cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    if (t <= n_arms) { arm <- arm_names[t] } else {
      avg_reward <- sums / counts; bonus <- sqrt(2 * log(t) / counts)
      ucb_score <- avg_reward + bonus; arm <- names(ucb_score)[which.max(ucb_score)]
    }
    reward <- rbinom(1, 1, true_ctr[arm])
    counts[arm] <- counts[arm] + 1; sums[arm] <- sums[arm] + reward
    regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward; cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(Round = t, Arm = arm, Reward = reward,
                                       Cumulative_Reward = cum_reward, Regret = regret, Cumulative_Regret = cum_regret,
                                       Algorithm_Used = "UCB1"))
  }
  log_df
}

simulate_thompson <- function(n_rounds) {
  alpha <- setNames(rep(1, n_arms), arm_names); beta <- setNames(rep(1, n_arms), arm_names)
  log_df <- data.frame(); cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    sampled_theta <- sapply(arm_names, function(a) rbeta(1, alpha[a], beta[a]))
    arm <- names(sampled_theta)[which.max(sampled_theta)]
    reward <- rbinom(1, 1, true_ctr[arm])
    if (reward == 1) alpha[arm] <- alpha[arm] + 1 else beta[arm] <- beta[arm] + 1
    regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward; cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(Round = t, Arm = arm, Reward = reward,
                                       Cumulative_Reward = cum_reward, Regret = regret, Cumulative_Regret = cum_regret,
                                       Algorithm_Used = "Thompson Sampling"))
  }
  log_df
}

cat("Running simulations for all 5 strategies...\n")
log_random   <- simulate_random(n_rounds)
log_greedy   <- simulate_greedy(n_rounds)
log_epsilon  <- simulate_epsilon_greedy(n_rounds, epsilon = 0.1)
log_ucb1     <- simulate_ucb1(n_rounds)
log_thompson <- simulate_thompson(n_rounds)

bandit_df <- bind_rows(log_random, log_greedy, log_epsilon, log_ucb1, log_thompson)
bandit_df$Session_ID  <- 1:nrow(bandit_df)
bandit_df$Device_Type <- sample(device_types, nrow(bandit_df), replace = TRUE)
bandit_df$Time_of_Day <- sample(times_of_day, nrow(bandit_df), replace = TRUE)
bandit_df$True_CTR    <- true_ctr[bandit_df$Arm]
bandit_df <- bandit_df[, c("Session_ID", "Round", "Algorithm_Used", "Arm", "Device_Type",
                           "Time_of_Day", "Reward", "True_CTR", "Cumulative_Reward",
                           "Regret", "Cumulative_Regret")]

write.csv(bandit_df, "Bandit_Ad_CTR_Dataset.csv", row.names = FALSE)
cat(sprintf("Dataset generated and saved: %d rows x %d columns -> Bandit_Ad_CTR_Dataset.csv\n\n",
            nrow(bandit_df), ncol(bandit_df)))

# ============================================================================
# PART 2. SUMMARY TABLE -- Domain Parameters + RL/Bandit Terms
# ============================================================================
rl_summary <- data.frame(
  Term = c(
    "Domain",
    "Number of Arms (Actions)",
    "Arm Names",
    "Rounds per Algorithm (Time Horizon)",
    "Total Logged Rounds (all algorithms)",
    "State Representation",
    "Reward Signal",
    "Best True Arm",
    "Best True CTR (Reward Probability)",
    "Average Reward (overall, all algorithms)",
    "Average Regret per Round (overall)",
    "Cumulative Regret Definition",
    "Discount Factor (gamma)",
    "Policy",
    "Algorithms Compared",
    "Exploration Mechanism per Algorithm"
  ),
  Value = c(
    "Online Advertising -- Click-Through-Rate (CTR) Optimization",
    n_arms,
    paste(arm_names, collapse = ", "),
    n_rounds,
    nrow(bandit_df),
    "Stateless / context-free bandit -- no explicit environment state; each round is independent",
    "Binary click outcome (1 = click, 0 = no click), drawn from each arm's true CTR",
    best_arm,
    round(best_ctr, 3),
    round(mean(bandit_df$Reward), 4),
    round(mean(bandit_df$Regret), 4),
    "best_CTR - true_CTR(chosen arm), accumulated per round",
    "Not applicable -- bandits are single-step decisions with no sequential state transitions to discount",
    "Mapping from round history to next arm choice; differs per algorithm below",
    paste(unique(bandit_df$Algorithm_Used), collapse = " | "),
    "Random=100% random | Greedy=0% (locks on first estimate) | Epsilon-Greedy=10% random | UCB1=confidence bonus | Thompson=posterior sampling"
  )
)
cat("=== RL / Bandit Terminology Summary Table ===\n")
print(rl_summary, row.names = FALSE)
write.csv(rl_summary, "Bandit_RL_Summary_Table.csv", row.names = FALSE)

algo_summary <- bandit_df %>%
  group_by(Algorithm_Used) %>%
  summarise(
    Total_Reward        = sum(Reward),
    Avg_Reward          = round(mean(Reward), 4),
    Avg_Regret          = round(mean(Regret), 4),
    Final_Cum_Reward    = max(Cumulative_Reward),
    Final_Cum_Regret    = max(Cumulative_Regret),
    Pct_Best_Arm_Chosen = round(mean(Arm == best_arm) * 100, 1),
    .groups = "drop"
  ) %>%
  arrange(Final_Cum_Regret)

cat("\n=== Per-Algorithm Performance Summary Table ===\n")
print(algo_summary, n = Inf)
write.csv(algo_summary, "Bandit_Algorithm_Performance.csv", row.names = FALSE)

# ============================================================================
# PART 3. VISUALIZATIONS
# ============================================================================
algo_levels <- unique(bandit_df$Algorithm_Used)
bandit_df$Algorithm_Used <- factor(bandit_df$Algorithm_Used, levels = algo_levels)

## ---- Fig 1: Cumulative Regret over Rounds (line) ----
p1 <- ggplot(bandit_df, aes(x = Round, y = Cumulative_Regret, color = Algorithm_Used)) +
  geom_line(linewidth = 1) +
  labs(title = "Figure 1. Cumulative Regret Over Rounds",
       subtitle = "Lower = faster learning of the best-performing ad",
       x = "X-Axis: Round Number (Time Step)", y = "Y-Axis: Cumulative Regret",
       color = "Algorithm")
ggsave("fig01_cumulative_regret.png", p1, width = 9, height = 6, dpi = 150)

## ---- Fig 2: Cumulative Reward over Rounds (line) ----
p2 <- ggplot(bandit_df, aes(x = Round, y = Cumulative_Reward, color = Algorithm_Used)) +
  geom_line(linewidth = 1) +
  labs(title = "Figure 2. Cumulative Reward (Clicks) Over Rounds",
       x = "X-Axis: Round Number (Time Step)", y = "Y-Axis: Cumulative Reward (Total Clicks)",
       color = "Algorithm")
ggsave("fig02_cumulative_reward.png", p2, width = 9, height = 6, dpi = 150)

## ---- Fig 3: Reward Distribution Curve (KDE / density) ----
p3 <- ggplot(bandit_df, aes(x = Reward, fill = Algorithm_Used)) +
  geom_density(alpha = 0.35) +
  labs(title = "Figure 3. Reward Distribution Curve (Density)",
       x = "X-Axis: Reward (0 = No Click, 1 = Click)", y = "Y-Axis: Probability Density",
       fill = "Algorithm")
ggsave("fig03_reward_distribution.png", p3, width = 9, height = 6, dpi = 150)

## ---- Fig 4: Regret Distribution Curve (KDE / density) ----
p4 <- ggplot(bandit_df, aes(x = Regret, fill = Algorithm_Used)) +
  geom_density(alpha = 0.35) +
  labs(title = "Figure 4. Per-Round Regret Distribution Curve (Density)",
       x = "X-Axis: Regret (best_CTR - chosen_arm_CTR)", y = "Y-Axis: Probability Density",
       fill = "Algorithm")
ggsave("fig04_regret_distribution.png", p4, width = 9, height = 6, dpi = 150)

## ---- Fig 5: Cumulative Reward Distribution Curve (KDE) ----
p5 <- ggplot(bandit_df, aes(x = Cumulative_Reward, fill = Algorithm_Used)) +
  geom_density(alpha = 0.35) +
  labs(title = "Figure 5. Cumulative Reward Distribution Curve (Density)",
       x = "X-Axis: Cumulative Reward", y = "Y-Axis: Probability Density",
       fill = "Algorithm")
ggsave("fig05_cumreward_distribution.png", p5, width = 9, height = 6, dpi = 150)

## ---- Fig 6: Correlation Heatmap of Numeric Variables ----
num_cols <- bandit_df[, c("Round", "Reward", "True_CTR", "Cumulative_Reward",
                          "Regret", "Cumulative_Regret")]
corr_mat <- cor(num_cols)
corr_df  <- as.data.frame(as.table(corr_mat))
names(corr_df) <- c("Var1", "Var2", "Correlation")
p6 <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile() +
  geom_text(aes(label = round(Correlation, 2)), family = FONT_FAMILY, size = 3.5) +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick", midpoint = 0) +
  labs(title = "Figure 6. Correlation Heatmap of Numeric Variables",
       x = "X-Axis: Variable", y = "Y-Axis: Variable") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("fig06_correlation_heatmap.png", p6, width = 8, height = 7, dpi = 150)

## ---- Fig 7: Arm x Algorithm Heatmap (selection counts) ----
arm_algo_ct <- as.data.frame(table(bandit_df$Algorithm_Used, bandit_df$Arm))
names(arm_algo_ct) <- c("Algorithm", "Arm", "Count")
p7 <- ggplot(arm_algo_ct, aes(x = Arm, y = Algorithm, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), family = FONT_FAMILY, size = 3.5) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(title = "Figure 7. Arm Selection Heatmap by Algorithm",
       x = "X-Axis: Ad Variant (Arm)", y = "Y-Axis: Algorithm")
ggsave("fig07_arm_algorithm_heatmap.png", p7, width = 8, height = 6, dpi = 150)

## ---- Fig 8: Round-Bin x Arm Heatmap (exploration-over-time pattern) ----
bandit_df$Round_Bin <- cut(bandit_df$Round, breaks = 10, labels = FALSE)
roundbin_arm_ct <- as.data.frame(table(bandit_df$Round_Bin, bandit_df$Arm))
names(roundbin_arm_ct) <- c("Round_Bin", "Arm", "Count")
p8 <- ggplot(roundbin_arm_ct, aes(x = Arm, y = Round_Bin, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), family = FONT_FAMILY, size = 3) +
  scale_fill_gradient(low = "white", high = "purple4") +
  labs(title = "Figure 8. Round-Bin vs Arm Heatmap (All Algorithms Combined)",
       subtitle = "Shows how arm selection shifts as rounds progress",
       x = "X-Axis: Ad Variant (Arm)", y = "Y-Axis: Round Bin (1 = earliest, 10 = latest)")
ggsave("fig08_roundbin_arm_heatmap.png", p8, width = 8, height = 7, dpi = 150)

## ---- Fig 9: Scatter Plot -- Cumulative Reward vs Cumulative Regret ----
p9 <- ggplot(bandit_df, aes(x = Cumulative_Regret, y = Cumulative_Reward, color = Algorithm_Used)) +
  geom_point(alpha = 0.3, size = 0.8) +
  labs(title = "Figure 9. Scatter Plot: Cumulative Reward vs Cumulative Regret",
       x = "X-Axis: Cumulative Regret", y = "Y-Axis: Cumulative Reward",
       color = "Algorithm")
ggsave("fig09_scatter_reward_vs_regret.png", p9, width = 9, height = 6, dpi = 150)

## ---- Fig 10: Boxplot of Regret by Algorithm ----
p10 <- ggplot(bandit_df, aes(x = Algorithm_Used, y = Regret, fill = Algorithm_Used)) +
  geom_boxplot() +
  labs(title = "Figure 10. Boxplot: Per-Round Regret by Algorithm",
       x = "X-Axis: Algorithm", y = "Y-Axis: Regret") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1), legend.position = "none")
ggsave("fig10_regret_boxplot.png", p10, width = 9, height = 6, dpi = 150)

## ---- Fig 11: Violin Plot of Reward by Algorithm ----
p11 <- ggplot(bandit_df, aes(x = Algorithm_Used, y = Reward, fill = Algorithm_Used)) +
  geom_violin() +
  labs(title = "Figure 11. Violin Plot: Reward Distribution by Algorithm",
       x = "X-Axis: Algorithm", y = "Y-Axis: Reward (0/1)") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1), legend.position = "none")
ggsave("fig11_reward_violin.png", p11, width = 9, height = 6, dpi = 150)

## ---- Fig 12: Arm Selection Frequency (bar, faceted by algorithm) ----
p12 <- ggplot(bandit_df, aes(x = Arm, fill = Arm)) +
  geom_bar() +
  facet_wrap(~Algorithm_Used) +
  labs(title = "Figure 12. Arm Selection Frequency by Algorithm",
       x = "X-Axis: Ad Variant (Arm)", y = "Y-Axis: Times Selected") +
  theme(legend.position = "none")
ggsave("fig12_arm_selection_frequency.png", p12, width = 10, height = 7, dpi = 150)

## ---- Fig 13: Final Cumulative Regret Comparison (bar) ----
final_regret <- bandit_df %>% group_by(Algorithm_Used) %>%
  summarise(Final_Regret = max(Cumulative_Regret), .groups = "drop")
p13 <- ggplot(final_regret, aes(x = reorder(Algorithm_Used, Final_Regret), y = Final_Regret, fill = Algorithm_Used)) +
  geom_col() + coord_flip() +
  labs(title = "Figure 13. Final Cumulative Regret by Algorithm",
       x = "X-Axis: Algorithm", y = "Y-Axis: Final Cumulative Regret") +
  theme(legend.position = "none")
ggsave("fig13_final_regret_comparison.png", p13, width = 9, height = 6, dpi = 150)

## ---- Fig 14: Percent Optimal Arm Chosen Over Time (rolling average) ----
rolling_optimal <- bandit_df %>%
  group_by(Algorithm_Used) %>%
  arrange(Round) %>%
  mutate(Is_Optimal = as.numeric(Arm == best_arm),
         Rolling_Optimal_Pct = cummean(Is_Optimal) * 100)
p14 <- ggplot(rolling_optimal, aes(x = Round, y = Rolling_Optimal_Pct, color = Algorithm_Used)) +
  geom_line(linewidth = 1) +
  labs(title = "Figure 14. Percent of Rounds the Optimal Ad was Chosen (Running Average)",
       x = "X-Axis: Round Number", y = "Y-Axis: % Optimal Arm Chosen (Running Avg)",
       color = "Algorithm")
ggsave("fig14_optimal_arm_rate.png", p14, width = 9, height = 6, dpi = 150)

## ---- Fig 15: Estimated CTR per Arm vs True CTR (bar + point, faceted) ----
estimated_ctr <- bandit_df %>%
  group_by(Algorithm_Used, Arm) %>%
  summarise(Estimated_CTR = mean(Reward), .groups = "drop") %>%
  left_join(data.frame(Arm = arm_names, True_CTR = as.numeric(true_ctr)), by = "Arm")
p15 <- ggplot(estimated_ctr, aes(x = Arm)) +
  geom_col(aes(y = Estimated_CTR), fill = "lightblue") +
  geom_point(aes(y = True_CTR), color = "red", size = 3) +
  facet_wrap(~Algorithm_Used) +
  labs(title = "Figure 15. Estimated CTR per Arm (bars) vs True CTR (red dots)",
       x = "X-Axis: Ad Variant (Arm)", y = "Y-Axis: Click-Through Rate")
ggsave("fig15_estimated_vs_true_ctr.png", p15, width = 10, height = 7, dpi = 150)

## ---- Fig 16: Arm-Switch Transition Network (analogous to a state-transition map) ----
# Builds a directed graph of how often each algorithm switches from one arm
# to another on consecutive rounds -- the bandit analogue of a state-transition
# network diagram.
library(igraph)
switch_counts <- bandit_df %>%
  arrange(Algorithm_Used, Round) %>%
  group_by(Algorithm_Used) %>%
  mutate(Prev_Arm = lag(Arm)) %>%
  filter(!is.na(Prev_Arm)) %>%
  ungroup() %>%
  count(Prev_Arm, Arm, name = "Weight")

g <- graph_from_data_frame(switch_counts[, c("Prev_Arm", "Arm", "Weight")], directed = TRUE)
png("fig16_arm_switch_network.png", width = 900, height = 900, res = 150)
par(family = FONT_FAMILY)
plot(g,
     edge.width = E(g)$Weight / max(E(g)$Weight) * 8,
     edge.arrow.size = 0.5,
     vertex.color = "lightblue", vertex.size = 40,
     vertex.label.family = FONT_FAMILY, vertex.label.color = "black",
     main = "Figure 16. Arm-Switch Transition Network (All Algorithms Combined)")
dev.off()

## ---- Fig 17: Pairwise Relationships (pairplot equivalent) ----
pair_cols <- bandit_df[, c("Round", "Reward", "Cumulative_Reward", "Regret", "Cumulative_Regret")]
p17 <- tryCatch({
  GGally::ggpairs(pair_cols) +
    labs(title = "Figure 17. Pairwise Relationships Among Core Numeric Variables") +
    theme(text = element_text(family = FONT_FAMILY))
}, error = function(e) NULL)
if (!is.null(p17)) {
  ggsave("fig17_pairwise_relationships.png", p17, width = 10, height = 10, dpi = 150)
} else {
  png("fig17_pairwise_relationships.png", width = 1000, height = 1000, res = 150)
  par(family = FONT_FAMILY)
  pairs(pair_cols, main = "Figure 17. Pairwise Relationships Among Core Numeric Variables")
  dev.off()
}

cat("\nSaved 17 figures (fig01_...png through fig17_...png)\n")

# ============================================================================
# PART 4. OVERFITTING / UNDERFITTING GRAPHICAL ANALYSIS (on the CSV dataset)
# ============================================================================
# Demonstration: fit polynomial regression models of increasing complexity to
# predict a noisy rolling-window CTR estimate from Round, using a small
# train/test split -- the classic setup that visibly shows underfitting
# (low-degree, high bias) transitioning into overfitting (high-degree, high
# variance) as model complexity increases.

window <- 10
bandit_df_sorted <- bandit_df %>% arrange(Algorithm_Used, Round)
bandit_df_sorted <- bandit_df_sorted %>%
  group_by(Algorithm_Used) %>%
  mutate(Rolling_CTR = sapply(seq_along(Reward), function(i) {
    lo <- max(1, i - window + 1); mean(Reward[lo:i])
  })) %>%
  ungroup()

# Focus on Thompson Sampling's log, and intentionally subsample to a small
# dataset -- small samples make the overfitting effect visible at moderate
# polynomial degrees (with 2000 full rows, noise averages out and no
# realistic degree would visibly overfit).
model_data <- bandit_df_sorted %>% filter(Algorithm_Used == "Thompson Sampling")
set.seed(7)
small_sample <- model_data[sample(1:nrow(model_data), 90), ]
small_sample <- small_sample[order(small_sample$Round), ]

set.seed(1)
train_idx <- sample(1:nrow(small_sample), size = round(0.7 * nrow(small_sample)))
train_data <- small_sample[train_idx, ]
test_data  <- small_sample[-train_idx, ]

degrees <- 1:20
train_rmse <- numeric(length(degrees))
test_rmse  <- numeric(length(degrees))
for (i in seq_along(degrees)) {
  d <- degrees[i]
  model <- tryCatch(lm(Rolling_CTR ~ poly(Round, d), data = train_data), error = function(e) NULL)
  if (is.null(model)) { train_rmse[i] <- NA; test_rmse[i] <- NA; next }
  pred_train <- predict(model, train_data)
  pred_test  <- tryCatch(predict(model, test_data), error = function(e) rep(NA, nrow(test_data)))
  train_rmse[i] <- sqrt(mean((train_data$Rolling_CTR - pred_train)^2))
  test_rmse[i]  <- sqrt(mean((test_data$Rolling_CTR - pred_test)^2, na.rm = TRUE))
}
fit_curve_df <- data.frame(Degree = degrees, Train_RMSE = train_rmse, Test_RMSE = test_rmse)
best_degree <- fit_curve_df$Degree[which.min(fit_curve_df$Test_RMSE)]

cat("\n=== Overfitting / Underfitting Analysis ===\n")
print(fit_curve_df, row.names = FALSE)
cat(sprintf("\nBest generalizing model: polynomial degree = %d (lowest Test RMSE)\n", best_degree))
write.csv(fit_curve_df, "Overfitting_Underfitting_Table.csv", row.names = FALSE)

## ---- Fig 18: Train vs Test RMSE across Model Complexity (the core curve) ----
fit_curve_long <- fit_curve_df %>%
  tidyr::pivot_longer(cols = c(Train_RMSE, Test_RMSE), names_to = "Set", values_to = "RMSE")
p18 <- ggplot(fit_curve_long, aes(x = Degree, y = RMSE, color = Set)) +
  geom_line(linewidth = 1) + geom_point(size = 1.5) +
  geom_vline(xintercept = best_degree, linetype = "dashed", color = "grey40") +
  annotate("text", x = best_degree, y = max(fit_curve_long$RMSE, na.rm = TRUE),
           label = "Best Generalization", family = FONT_FAMILY, hjust = -0.1, size = 3.5) +
  labs(title = "Figure 18. Overfitting/Underfitting Curve: Train vs Test RMSE by Model Complexity",
       subtitle = "Left side = underfitting (high bias) | Right side = overfitting (high variance)",
       x = "X-Axis: Polynomial Model Complexity (Degree)", y = "Y-Axis: RMSE (Root Mean Squared Error)",
       color = "Dataset")
ggsave("fig18_overfitting_underfitting_curve.png", p18, width = 9, height = 6, dpi = 150)

## ---- Fig 19: Visual Overfitting Demo -- Underfit vs Good Fit vs Overfit curves ----
underfit_degree <- 1
good_degree     <- best_degree
overfit_degree  <- max(degrees)

fit_grid <- data.frame(Round = seq(min(small_sample$Round), max(small_sample$Round), length.out = 200))
pred_line <- function(d) {
  m <- lm(Rolling_CTR ~ poly(Round, d), data = small_sample)
  predict(m, newdata = fit_grid)
}
fit_grid$Underfit  <- pred_line(underfit_degree)
fit_grid$Good_Fit  <- pred_line(good_degree)
fit_grid$Overfit   <- pred_line(overfit_degree)

fit_grid_long <- tidyr::pivot_longer(fit_grid, cols = c(Underfit, Good_Fit, Overfit),
                                     names_to = "Fit_Type", values_to = "Predicted_CTR")

p19 <- ggplot() +
  geom_point(data = small_sample, aes(x = Round, y = Rolling_CTR), color = "grey40", alpha = 0.6) +
  geom_line(data = fit_grid_long, aes(x = Round, y = Predicted_CTR, color = Fit_Type), linewidth = 1.1) +
  labs(title = "Figure 19. Visual Comparison: Underfitting vs Good Fit vs Overfitting",
       subtitle = sprintf("Degrees shown -- Underfit: %d | Good Fit: %d | Overfit: %d",
                          underfit_degree, good_degree, overfit_degree),
       x = "X-Axis: Round Number", y = "Y-Axis: Rolling CTR Estimate (window = 10 rounds)",
       color = "Model Fit")
ggsave("fig19_underfit_goodfit_overfit_visual.png", p19, width = 9, height = 6, dpi = 150)

cat("\nSaved 2 overfitting/underfitting figures (fig18, fig19) and Overfitting_Underfitting_Table.csv\n")
cat("\n============================================================\n")
cat("ALL OUTPUTS SAVED IN:", getwd(), "\n")
cat("CSV files : Bandit_Ad_CTR_Dataset.csv, Bandit_RL_Summary_Table.csv,\n")
cat("            Bandit_Algorithm_Performance.csv, Overfitting_Underfitting_Table.csv\n")
cat("Figures   : fig01_...png through fig19_...png (19 total)\n")
cat("============================================================\n")