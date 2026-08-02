# ============================================================================
# Exploration vs. Exploitation: Multi-Armed Bandit Case Study (R)
# Domain: Online Advertising -- Click-Through-Rate (CTR) Optimization
# ============================================================================
# Problem: A platform can show one of several ad variants ("arms") to each
# visitor. The TRUE click-through rate of each ad is unknown to the
# algorithm and must be learned from experience, round by round.
#
# This script:
#   1. Defines the ad environment (hidden true CTRs per arm)
#   2. Implements 5 strategies spanning the explore/exploit spectrum:
#        - Random          (pure exploration)
#        - Greedy          (pure exploitation)
#        - Epsilon-Greedy  (mostly exploit, occasionally explore)
#        - UCB1            (optimism under uncertainty)
#        - Thompson Sampling (Bayesian probability matching)
#   3. Simulates each strategy over many rounds, logging every round
#   4. Saves one combined CSV dataset (one row per round per algorithm)
#   5. Builds an RL/bandit-terms summary table
#   6. Produces comparison visualizations (regret, reward, arm selection)
# ============================================================================

set.seed(42)

## ---------------------------------------------------------------------
## 1. Install / load required packages
## ---------------------------------------------------------------------
required_pkgs <- c("ggplot2", "dplyr", "tidyr")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}
library(ggplot2)
library(dplyr)
library(tidyr)

## ---------------------------------------------------------------------
## 2. Define the Ad Environment (ground truth, hidden from the algorithms)
## ---------------------------------------------------------------------
arm_names <- c("Ad_A", "Ad_B", "Ad_C", "Ad_D")
true_ctr  <- c(Ad_A = 0.05, Ad_B = 0.08, Ad_C = 0.12, Ad_D = 0.10)
n_arms    <- length(arm_names)
best_ctr  <- max(true_ctr)
best_arm  <- names(true_ctr)[which.max(true_ctr)]

n_rounds  <- 2000   # number of ad impressions simulated per algorithm

device_types <- c("Mobile", "Desktop", "Tablet")
times_of_day <- c("Morning", "Afternoon", "Evening")

cat("True CTRs (hidden from all algorithms):\n")
print(true_ctr)
cat(sprintf("Best possible arm: %s (CTR = %.2f)\n\n", best_arm, best_ctr))

## ---------------------------------------------------------------------
## 3. Bandit Algorithm Implementations
## ---------------------------------------------------------------------
# Each function simulates n_rounds of interaction with the environment and
# returns a data.frame log: Round, Arm, Reward(click), CumReward, Regret,
# CumRegret, Algorithm_Used.

simulate_random <- function(n_rounds) {
  # ---- Pure Exploration: pick an arm uniformly at random every round ----
  log_df <- data.frame()
  cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    arm <- sample(arm_names, 1)
    reward <- rbinom(1, 1, true_ctr[arm])
    regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward
    cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(
      Round = t, Arm = arm, Reward = reward,
      Cumulative_Reward = cum_reward, Regret = regret,
      Cumulative_Regret = cum_regret, Algorithm_Used = "Random (Pure Exploration)"
    ))
  }
  log_df
}

simulate_greedy <- function(n_rounds) {
  # ---- Pure Exploitation: always pick the best arm seen so far
  #      (each arm is tried once first to avoid divide-by-zero) ----
  counts <- setNames(rep(0, n_arms), arm_names)
  sums   <- setNames(rep(0, n_arms), arm_names)
  log_df <- data.frame()
  cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    if (t <= n_arms) {
      arm <- arm_names[t]   # try each arm once initially
    } else {
      avg_reward <- sums / counts
      arm <- names(avg_reward)[which.max(avg_reward)]
    }
    reward <- rbinom(1, 1, true_ctr[arm])
    counts[arm] <- counts[arm] + 1
    sums[arm]   <- sums[arm] + reward
    regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward
    cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(
      Round = t, Arm = arm, Reward = reward,
      Cumulative_Reward = cum_reward, Regret = regret,
      Cumulative_Regret = cum_regret, Algorithm_Used = "Greedy (Pure Exploitation)"
    ))
  }
  log_df
}

simulate_epsilon_greedy <- function(n_rounds, epsilon = 0.1) {
  # ---- Epsilon-Greedy: explore with probability epsilon, else exploit ----
  counts <- setNames(rep(0, n_arms), arm_names)
  sums   <- setNames(rep(0, n_arms), arm_names)
  log_df <- data.frame()
  cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    if (runif(1) < epsilon || t <= n_arms) {
      arm <- sample(arm_names, 1)
    } else {
      avg_reward <- sums / counts
      arm <- names(avg_reward)[which.max(avg_reward)]
    }
    reward <- rbinom(1, 1, true_ctr[arm])
    counts[arm] <- counts[arm] + 1
    sums[arm]   <- sums[arm] + reward
    regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward
    cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(
      Round = t, Arm = arm, Reward = reward,
      Cumulative_Reward = cum_reward, Regret = regret,
      Cumulative_Regret = cum_regret, Algorithm_Used = "Epsilon-Greedy (eps=0.1)"
    ))
  }
  log_df
}

simulate_ucb1 <- function(n_rounds) {
  # ---- UCB1: pick the arm with the highest upper-confidence bound ----
  counts <- setNames(rep(0, n_arms), arm_names)
  sums   <- setNames(rep(0, n_arms), arm_names)
  log_df <- data.frame()
  cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    if (t <= n_arms) {
      arm <- arm_names[t]
    } else {
      avg_reward <- sums / counts
      bonus <- sqrt(2 * log(t) / counts)
      ucb_score <- avg_reward + bonus
      arm <- names(ucb_score)[which.max(ucb_score)]
    }
    reward <- rbinom(1, 1, true_ctr[arm])
    counts[arm] <- counts[arm] + 1
    sums[arm]   <- sums[arm] + reward
    regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward
    cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(
      Round = t, Arm = arm, Reward = reward,
      Cumulative_Reward = cum_reward, Regret = regret,
      Cumulative_Regret = cum_regret, Algorithm_Used = "UCB1"
    ))
  }
  log_df
}

simulate_thompson <- function(n_rounds) {
  # ---- Thompson Sampling: Beta-Bernoulli conjugate posterior per arm ----
  alpha <- setNames(rep(1, n_arms), arm_names)  # prior successes
  beta  <- setNames(rep(1, n_arms), arm_names)  # prior failures
  log_df <- data.frame()
  cum_reward <- 0; cum_regret <- 0
  for (t in 1:n_rounds) {
    sampled_theta <- sapply(arm_names, function(a) rbeta(1, alpha[a], beta[a]))
    arm <- names(sampled_theta)[which.max(sampled_theta)]
    reward <- rbinom(1, 1, true_ctr[arm])
    if (reward == 1) alpha[arm] <- alpha[arm] + 1 else beta[arm] <- beta[arm] + 1
    regret <- best_ctr - true_ctr[arm]
    cum_reward <- cum_reward + reward
    cum_regret <- cum_regret + regret
    log_df <- rbind(log_df, data.frame(
      Round = t, Arm = arm, Reward = reward,
      Cumulative_Reward = cum_reward, Regret = regret,
      Cumulative_Regret = cum_regret, Algorithm_Used = "Thompson Sampling"
    ))
  }
  log_df
}

## ---------------------------------------------------------------------
## 4. Run all 5 strategies and combine into one dataset
## ---------------------------------------------------------------------
cat("Running simulations...\n")
log_random   <- simulate_random(n_rounds)
log_greedy   <- simulate_greedy(n_rounds)
log_epsilon  <- simulate_epsilon_greedy(n_rounds, epsilon = 0.1)
log_ucb1     <- simulate_ucb1(n_rounds)
log_thompson <- simulate_thompson(n_rounds)

bandit_df <- bind_rows(log_random, log_greedy, log_epsilon, log_ucb1, log_thompson)

# Add contextual/descriptive columns (10th parameter set to match dataset spec)
bandit_df$Session_ID    <- 1:nrow(bandit_df)
bandit_df$Device_Type   <- sample(device_types, nrow(bandit_df), replace = TRUE)
bandit_df$Time_of_Day   <- sample(times_of_day, nrow(bandit_df), replace = TRUE)
bandit_df$True_CTR      <- true_ctr[bandit_df$Arm]

# Reorder columns for readability
bandit_df <- bandit_df[, c("Session_ID", "Round", "Algorithm_Used", "Arm",
                           "Device_Type", "Time_of_Day", "Reward", "True_CTR",
                           "Cumulative_Reward", "Regret", "Cumulative_Regret")]

cat(sprintf("Generated dataset: %d rows x %d columns\n", nrow(bandit_df), ncol(bandit_df)))

write.csv(bandit_df, "Bandit_Ad_CTR_Dataset.csv", row.names = FALSE)
cat("Saved: Bandit_Ad_CTR_Dataset.csv\n\n")

## ---------------------------------------------------------------------
## 5. Bandit / RL Terminology Summary Table
## ---------------------------------------------------------------------
summary_table <- data.frame(
  Metric = c(
    "Number of Arms (Ad Variants)",
    "Rounds per Algorithm",
    "Total Rounds (all algorithms combined)",
    "Best True Arm",
    "Best True CTR",
    "Algorithms Compared",
    "Exploration Strategy Spectrum",
    "Reward Type",
    "Regret Definition"
  ),
  Value = c(
    n_arms,
    n_rounds,
    nrow(bandit_df),
    best_arm,
    round(best_ctr, 3),
    paste(unique(bandit_df$Algorithm_Used), collapse = " | "),
    "Random (100% explore) -> Epsilon-Greedy -> UCB1 -> Thompson -> Greedy (100% exploit)",
    "Binary click (1 = click, 0 = no click)",
    "best_CTR - true_CTR(chosen arm), accumulated per round"
  )
)
cat("=== Bandit Case Study Summary Table ===\n")
print(summary_table, row.names = FALSE)
write.csv(summary_table, "Bandit_Summary_Table.csv", row.names = FALSE)

# Per-algorithm performance summary
algo_summary <- bandit_df %>%
  group_by(Algorithm_Used) %>%
  summarise(
    Total_Reward       = sum(Reward),
    Avg_Reward         = round(mean(Reward), 4),
    Final_Cum_Regret   = max(Cumulative_Regret),
    Pct_Best_Arm_Chosen = round(mean(Arm == best_arm) * 100, 1)
  ) %>%
  arrange(Final_Cum_Regret)

cat("\n=== Per-Algorithm Performance Summary ===\n")
print(algo_summary, row.names = FALSE)
write.csv(algo_summary, "Bandit_Algorithm_Performance.csv", row.names = FALSE)

## ---------------------------------------------------------------------
## 6. Visualizations
## ---------------------------------------------------------------------
theme_set(theme_minimal(base_size = 13))

# Figure 1: Cumulative Regret over Rounds (the KEY exploration/exploitation plot)
p1 <- ggplot(bandit_df, aes(x = Round, y = Cumulative_Regret, color = Algorithm_Used)) +
  geom_line(linewidth = 1) +
  labs(title = "Figure 1. Cumulative Regret Over Rounds",
       subtitle = "Lower is better -- shows how fast each strategy learns the best ad",
       x = "X-Axis: Round Number", y = "Y-Axis: Cumulative Regret", color = "Algorithm")
ggsave("fig1_cumulative_regret.png", p1, width = 9, height = 6, dpi = 150)

# Figure 2: Cumulative Reward (Clicks) over Rounds
p2 <- ggplot(bandit_df, aes(x = Round, y = Cumulative_Reward, color = Algorithm_Used)) +
  geom_line(linewidth = 1) +
  labs(title = "Figure 2. Cumulative Reward (Clicks) Over Rounds",
       x = "X-Axis: Round Number", y = "Y-Axis: Cumulative Reward (Clicks)", color = "Algorithm")
ggsave("fig2_cumulative_reward.png", p2, width = 9, height = 6, dpi = 150)

# Figure 3: Arm Selection Frequency per Algorithm
p3 <- ggplot(bandit_df, aes(x = Arm, fill = Arm)) +
  geom_bar() +
  facet_wrap(~Algorithm_Used) +
  labs(title = "Figure 3. Arm Selection Frequency by Algorithm",
       x = "X-Axis: Ad Variant (Arm)", y = "Y-Axis: Times Selected") +
  theme(legend.position = "none")
ggsave("fig3_arm_selection_frequency.png", p3, width = 10, height = 7, dpi = 150)

# Figure 4: Final Cumulative Regret Comparison (bar chart)
final_regret <- bandit_df %>% group_by(Algorithm_Used) %>%
  summarise(Final_Regret = max(Cumulative_Regret))
p4 <- ggplot(final_regret, aes(x = reorder(Algorithm_Used, Final_Regret), y = Final_Regret, fill = Algorithm_Used)) +
  geom_col() + coord_flip() +
  labs(title = "Figure 4. Final Cumulative Regret by Algorithm",
       x = "X-Axis: Algorithm", y = "Y-Axis: Final Cumulative Regret") +
  theme(legend.position = "none")
ggsave("fig4_final_regret_comparison.png", p4, width = 9, height = 6, dpi = 150)

# Figure 5: % of Rounds the Optimal Arm was Chosen (rolling, per algorithm)
rolling_optimal <- bandit_df %>%
  group_by(Algorithm_Used) %>%
  arrange(Round) %>%
  mutate(Is_Optimal = as.numeric(Arm == best_arm),
         Rolling_Optimal_Pct = cummean(Is_Optimal) * 100)
p5 <- ggplot(rolling_optimal, aes(x = Round, y = Rolling_Optimal_Pct, color = Algorithm_Used)) +
  geom_line(linewidth = 1) +
  labs(title = "Figure 5. Percent of Rounds the Optimal Ad was Chosen (Running Average)",
       x = "X-Axis: Round Number", y = "Y-Axis: % Optimal Arm Chosen", color = "Algorithm")
ggsave("fig5_optimal_arm_rate.png", p5, width = 9, height = 6, dpi = 150)

# Figure 6: Estimated CTR per Arm vs True CTR (final estimates, per algorithm)
estimated_ctr <- bandit_df %>%
  group_by(Algorithm_Used, Arm) %>%
  summarise(Estimated_CTR = mean(Reward), .groups = "drop") %>%
  left_join(data.frame(Arm = arm_names, True_CTR = as.numeric(true_ctr)), by = "Arm")
p6 <- ggplot(estimated_ctr, aes(x = Arm)) +
  geom_col(aes(y = Estimated_CTR, fill = "Estimated"), position = "dodge") +
  geom_point(aes(y = True_CTR, color = "True CTR"), size = 3) +
  facet_wrap(~Algorithm_Used) +
  labs(title = "Figure 6. Estimated CTR per Arm vs. True CTR",
       x = "X-Axis: Ad Variant (Arm)", y = "Y-Axis: Click-Through Rate",
       fill = "", color = "")
ggsave("fig6_estimated_vs_true_ctr.png", p6, width = 10, height = 7, dpi = 150)

cat("\nSaved 6 comparison figures (fig1_...png through fig6_...png)\n")
cat("Saved 3 CSV outputs: Bandit_Ad_CTR_Dataset.csv, Bandit_Summary_Table.csv, Bandit_Algorithm_Performance.csv\n")