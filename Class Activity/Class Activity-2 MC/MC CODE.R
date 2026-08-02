#=========================================================
# MONTE CARLO
#=========================================================
#
# METHOD: First-visit Monte Carlo Control with epsilon-greedy
#         exploration. Instead of computing V*(s) analytically
#         via the Bellman equation (value iteration), we SIMULATE
#         thousands of random episodes through the MDP, record the
#         rewards actually received, and average them to estimate
#         Q(s,a). The policy is improved episode-by-episode.
#=========================================================

setwd("C:/Users/lsk87/OneDrive/Desktop/inventory/r pro/activity")
library(knitr)
library(ggplot2)
library(patchwork)

set.seed(42)

cat("\n=========================================\n")
cat(" MONTE CARLO SIMULATION\n")
cat("=========================================\n")

#---------------------------------------------------------
# 1. Environment definition (states, actions, transitions, rewards)
#    Same environment as the value-iteration version, so results
#    are comparable.
#---------------------------------------------------------

states  <- c("S1", "S2", "S3")
actions <- c("A1", "A2")
gamma   <- 0.9

P <- list(
  A1 = matrix(c(
    0.2, 0.6, 0.2,
    0.5, 0.0, 0.5,
    0.5, 0.4, 0.1
  ), nrow = 3, byrow = TRUE, dimnames = list(states, states)),
  
  A2 = matrix(c(
    0.0, 0.2, 0.8,
    0.3, 0.2, 0.5,
    0.1, 0.6, 0.3
  ), nrow = 3, byrow = TRUE, dimnames = list(states, states))
)

reward_lookup <- c(
  "S1.A1.S1" = 0,  "S1.A1.S2" = 5,  "S1.A1.S3" = -1,
  "S1.A2.S1" = 0,  "S1.A2.S2" = 10, "S1.A2.S3" = -5,
  "S2.A1.S1" = 3,  "S2.A1.S2" = 0,  "S2.A1.S3" = 2,
  "S2.A2.S1" = 7,  "S2.A2.S2" = 0,  "S2.A2.S3" = 1,
  "S3.A1.S1" = 4,  "S3.A1.S2" = 0,  "S3.A1.S3" = 0,
  "S3.A2.S1" = 6,  "S3.A2.S2" = -2, "S3.A2.S3" = 0
)

cat("\nStates  :", paste(states,  collapse = ", "))
cat("\nActions :", paste(actions, collapse = ", "))
cat("\nDiscount factor (gamma) :", gamma, "\n")

#---------------------------------------------------------
# 2. Episode simulator
#    Draws a next state from P[[a]][s,], looks up the reward,
#    and steps forward for a fixed horizon.
#---------------------------------------------------------

episode_length <- 15

simulate_step <- function(s, a) {
  s2 <- sample(states, size = 1, prob = P[[a]][s, ])
  r  <- reward_lookup[paste(s, a, s2, sep = ".")]
  list(next_state = s2, reward = unname(r))
}

#---------------------------------------------------------
# 3. First-visit Monte Carlo Control (epsilon-greedy)
#---------------------------------------------------------

n_episodes <- 3000
epsilon    <- 0.2   # exploration rate

Q <- matrix(0, nrow = length(states), ncol = length(actions),
            dimnames = list(states, actions))
N <- matrix(0, nrow = length(states), ncol = length(actions),
            dimnames = list(states, actions))   # visit counts for averaging

epsilon_greedy_action <- function(s) {
  if (runif(1) < epsilon) {
    sample(actions, 1)
  } else {
    actions[which.max(Q[s, ])]
  }
}

# --- logging structures for the plots ---
return_log      <- numeric(n_episodes)              # total reward per episode
qhistory_log    <- vector("list", n_episodes)        # running Q snapshot each ep
visit_counter   <- setNames(rep(0, length(states)), states)
action_counter  <- matrix(0, nrow = length(states), ncol = length(actions),
                          dimnames = list(states, actions))
sample_traj     <- NULL   # we will capture one full trajectory for plotting

for (ep in 1:n_episodes) {
  
  s <- sample(states, 1)                 # random start state each episode
  traj_states  <- character(0)
  traj_actions <- character(0)
  traj_rewards <- numeric(0)
  
  for (t in 1:episode_length) {
    a <- epsilon_greedy_action(s)
    step <- simulate_step(s, a)
    
    visit_counter[s]      <- visit_counter[s] + 1
    action_counter[s, a]  <- action_counter[s, a] + 1
    
    traj_states  <- c(traj_states, s)
    traj_actions <- c(traj_actions, a)
    traj_rewards <- c(traj_rewards, step$reward)
    
    s <- step$next_state
  }
  
  # --- compute discounted returns G_t backward through the episode ---
  G <- 0
  G_t <- numeric(episode_length)
  for (t in episode_length:1) {
    G <- traj_rewards[t] + gamma * G
    G_t[t] <- G
  }
  return_log[ep] <- G_t[1]
  
  # --- first-visit MC update: only the FIRST occurrence of each (s,a) counts ---
  seen <- character(0)
  for (t in 1:episode_length) {
    key <- paste(traj_states[t], traj_actions[t])
    if (!(key %in% seen)) {
      seen <- c(seen, key)
      s_t <- traj_states[t]; a_t <- traj_actions[t]
      N[s_t, a_t] <- N[s_t, a_t] + 1
      Q[s_t, a_t] <- Q[s_t, a_t] + (G_t[t] - Q[s_t, a_t]) / N[s_t, a_t]
    }
  }
  
  qhistory_log[[ep]] <- as.vector(Q)   # snapshot for convergence plot
  
  # keep the very first episode as the "sample trajectory" to visualize
  if (ep == 1) {
    sample_traj <- data.frame(Step = 1:episode_length, State = traj_states,
                              Action = traj_actions, Reward = traj_rewards)
  }
}

policy_mc <- sapply(states, function(s) actions[which.max(Q[s, ])])

cat("\n====================================")
cat("\nMonte Carlo Q(s,a) estimates (after", n_episodes, "episodes)")
cat("\n====================================\n")
print(kable(round(Q, 2)))

cat("\n====================================")
cat("\nMonte Carlo Derived Policy  pi(s)")
cat("\n====================================\n")
print(kable(data.frame(State = states, BestAction = policy_mc)))

cat("\nMean return over all episodes:", round(mean(return_log), 2), "\n")

#---------------------------------------------------------
# 4. VISUALIZATION SUITE — entirely different chart types
#    from the value-iteration script
#---------------------------------------------------------

state_cols  <- c(S1 = "#DC2626", S2 = "#EA580C", S3 = "#16A34A")   # different palette
action_cols <- c(A1 = "#9333EA", A2 = "#0891B2")

theme_mc <- theme_minimal(base_size = 14) +
  theme(
    plot.title    = element_text(face = "bold", size = 16, margin = margin(b = 4)),
    plot.subtitle = element_text(color = "grey40", size = 11, margin = margin(b = 10)),
    axis.title    = element_text(face = "bold", size = 13),
    axis.text     = element_text(size = 11),
    panel.grid.minor = element_blank(),
    plot.margin   = margin(15, 20, 15, 15)
  )

## 4a. Sample trajectory plot (path through states over one episode) ----
p_traj <- ggplot(sample_traj, aes(Step, State, group = 1)) +
  geom_step(color = "grey60", linewidth = 0.8) +
  geom_point(aes(color = State, shape = Action), size = 5) +
  geom_text(aes(label = Reward), vjust = -1.6, size = 3.8, color = "grey30") +
  scale_color_manual(values = state_cols) +
  scale_x_continuous(breaks = 1:episode_length) +
  labs(title = "Sample Simulated Trajectory (Episode 1)",
       subtitle = "Point = state visited, label = reward received, shape = action taken",
       x = "Time Step", y = "State") +
  theme_mc +
  theme(legend.position = "right",
        legend.box = "vertical",
        legend.title = element_text(face = "bold"))

## 4b. Histogram of episode returns ---------------------------------------
p_hist <- ggplot(data.frame(Return = return_log), aes(Return)) +
  geom_histogram(bins = 40, fill = "#0891B2", color = "white", alpha = 0.85) +
  geom_vline(xintercept = mean(return_log), color = "#DC2626",
             linetype = "dashed", linewidth = 1) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Distribution of Episode Returns",
       subtitle = paste0("n = ", n_episodes, " episodes  |  mean return = ",
                         round(mean(return_log), 2), " (red dashed line)"),
       x = "Discounted Return G", y = "Number of Episodes") +
  theme_mc

## 4c. Q-value learning curves (running estimate vs. episode) ------------
qhist_df <- do.call(rbind, lapply(seq_along(qhistory_log), function(ep) {
  vals <- qhistory_log[[ep]]
  data.frame(Episode = ep,
             State  = rep(states, times = length(actions)),
             Action = rep(actions, each = length(states)),
             Q = vals)
}))
# thin to every 10th episode so the plot isn't overloaded
qhist_thin <- qhist_df[qhist_df$Episode %% 10 == 0, ]

p_learn <- ggplot(qhist_thin, aes(Episode, Q, color = Action)) +
  geom_line(linewidth = 1) +
  facet_wrap(~State, nrow = 1, labeller = label_both) +
  scale_color_manual(values = action_cols) +
  labs(title = "Monte Carlo Q(s,a) Learning Curves",
       subtitle = "Running average of first-visit returns as episodes accumulate (one panel per state)",
       x = "Episode", y = "Estimated Q(s,a)") +
  theme_mc +
  theme(strip.text = element_text(face = "bold", size = 12),
        strip.background = element_rect(fill = "grey92", color = NA),
        legend.position = "right",
        legend.title = element_text(face = "bold"),
        panel.spacing = unit(1.2, "lines"))

## 4d. State visitation bar chart -----------------------------------------
visit_df <- data.frame(State = states, Visits = as.numeric(visit_counter))
p_visits <- ggplot(visit_df, aes(State, Visits, fill = State)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = Visits), vjust = -0.5, size = 4.2, fontface = "bold") +
  scale_fill_manual(values = state_cols) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "State Visitation Counts", subtitle = "Across all simulated episodes",
       x = "State", y = "Times Visited") +
  theme_mc + theme(legend.position = "none")

## 4e. Action-selection bar chart (% of A1 vs A2 chosen, per state) -------
action_long <- as.data.frame(as.table(action_counter))
colnames(action_long) <- c("State", "Action", "Count")
action_long <- do.call(rbind, lapply(states, function(s) {
  sub <- action_long[action_long$State == s, ]
  sub$Percent <- round(100 * sub$Count / sum(sub$Count), 1)
  sub
}))

p_donut <- ggplot(action_long, aes(Action, Percent, fill = Action)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = paste0(Percent, "%")), vjust = -0.5, size = 4, fontface = "bold") +
  facet_wrap(~State, nrow = 1, labeller = label_both) +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.12))) +
  scale_fill_manual(values = action_cols) +
  labs(title = "Action Selection Frequency per State",
       subtitle = "Share of A1 vs A2 chosen during simulation (epsilon-greedy)",
       x = "Action", y = "% of Visits") +
  theme_mc +
  theme(strip.text = element_text(face = "bold", size = 12),
        strip.background = element_rect(fill = "grey92", color = NA),
        legend.position = "none",
        panel.spacing = unit(1.2, "lines"))

## 4f. Combined dashboard ---------------------------------------------------
#     Each row gets generous height; stacked single-column layout (rather
#     than cramming two plots side by side) so every axis/legend/title
#     stays fully readable at print size.
dashboard_mc <- p_traj / p_hist / p_learn / p_visits / p_donut +
  plot_layout(heights = c(1, 1, 1, 1, 1)) +
  plot_annotation(
    title = "Monte Carlo MDP Simulation Dashboard",
    subtitle = paste0(n_episodes, " episodes  |  episode length = ", episode_length,
                      "  |  epsilon = ", epsilon, "  |  gamma = ", gamma),
    theme = theme(plot.title = element_text(face = "bold", size = 22, margin = margin(b = 4)),
                  plot.subtitle = element_text(color = "grey40", size = 13, margin = margin(b = 10)))
  )

print(dashboard_mc)

# ---- save individual plots (these are the ones to actually inspect closely) ----
ggsave("mc_plot_trajectory.png",     p_traj,   width = 10, height = 5,   dpi = 200)
ggsave("mc_plot_return_hist.png",    p_hist,   width = 9,  height = 5,   dpi = 200)
ggsave("mc_plot_learning_curve.png", p_learn,  width = 13, height = 5.5, dpi = 200)
ggsave("mc_plot_state_visits.png",   p_visits, width = 7,  height = 5.5, dpi = 200)
ggsave("mc_plot_action_donut.png",   p_donut,  width = 10, height = 5.5, dpi = 200)

# combined dashboard: tall single-column canvas so nothing gets squeezed
ggsave("mc_dashboard.png", dashboard_mc, width = 13, height = 26, dpi = 200, limitsize = FALSE)

cat("\nSaved visualization PNGs:")
cat("\n  - mc_plot_trajectory.png")
cat("\n  - mc_plot_return_hist.png")
cat("\n  - mc_plot_learning_curve.png")
cat("\n  - mc_plot_state_visits.png")
cat("\n  - mc_plot_action_donut.png")
cat("\n  - mc_dashboard.png  (all combined, tall single-column layout)")
cat("\n(all files written to your current working directory: ", getwd(), ")\n")