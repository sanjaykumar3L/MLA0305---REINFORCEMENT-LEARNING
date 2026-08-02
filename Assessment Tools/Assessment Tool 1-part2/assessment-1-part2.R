#=========================================================
# MARKOV DECISION PROCESS (MDP) SIMULATION IN R  (v3)
# v2 additions: vectorized expected-reward calc, value
#               iteration, optimal policy extraction,
#               policy-aware DiagrammeR plot
# v3 additions (THIS VERSION): heavier focus on VISUALIZATION
#   - value-iteration convergence plot (V(s) vs iteration)
#   - expected immediate reward R(s,a) bar chart
#   - transition-probability heatmaps (one per action)
#   - optimal value function bar chart
#   - policy summary plot (best action per state)
#   - all plots arranged into one combined dashboard PNG
#   - every plot also saved individually to disk
#=========================================================

library(DiagrammeR)
library(knitr)
library(ggplot2)
library(reshape2)   # melt() for heatmaps
library(patchwork)  # combine ggplots into one dashboard

cat("\n=========================================\n")
cat(" MARKOV DECISION PROCESS (MDP)\n")
cat("=========================================\n")

#---------------------------------------------------------
# 1. States and Actions
#---------------------------------------------------------

states  <- c("S1", "S2", "S3")
actions <- c("A1", "A2")
gamma   <- 0.9   # discount factor for value iteration

cat("\nStates  :", paste(states,  collapse = ", "))
cat("\nActions :", paste(actions, collapse = ", "))
cat("\nDiscount factor (gamma) :", gamma, "\n")

#---------------------------------------------------------
# 2. Transition Probability Matrices
#---------------------------------------------------------

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

for (a in actions) {
  row_sums <- round(rowSums(P[[a]]), 6)
  if (any(row_sums != 1)) {
    stop(paste0("Transition matrix for action ", a,
                " has rows that do not sum to 1: ",
                paste(row_sums, collapse = ", ")))
  }
}
cat("\nTransition matrices validated: all rows sum to 1.\n")

for (a in actions) {
  cat("\n====================================")
  cat("\nTransition Probability Matrix (", a, ")")
  cat("\n====================================\n")
  print(P[[a]])
}

#---------------------------------------------------------
# 3. Reward Table
#---------------------------------------------------------

reward_table <- expand.grid(
  Current_State = states,
  Action         = actions,
  Next_State     = states,
  stringsAsFactors = FALSE
)

reward_lookup <- c(
  "S1.A1.S1" = 0,  "S1.A1.S2" = 5,  "S1.A1.S3" = -1,
  "S1.A2.S1" = 0,  "S1.A2.S2" = 10, "S1.A2.S3" = -5,
  "S2.A1.S1" = 3,  "S2.A1.S2" = 0,  "S2.A1.S3" = 2,
  "S2.A2.S1" = 7,  "S2.A2.S2" = 0,  "S2.A2.S3" = 1,
  "S3.A1.S1" = 4,  "S3.A1.S2" = 0,  "S3.A1.S3" = 0,
  "S3.A2.S1" = 6,  "S3.A2.S2" = -2, "S3.A2.S3" = 0
)

reward_table$Reward <- reward_lookup[
  paste(reward_table$Current_State, reward_table$Action,
        reward_table$Next_State, sep = ".")
]

cat("\n====================================")
cat("\nReward Table")
cat("\n====================================\n")
print(kable(reward_table[order(reward_table$Current_State,
                               reward_table$Action), ]))

#---------------------------------------------------------
# 4. Expected Immediate Reward  R(s,a) = sum_s' P(s'|s,a) * R(s,a,s')
#---------------------------------------------------------

expected_reward <- function(s, a) {
  r_row <- reward_lookup[paste(s, a, states, sep = ".")]
  p_row <- P[[a]][s, ]
  sum(p_row * r_row)
}

summary_table <- do.call(rbind, lapply(states, function(s) {
  do.call(rbind, lapply(actions, function(a) {
    data.frame(State = s, Action = a,
               ExpectedReward = round(expected_reward(s, a), 2))
  }))
}))

cat("\n====================================")
cat("\nExpected Immediate Reward  R(s,a)")
cat("\n====================================\n")
print(kable(summary_table))

#---------------------------------------------------------
# 5. Value Iteration -> Optimal Value Function & Policy
#    NOW TRACKS FULL HISTORY of V(s) at every iteration so
#    we can plot convergence afterwards.
#---------------------------------------------------------

V <- setNames(rep(0, length(states)), states)
theta <- 1e-6
max_iter <- 1000

# history: one row per iteration per state, for the convergence plot
V_history <- data.frame(Iteration = 0, State = states, Value = V,
                        row.names = NULL)

for (iter in 1:max_iter) {
  V_new <- V
  for (s in states) {
    action_values <- sapply(actions, function(a) {
      expected_reward(s, a) + gamma * sum(P[[a]][s, ] * V[states])
    })
    V_new[s] <- max(action_values)
  }
  delta <- max(abs(V_new - V))
  V <- V_new
  
  V_history <- rbind(V_history,
                     data.frame(Iteration = iter, State = states,
                                Value = V, row.names = NULL))
  
  if (delta < theta) {
    cat("\nValue iteration converged after", iter, "iterations.\n")
    break
  }
}

policy <- sapply(states, function(s) {
  action_values <- sapply(actions, function(a) {
    expected_reward(s, a) + gamma * sum(P[[a]][s, ] * V[states])
  })
  actions[which.max(action_values)]
})

cat("\n====================================")
cat("\nOptimal State Values  V*(s)")
cat("\n====================================\n")
print(kable(data.frame(State = states, Value = round(V, 3))))

cat("\n====================================")
cat("\nOptimal Policy  pi*(s)")
cat("\n====================================\n")
print(kable(data.frame(State = states, BestAction = policy)))

#---------------------------------------------------------
# 6. VISUALIZATION SUITE (ggplot2)
#    6a. Value-iteration convergence line plot
#    6b. Expected immediate reward R(s,a) bar chart
#    6c. Transition-probability heatmaps (per action)
#    6d. Optimal value function V*(s) bar chart
#    6e. Optimal policy summary plot
#    6f. Combined dashboard (patchwork) saved as one PNG
#---------------------------------------------------------

state_cols <- c(S1 = "#2563EB", S2 = "#7C3AED", S3 = "#059669")

theme_mdp <- theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(color = "grey40", size = 10),
    panel.grid.minor = element_blank()
  )

## 6a. Convergence plot: V(s) across iterations ------------------------
p_convergence <- ggplot(V_history, aes(Iteration, Value, color = State)) +
  geom_line(linewidth = 1.1) +
  geom_point(data = subset(V_history, Iteration == max(Iteration)),
             size = 2.5) +
  scale_color_manual(values = state_cols) +
  labs(title = "Value Iteration Convergence",
       subtitle = paste0("gamma = ", gamma, "  |  converged in ",
                         max(V_history$Iteration), " iterations"),
       x = "Iteration", y = "V(s)") +
  theme_mdp

## 6b. Expected immediate reward R(s,a) bar chart -----------------------
p_reward <- ggplot(summary_table, aes(State, ExpectedReward, fill = Action)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = ExpectedReward),
            position = position_dodge(width = 0.7), vjust = -0.4, size = 3.2) +
  scale_fill_manual(values = c(A1 = "#F59E0B", A2 = "#0EA5E9")) +
  labs(title = "Expected Immediate Reward  R(s,a)",
       x = NULL, y = "Expected Reward") +
  theme_mdp

## 6c. Transition-probability heatmaps (one panel per action) ----------
trans_long <- do.call(rbind, lapply(actions, function(a) {
  m <- melt(P[[a]], varnames = c("From", "To"), value.name = "Prob")
  m$Action <- a
  m
}))

p_heatmap <- ggplot(trans_long, aes(To, From, fill = Prob)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.2f", Prob)), size = 3.5,
            color = ifelse(trans_long$Prob > 0.5, "white", "black")) +
  facet_wrap(~Action, labeller = label_both) +
  scale_fill_gradient(low = "#EFF6FF", high = "#1D4ED8", limits = c(0, 1)) +
  labs(title = "Transition Probability Matrices P(s'|s,a)",
       x = "Next State", y = "Current State") +
  theme_mdp +
  theme(panel.grid = element_blank())

## 6d. Optimal value function V*(s) bar chart ---------------------------
value_df <- data.frame(State = states, Value = round(V, 3))
p_value <- ggplot(value_df, aes(State, Value, fill = State)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = Value), vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = state_cols) +
  labs(title = "Optimal State Values  V*(s)", x = NULL, y = "V*(s)") +
  theme_mdp + theme(legend.position = "none")

## 6e. Optimal policy summary plot --------------------------------------
policy_df <- data.frame(State = states, BestAction = policy)
p_policy <- ggplot(policy_df, aes(State, y = 1, fill = BestAction)) +
  geom_tile(width = 0.9, height = 0.9, color = "white", linewidth = 1) +
  geom_text(aes(label = BestAction), color = "white", fontface = "bold", size = 5) +
  scale_fill_manual(values = c(A1 = "#F59E0B", A2 = "#0EA5E9")) +
  labs(title = "Optimal Policy  pi*(s)", x = NULL, y = NULL) +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
        legend.position = "bottom")

## 6f. Combine everything into one dashboard PNG -------------------------
dashboard <- (p_convergence + p_value) /
  (p_reward + p_policy) /
  p_heatmap +
  plot_layout(heights = c(1, 1, 1.1)) +
  plot_annotation(
    title = "MDP Solution Dashboard",
    subtitle = paste0("States: ", paste(states, collapse = ", "),
                      "  |  Actions: ", paste(actions, collapse = ", ")),
    theme = theme(plot.title = element_text(face = "bold", size = 18),
                  plot.subtitle = element_text(color = "grey40"))
  )

print(dashboard)

# ---- save each plot individually + the combined dashboard ----
ggsave("plot_value_convergence.png", p_convergence, width = 7, height = 4.5, dpi = 150)
ggsave("plot_expected_reward.png",   p_reward,      width = 7, height = 4.5, dpi = 150)
ggsave("plot_transition_heatmap.png",p_heatmap,     width = 8, height = 4.5, dpi = 150)
ggsave("plot_optimal_values.png",    p_value,       width = 6, height = 4.5, dpi = 150)
ggsave("plot_optimal_policy.png",    p_policy,      width = 6, height = 3,   dpi = 150)
ggsave("mdp_dashboard.png",          dashboard,     width = 12, height = 14, dpi = 150)

cat("\nSaved visualization PNGs:")
cat("\n  - plot_value_convergence.png")
cat("\n  - plot_expected_reward.png")
cat("\n  - plot_transition_heatmap.png")
cat("\n  - plot_optimal_values.png")
cat("\n  - plot_optimal_policy.png")
cat("\n  - mdp_dashboard.png  (all combined)\n")

#---------------------------------------------------------
# 7. Policy Diagram (DiagrammeR) -- optimal action per state
#---------------------------------------------------------

node_fill <- state_cols
edge_col  <- "#B45309"

node_lines <- sapply(states, function(s) {
  sprintf(
    '%s [label=<<B>%s</B><BR/><FONT POINT-SIZE="11">V* = %.2f</FONT><BR/><FONT POINT-SIZE="10">best: %s</FONT>>, fillcolor="%s"]',
    s, s, V[s], policy[s], node_fill[s]
  )
})

edge_lines <- c()
for (s in states) {
  a <- policy[s]
  for (s2 in states) {
    prob <- P[[a]][s, s2]
    if (prob > 0.05) {
      edge_lines <- c(edge_lines, sprintf(
        '%s -> %s [label=<<FONT COLOR="#854F0B"><B>%s : %.1f</B></FONT>>, color="%s", penwidth=%.1f, arrowsize=0.9, fontsize=12]',
        s, s2, a, prob, edge_col, 1.5 + prob * 3))
    }
  }
}

graph_body <- paste(c(node_lines, edge_lines), collapse = "\n")

build_mdp_dot <- function() {
  sprintf('
digraph MDP {

  graph [
    layout = dot,
    rankdir = LR,
    bgcolor = "#F8FAFC",
    fontname = "Helvetica",
    label = <<BR/><FONT POINT-SIZE="20"><B>Optimal MDP Policy</B></FONT><BR/><FONT POINT-SIZE="12" COLOR="#64748B">Each arrow = the best action from that state   |   gamma = %.1f</FONT><BR/>>,
    labelloc = t,
    fontsize = 20,
    pad = 0.4,
    nodesep = 0.7,
    ranksep = 1.0
  ]

  node [
    shape = circle,
    style = "filled",
    fontcolor = white,
    fontname = "Helvetica",
    fontsize = 16,
    width = 1.3,
    penwidth = 2,
    color = "#1E293B"
  ]

  edge [
    fontname = "Helvetica",
    color = "%s",
    fontcolor = "#854F0B",
    curved = true
  ]

  %s

  legend [
    shape = plaintext,
    fontsize = 11,
    fontcolor = "#854F0B",
    label = "-> arrow = optimal action   |   thicker = higher probability"
  ]

}
', gamma, edge_col, graph_body)
}

grViz(build_mdp_dot())

cat("\nEach arrow shows the single OPTIMAL action to take from that state,")
cat("\nlabeled with the action name and transition probability.")
cat("\nLine thickness scales with probability. Each node also shows its")
cat("\noptimal value V*(s) and best action.\n")

#---------------------------------------------------------
# 8. Export the DiagrammeR diagram to PNG / PDF / SVG
#---------------------------------------------------------

library(DiagrammeRsvg)
library(rsvg)

mdp_graph <- grViz(build_mdp_dot())
svg_code <- export_svg(mdp_graph)

writeLines(svg_code, "mdp_optimal_policy.svg")
rsvg_png(charToRaw(svg_code), file = "mdp_optimal_policy.png", width = 1600)
rsvg_pdf(charToRaw(svg_code), file = "mdp_optimal_policy.pdf")

cat("\nSaved diagram as:")
cat("\n  - mdp_optimal_policy.svg")
cat("\n  - mdp_optimal_policy.png")
cat("\n  - mdp_optimal_policy.pdf")
cat("\n(all files written to your current working directory: ", getwd(), ")\n")