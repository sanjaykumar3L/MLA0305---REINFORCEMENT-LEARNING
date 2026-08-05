#=========================================================
# MARKOV DECISION PROCESS (MDP) SIMULATION IN R  (v2)
# Adds: vectorized expected-reward calc, value iteration,
#       optimal policy extraction, and a policy-aware plot
#=========================================================

library(DiagrammeR)
library(knitr)

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
#    Stored in a named list -> P[["A1"]], P[["A2"]]
#    Each row must sum to 1 (checked below)
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

# --- sanity check: every row of every action matrix sums to 1 ---
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
# 3. Reward Table  (long format, same idea as before)
#---------------------------------------------------------

reward_table <- expand.grid(
  Current_State = states,
  Action         = actions,
  Next_State     = states,
  stringsAsFactors = FALSE
)

# Assign rewards (edit these values as needed)
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
#    Vectorized instead of nested loops
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
#    V(s) = max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) V(s') ]
#---------------------------------------------------------

V <- setNames(rep(0, length(states)), states)
theta <- 1e-6      # convergence threshold
max_iter <- 1000

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
  if (delta < theta) {
    cat("\nValue iteration converged after", iter, "iterations.\n")
    break
  }
}

# Extract greedy optimal policy from converged V
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
# 6. Visualization
#    All transitions shown faintly; edges belonging to the
#    OPTIMAL action from each state are highlighted.
#---------------------------------------------------------

edge_lines <- c()

for (s in states) {
  for (a in actions) {
    is_best <- (policy[s] == a)
    for (s2 in states) {
      prob <- P[[a]][s, s2]
      if (prob > 0) {
        if (is_best) {
          style <- sprintf(
            "%s -> %s [label='%s : %.1f', color='forestgreen', penwidth=3]",
            s, s2, a, prob)
        } else {
          style <- sprintf(
            "%s -> %s [label='%s : %.1f', color='gray80', style=dashed, fontcolor='gray60']",
            s, s2, a, prob)
        }
        edge_lines <- c(edge_lines, style)
      }
    }
  }
}

graph_body <- paste(edge_lines, collapse = "\n")

grViz(sprintf("
digraph MDP {

graph [layout = dot, rankdir = LR]

node [shape=circle, style=filled, fontcolor=white, fontsize=18]

S1 [fillcolor=red]
S2 [fillcolor=blue]
S3 [fillcolor=green]

%s

}
", graph_body))

cat("\nGreen solid edges = transitions under the optimal action for that state.")
cat("\nGray dashed edges = transitions under the non-optimal action.\n")