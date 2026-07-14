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
# 6. Visualization  (SIMPLIFIED for readability)
#    Only the OPTIMAL action's edges are drawn from each
#    state. This is deliberately less "complete" than showing
#    every transition -- but far easier to read at a glance,
#    since the whole point of solving the MDP is to know what
#    to do, not to see every possible transition again.
#---------------------------------------------------------

# ---- palette: one colour per state --------------------------
node_fill  <- c(S1 = "#2563EB", S2 = "#7C3AED", S3 = "#059669")  # blue, violet, emerald
edge_col   <- "#B45309"   # amber/brown - the single "optimal path" colour

# ---- node definitions (state name + optimal value V* + best action) ----
node_lines <- sapply(states, function(s) {
  sprintf(
    '%s [label=<<B>%s</B><BR/><FONT POINT-SIZE="11">V* = %.2f</FONT><BR/><FONT POINT-SIZE="10">best: %s</FONT>>, fillcolor="%s"]',
    s, s, V[s], policy[s], node_fill[s]
  )
})

# ---- edge definitions: ONLY the optimal action's transitions -----------
edge_lines <- c()
for (s in states) {
  a <- policy[s]                 # the single best action for this state
  for (s2 in states) {
    prob <- P[[a]][s, s2]
    if (prob > 0.05) {           # drop near-zero edges to reduce clutter
      edge_lines <- c(edge_lines, sprintf(
        '%s -> %s [label=<<FONT COLOR="#854F0B"><B>%s : %.1f</B></FONT>>, color="%s", penwidth=%.1f, arrowsize=0.9, fontsize=12]',
        s, s2, a, prob, edge_col, 1.5 + prob * 3))   # thicker line = higher probability
    }
  }
}

graph_body <- paste(c(node_lines, edge_lines), collapse = "\n")

# ---- build the DOT source as a reusable function so Section 7 (export)
#      can regenerate the exact same graph without duplicating this code ----
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

  // ---- simple one-line legend ----
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
# 7. Export the diagram to PNG / PDF / SVG
#    grViz() only renders to the RStudio Viewer / browser by
#    default. Use DiagrammeRsvg + rsvg to save it as a file.
#---------------------------------------------------------

# install.packages(c("DiagrammeRsvg", "rsvg"))   # run once if not installed
library(DiagrammeRsvg)
library(rsvg)

# Reuse the same DOT source that was already rendered above - no duplication
mdp_graph <- grViz(build_mdp_dot())

# --- Convert to SVG text, then rasterize/save in whatever formats you need ---
svg_code <- export_svg(mdp_graph)

# SVG (vector, scales to any size, best for editing later)
writeLines(svg_code, "mdp_optimal_policy.svg")

# PNG (raster, good for slides/docs) - width controls resolution
rsvg_png(charToRaw(svg_code), file = "mdp_optimal_policy.png", width = 1600)

# PDF (vector, best for print / LaTeX reports)
rsvg_pdf(charToRaw(svg_code), file = "mdp_optimal_policy.pdf")

cat("\nSaved diagram as:")
cat("\n  - mdp_optimal_policy.svg")
cat("\n  - mdp_optimal_policy.png")
cat("\n  - mdp_optimal_policy.pdf")
cat("\n(files written to your current working directory: ", getwd(), ")\n")