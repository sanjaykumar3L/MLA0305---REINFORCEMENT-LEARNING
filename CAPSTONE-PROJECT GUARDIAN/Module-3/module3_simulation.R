#############################################################################
# PROJECT GUARDIAN - MODULE 3
# Hybrid ABM + DES Simulation with Reinforcement Learning (Tabular Q-Learning)
#
# ARCHITECTURE
#   DES layer  : incidents arrive as discrete events over simulated time.
#                Each event carries a severity level and the TRUE number
#                of rescue/medical units actually required to resolve it.
#   ABM layer  : the resource pool (rescue + medical units) is modelled
#                as a capacity-limited pool of autonomous agents - units
#                get dispatched (busy), then a portion return to the pool
#                as each incident resolves.
#   RL layer   : tabular Q-learning agent learns dispatch actions.
#                State  = (Severity Level) x (Resource Availability Bucket)
#                Action = number of units to dispatch (1 to 9)
#                Reward = dynamic reward function with interactive weights.
#
# Dispatch strategies run over the SAME incident stream:
#   "Existing System 1" -> fixed-rule heuristic dispatch
#   "Existing System 2" -> greedy immediate dispatch
#   "Proposed System"   -> Q-learning RL agent (learns optimal policy)
#############################################################################

library(shiny)
library(ggplot2)
library(DT)
library(reshape2)

plot_font_m3 <- "serif"
if (requireNamespace("extrafont", quietly = TRUE)) {
  tryCatch({
    extrafont::loadfonts(device = "win", quiet = TRUE)
    if ("Cambria" %in% extrafont::fonts()) plot_font_m3 <- "Cambria"
  }, error = function(e) NULL)
}

#############################################################################
## ------------------------- DES: INCIDENT STREAM --------------------------
#############################################################################
SEVERITY_LEVELS  <- c("Low", "Medium", "High", "Critical")
SEVERITY_WEIGHTS <- c(0.30, 0.35, 0.25, 0.10)
REQUIRED_UNITS_RANGE <- list(Low = 1:2, Medium = 2:4, High = 4:6, Critical = 6:9)
BASE_RESPONSE_TIME   <- list(Low = 10, Medium = 20, High = 35, Critical = 55)

generate_incident_stream <- function(n_incidents, seed = 1) {
  set.seed(seed)
  severity <- sample(SEVERITY_LEVELS, size = n_incidents, replace = TRUE, prob = SEVERITY_WEIGHTS)
  required_units <- sapply(severity, function(s) sample(REQUIRED_UNITS_RANGE[[s]], 1))
  base_time <- sapply(severity, function(s) BASE_RESPONSE_TIME[[s]])
  data.frame(
    Incident_ID = seq_len(n_incidents),
    Severity = factor(severity, levels = SEVERITY_LEVELS),
    Required_Units = required_units,
    Base_Response_Time = base_time
  )
}

#############################################################################
## ------------------------- DYNAMIC REWARD FUNCTION -------------------------
#############################################################################
compute_reward <- function(required, allocated, base_time, understaff_mult = 1.5, time_mult = 0.3) {
  mismatch_penalty <- abs(required - allocated) * 8
  if (allocated < required) mismatch_penalty <- mismatch_penalty * understaff_mult
  effective_time <- base_time + max(0, (required - allocated)) * 4
  time_penalty <- effective_time * time_mult
  reward <- 100 - mismatch_penalty - time_penalty
  max(reward, -50)
}

#############################################################################
## --------------- STRATEGY 1: EXISTING SYSTEM 1 (fixed heuristic) ---------
#############################################################################
heuristic_dispatch <- function(severity, available_units) {
  fixed_rule <- c(Low = 2, Medium = 3, High = 5, Critical = 7)
  min(fixed_rule[[as.character(severity)]], available_units)
}

run_heuristic <- function(incidents, total_pool, understaff_mult = 1.5, time_mult = 0.3) {
  n <- nrow(incidents)
  available_units <- total_pool
  cumulative_reward <- numeric(n)
  reward_log <- numeric(n)
  alloc_log  <- numeric(n)
  running_total <- 0
  
  for (i in seq_len(n)) {
    sev <- as.character(incidents$Severity[i])
    allocated <- heuristic_dispatch(sev, available_units)
    reward <- compute_reward(incidents$Required_Units[i], allocated, incidents$Base_Response_Time[i], understaff_mult, time_mult)
    
    available_units <- max(0, min(total_pool, available_units - allocated + round(allocated * 0.7)))
    
    running_total <- running_total + reward
    cumulative_reward[i] <- running_total
    reward_log[i] <- reward
    alloc_log[i]  <- allocated
  }
  list(cumulative_reward = cumulative_reward, reward_log = reward_log, alloc_log = alloc_log)
}

#############################################################################
## --------------- STRATEGY 2: EXISTING SYSTEM 2 (greedy dispatch) ---------
#############################################################################
run_greedy <- function(incidents, total_pool, understaff_mult = 1.5, time_mult = 0.3) {
  n <- nrow(incidents)
  available_units <- total_pool
  cumulative_reward <- numeric(n)
  reward_log <- numeric(n)
  alloc_log  <- numeric(n)
  running_total <- 0
  
  for (i in seq_len(n)) {
    req_u <- incidents$Required_Units[i]
    allocated <- min(req_u, available_units)
    reward <- compute_reward(req_u, allocated, incidents$Base_Response_Time[i], understaff_mult, time_mult)
    
    available_units <- max(0, min(total_pool, available_units - allocated + round(allocated * 0.7)))
    
    running_total <- running_total + reward
    cumulative_reward[i] <- running_total
    reward_log[i] <- reward
    alloc_log[i]  <- allocated
  }
  list(cumulative_reward = cumulative_reward, reward_log = reward_log, alloc_log = alloc_log)
}

#############################################################################
## ------------ STRATEGY 3: PROPOSED SYSTEM (tabular Q-learning) -----------
#############################################################################
ACTIONS <- 1:9
AVAIL_BUCKETS <- c("Low", "Medium", "High")

bucket_availability <- function(available, total_pool) {
  ratio <- available / total_pool
  if (ratio < 0.34) return("Low")
  if (ratio < 0.67) return("Medium")
  return("High")
}

init_q_table <- function() {
  states <- expand.grid(Severity = SEVERITY_LEVELS, Availability = AVAIL_BUCKETS, stringsAsFactors = FALSE)
  matrix(0, nrow = nrow(states), ncol = length(ACTIONS),
         dimnames = list(paste(states$Severity, states$Availability, sep = "_"), ACTIONS))
}

train_q_learning <- function(incidents, total_pool, alpha = 0.15, gamma = 0.85,
                              epsilon_start = 0.9, epsilon_end = 0.05,
                              understaff_mult = 1.5, time_mult = 0.3) {
  
  q_table <- init_q_table()
  n <- nrow(incidents)
  available_units <- total_pool
  cumulative_reward <- numeric(n)
  reward_log <- numeric(n)
  alloc_log  <- numeric(n)
  running_total <- 0
  
  for (i in seq_len(n)) {
    epsilon <- epsilon_start - (epsilon_start - epsilon_end) * (i / n)
    sev <- as.character(incidents$Severity[i])
    bucket <- bucket_availability(available_units, total_pool)
    state_key <- paste(sev, bucket, sep = "_")
    
    if (runif(1) < epsilon) {
      action_idx <- sample(seq_along(ACTIONS), 1)
    } else {
      action_idx <- which.max(q_table[state_key, ])
    }
    allocated <- min(ACTIONS[action_idx], available_units)
    
    reward <- compute_reward(incidents$Required_Units[i], allocated, incidents$Base_Response_Time[i], understaff_mult, time_mult)
    
    available_units <- max(0, min(total_pool, available_units - allocated + round(allocated * 0.7)))
    next_bucket <- bucket_availability(available_units, total_pool)
    next_state_key <- paste(sev, next_bucket, sep = "_")
    
    best_next <- max(q_table[next_state_key, ])
    q_table[state_key, action_idx] <- q_table[state_key, action_idx] +
      alpha * (reward + gamma * best_next - q_table[state_key, action_idx])
    
    running_total <- running_total + reward
    cumulative_reward[i] <- running_total
    reward_log[i] <- reward
    alloc_log[i]  <- allocated
  }
  
  list(q_table = q_table, cumulative_reward = cumulative_reward, reward_log = reward_log, alloc_log = alloc_log)
}

#############################################################################
## ------------------------- SHINY MODULE: UI -------------------------------
#############################################################################
mod3_simulation_ui <- function(id) {
  ns <- NS(id)
  
  tagList(
    h4("Module 3: Hybrid ABM + DES Simulation with Reinforcement Learning"),
    p("Incidents arrive as discrete events (DES). Rescue/Medical units behave as
      capacity-limited autonomous agents (ABM). A tabular Q-learning agent
      (Proposed System) learns how many units to dispatch per incident and is
      compared against baseline dispatchers on the exact same incident stream."),
    
    sidebarLayout(
      sidebarPanel(
        numericInput(ns("n_incidents"), "Number of incidents to simulate:",
                     value = 500, min = 100, max = 5000, step = 50),
        numericInput(ns("total_pool"), "Total resource pool (units available):",
                     value = 20, min = 5, max = 100, step = 1),
        numericInput(ns("sim_seed"), "Simulation seed:", value = 42, min = 1, step = 1),
        sliderInput(ns("understaff_mult"), "Under-Staffing Penalty Multiplier:",
                    min = 1.0, max = 3.0, value = 1.5, step = 0.1),
        sliderInput(ns("time_mult"), "Response Time Penalty Weight:",
                    min = 0.1, max = 1.0, value = 0.3, step = 0.05),
        actionButton(ns("run_sim"), "Run Simulation", icon = icon("play"),
                     class = "btn-generate", width = "100%")
      ),
      mainPanel(
        tabsetPanel(
          tabPanel("Learning Curve",
                   br(),
                   p("X-axis = incident number (simulated DES time). Y-axis = cumulative reward."),
                   plotOutput(ns("learning_curve"), height = "450px")),
          
          tabPanel("Final Performance Comparison",
                   br(),
                   p("Average reward per incident across the full run - higher is better."),
                   plotOutput(ns("performance_bar"), height = "400px"),
                   DTOutput(ns("performance_table"))),
          
          tabPanel("Learned Q-Table (Policy)",
                   br(),
                   p("State vs Action matrix learned by the RL agent."),
                   DTOutput(ns("q_table_view"))),
          
          tabPanel("Event Log Sample",
                   br(),
                   p("First 50 simulated incidents with strategy rewards side by side."),
                   DTOutput(ns("event_log")))
        )
      )
    )
  )
}

#############################################################################
## ----------------------- SHINY MODULE: SERVER -----------------------------
#############################################################################
mod3_simulation_server <- function(id) {
  moduleServer(id, function(input, output, session) {
    
    sim_results <- eventReactive(input$run_sim, {
      u_mult <- input$understaff_mult
      t_mult <- input$time_mult
      
      incidents <- generate_incident_stream(input$n_incidents, seed = input$sim_seed)
      
      set.seed(input$sim_seed)
      rl <- train_q_learning(incidents, total_pool = input$total_pool, understaff_mult = u_mult, time_mult = t_mult)
      
      set.seed(input$sim_seed)
      base <- run_heuristic(incidents, total_pool = input$total_pool, understaff_mult = u_mult, time_mult = t_mult)
      
      set.seed(input$sim_seed)
      greedy <- run_greedy(incidents, total_pool = input$total_pool, understaff_mult = u_mult, time_mult = t_mult)
      
      list(incidents = incidents, rl = rl, base = base, greedy = greedy)
    })
    
    output$learning_curve <- renderPlot({
      req(sim_results())
      res <- sim_results()
      n <- nrow(res$incidents)
      
      df <- data.frame(
        Incident_Number = rep(seq_len(n), 3),
        Cumulative_Reward = c(res$base$cumulative_reward, res$greedy$cumulative_reward, res$rl$cumulative_reward),
        Strategy = rep(c("Existing System 1 (Heuristic)", "Existing System 2 (Greedy)", "Proposed System (RL Q-Learning)"), each = n)
      )
      
      ggplot(df, aes(x = Incident_Number, y = Cumulative_Reward, color = Strategy)) +
        geom_line(linewidth = 1) +
        labs(title = "Cumulative Reward over Simulated Incidents",
             x = "Incident Number (Simulated Time / DES Clock)",
             y = "Cumulative Reward (higher = better)",
             color = "Dispatch Strategy") +
        scale_color_manual(values = c("Existing System 1 (Heuristic)" = "#d35400", "Existing System 2 (Greedy)" = "#7f8c8d", "Proposed System (RL Q-Learning)" = "#1a3c6e")) +
        theme_minimal(base_family = plot_font_m3) +
        theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    })
    
    output$performance_bar <- renderPlot({
      req(sim_results())
      res <- sim_results()
      
      perf <- data.frame(
        Strategy = c("Existing System 1 (Heuristic)", "Existing System 2 (Greedy)", "Proposed System (RL Q-Learning)"),
        Avg_Reward = c(mean(res$base$reward_log), mean(res$greedy$reward_log), mean(res$rl$reward_log))
      )
      perf$Avg_Reward <- round(perf$Avg_Reward, 3)
      
      ggplot(perf, aes(x = Strategy, y = Avg_Reward, fill = Strategy)) +
        geom_bar(stat = "identity", width = 0.5) +
        geom_text(aes(label = Avg_Reward), vjust = -0.4, family = plot_font_m3, size = 4.2) +
        scale_fill_manual(values = c("Existing System 1 (Heuristic)" = "#d35400", "Existing System 2 (Greedy)" = "#7f8c8d", "Proposed System (RL Q-Learning)" = "#1a3c6e")) +
        labs(title = "Average Reward per Incident: Existing vs Proposed",
             x = "Dispatch Strategy",
             y = "Average Reward per Incident (higher = better)") +
        theme_minimal(base_family = plot_font_m3) +
        theme(plot.title = element_text(face = "bold"), legend.position = "none")
    })
    
    output$performance_table <- renderDT({
      req(sim_results())
      res <- sim_results()
      perf <- data.frame(
        Metric = c("Average Reward per Incident", "Total Cumulative Reward",
                   "Best Incident Reward", "Worst Incident Reward"),
        Existing_System_1_Heuristic = c(
          round(mean(res$base$reward_log), 3),
          round(sum(res$base$reward_log), 1),
          round(max(res$base$reward_log), 2),
          round(min(res$base$reward_log), 2)
        ),
        Existing_System_2_Greedy = c(
          round(mean(res$greedy$reward_log), 3),
          round(sum(res$greedy$reward_log), 1),
          round(max(res$greedy$reward_log), 2),
          round(min(res$greedy$reward_log), 2)
        ),
        Proposed_System_RL = c(
          round(mean(res$rl$reward_log), 3),
          round(sum(res$rl$reward_log), 1),
          round(max(res$rl$reward_log), 2),
          round(min(res$rl$reward_log), 2)
        )
      )
      datatable(perf, options = list(dom = 't'), rownames = FALSE)
    })
    
    output$q_table_view <- renderDT({
      req(sim_results())
      q <- sim_results()$rl$q_table
      best_action <- apply(q, 1, function(row) ACTIONS[which.max(row)])
      out <- data.frame(
        State = rownames(q),
        round(q, 2),
        Best_Action_Units_To_Dispatch = best_action,
        check.names = FALSE
      )
      datatable(out, options = list(pageLength = 12, scrollX = TRUE), rownames = FALSE)
    })
    
    output$event_log <- renderDT({
      req(sim_results())
      res <- sim_results()
      n_show <- min(50, nrow(res$incidents))
      log_df <- data.frame(
        Incident_ID = res$incidents$Incident_ID[1:n_show],
        Severity = res$incidents$Severity[1:n_show],
        Required_Units = res$incidents$Required_Units[1:n_show],
        Existing_1_Reward = round(res$base$reward_log[1:n_show], 2),
        Existing_2_Reward = round(res$greedy$reward_log[1:n_show], 2),
        Proposed_RL_Reward = round(res$rl$reward_log[1:n_show], 2)
      )
      datatable(log_df, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
    })
    
    return(sim_results)
  })
}