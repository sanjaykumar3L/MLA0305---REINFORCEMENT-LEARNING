#############################################################################
# PROJECT GUARDIAN - MODULE 5
# Model Comparison & Multi-Graph Visualization Module
#
# ARCHITECTURE
#   Models Compared : 
#     1. Existing Model 1 -> Fixed-Rule Heuristic Dispatcher
#     2. Existing Model 2 -> Greedy / Un-optimized Dispatcher
#     3. Proposed Model   -> Hybrid RL (Tabular Q-Learning + ABM + DES)
#
#   Visualizations  :
#     - Cumulative Reward Convergence Trajectory (Line Chart)
#     - Moving Average Response Time (Line Chart)
#     - Head-to-Head Average Reward with Standard Error Bars (Bar Chart)
#     - Empirical CDF (eCDF) Stochastic Dominance Curve Plot (Line Plot)
#     - 6-Axis Polar Radar / Spider Chart (Polar Plot)
#     - Violin Plot + Boxplot Overlay (Distribution Plot)
#     - Scatter Workload vs Response Time with Trendlines (Scatter Plot)
#     - Response Time by Disaster Severity Level (Grouped Bar Chart)
#     - Resource Mismatch Rate (%) by Severity Level (Grouped Bar Chart)
#     - Multi-Metric System Performance Index (Comparative Bar Chart)
#
# USAGE (from main Source code.R):
#   source("../Module-5/module5_comparison.R")
#   ... inside top_tabs tabsetPanel:
#       tabPanel("Module 5: Model Comparison", mod5_comparison_ui("comp5"))
#   ... inside server body:
#       mod5_comparison_server("comp5", sim_results = sim_results)
#############################################################################

library(shiny)
library(ggplot2)
library(DT)
library(reshape2)

## Cambria font attempt (falls back silently to serif)
plot_font_m5 <- "serif"
if (requireNamespace("extrafont", quietly = TRUE)) {
  tryCatch({
    extrafont::loadfonts(device = "win", quiet = TRUE)
    if ("Cambria" %in% extrafont::fonts()) plot_font_m5 <- "Cambria"
  }, error = function(e) NULL)
}

#############################################################################
## ------------------------- SHINY MODULE: UI -------------------------------
#############################################################################
mod5_comparison_ui <- function(id) {
  ns <- NS(id)
  
  tagList(
    h4("Module 5: Benchmark Model Comparison & Multi-Graph Visualizations"),
    p("Compares the Proposed Hybrid Model (RL Q-Learning + ABM + DES) against two established baseline models:
      (1) Fixed-Rule Heuristic Dispatcher and (2) Greedy Dispatcher across multiple graphical visualizations and performance parameters."),
    
    uiOutput(ns("comp_content"))
  )
}

#############################################################################
## ----------------------- SHINY MODULE: SERVER -----------------------------
#############################################################################
mod5_comparison_server <- function(id, sim_results) {
  moduleServer(id, function(input, output, session) {
    
    output$comp_content <- renderUI({
      if (is.null(sim_results())) {
        tagList(
          div(class = "well",
              h4("Module 5 is waiting for simulation data"),
              p("Please run the simulation in ", strong("Module 3: Simulation"), " first. Once executed, this tab will automatically populate with multi-model comparison charts and benchmark reports.")
          )
        )
      } else {
        ns <- session$ns
        tagList(
          tabsetPanel(
            tabPanel("Head-to-Head Overview",
                     br(),
                     h4("3-Model Head-to-Head Performance Summary"),
                     p("Compares overall reward, response speed, allocation accuracy, and win-rate percentage across all 3 models."),
                     DTOutput(ns("head_to_head_table")),
                     br(),
                     downloadButton(ns("download_comparison_csv"), "Export Comparison Report (.csv)", class = "btn-export"),
                     br(), br(),
                     h4("Average Reward Head-to-Head (with Standard Error Bars)"),
                     plotOutput(ns("reward_bar_se"), height = "420px")
            ),
            
            tabPanel("Dynamic Trajectory & Convergence",
                     br(),
                     h4("1. Cumulative Reward Learning Trajectory (Line Chart)"),
                     p("Tracks long-term reward accumulation over simulated incident arrival sequence (DES clock)."),
                     plotOutput(ns("cumulative_line_chart"), height = "400px"),
                     br(),
                     h4("2. Moving Average Response Time Trajectory (Window = 20 Incidents)"),
                     p("Shows real-time reduction in effective response delay as the RL agent adapts."),
                     plotOutput(ns("moving_avg_time_chart"), height = "400px")
            ),
            
            tabPanel("Empirical CDF & Stochastic Dominance",
                     br(),
                     h4("Empirical Cumulative Distribution Function (eCDF) Plot"),
                     p("Measures risk and failure probability. A curve further to the right demonstrates First-Order Stochastic Dominance (superior performance under operational uncertainty)."),
                     plotOutput(ns("ecdf_plot"), height = "450px")
            ),
            
            tabPanel("Polar Radar & Violin Distributions",
                     br(),
                     h4("1. 6-Axis Polar Radar / Spider Chart Comparison"),
                     p("Multi-dimensional radar comparison across 6 operational capabilities."),
                     plotOutput(ns("polar_radar_chart"), height = "450px"),
                     br(),
                     h4("2. Violin Plot + Boxplot Overlay (Distribution Shape & Density)"),
                     p("Visualizes median, quartiles, and multimodal distribution density across the 3 models."),
                     plotOutput(ns("violin_boxplot_chart"), height = "450px")
            ),
            
            tabPanel("Workload Scaling & Severity Charts",
                     br(),
                     h4("1. Response Delay vs Workload Demand (Scatter Plot with Smoothing Trendlines)"),
                     p("Tracks how response delay scales as required units increase from 1 to 9."),
                     plotOutput(ns("workload_scatter_chart"), height = "420px"),
                     br(),
                     h4("2. Effective Response Time by Disaster Severity Level (Grouped Bar Chart)"),
                     plotOutput(ns("severity_response_chart"), height = "420px"),
                     br(),
                     h4("3. Resource Mismatch Rate (%) by Disaster Severity Level (Grouped Bar Chart)"),
                     plotOutput(ns("severity_mismatch_chart"), height = "420px")
            ),
            
            tabPanel("Multi-Metric Performance Profile",
                     br(),
                     h4("Normalized Performance Index across System Dimensions"),
                     p("Compares normalized scores (0 - 100) across core operational dimensions."),
                     plotOutput(ns("multi_metric_profile_chart"), height = "460px")
            )
          )
        )
      }
    })
    
    ## ---- Computation helper reactive ----
    comp_data <- reactive({
      req(sim_results())
      res <- sim_results()
      
      incidents <- res$incidents
      n <- nrow(incidents)
      
      reward_base   <- res$base$reward_log
      reward_greedy <- res$greedy$reward_log
      reward_rl     <- res$rl$reward_log
      
      cum_base   <- res$base$cumulative_reward
      cum_greedy <- res$greedy$cumulative_reward
      cum_rl     <- res$rl$cumulative_reward
      
      alloc_base   <- if (!is.null(res$base$alloc_log)) res$base$alloc_log else sapply(seq_len(n), function(i) heuristic_dispatch(as.character(incidents$Severity[i]), 20))
      alloc_greedy <- if (!is.null(res$greedy$alloc_log)) res$greedy$alloc_log else alloc_base
      alloc_rl     <- if (!is.null(res$rl$alloc_log)) res$rl$alloc_log else alloc_base
      
      req_units <- incidents$Required_Units
      base_time <- incidents$Base_Response_Time
      
      eff_time_base   <- base_time + max(0, (req_units - alloc_base)) * 4
      eff_time_greedy <- base_time + max(0, (req_units - alloc_greedy)) * 4
      eff_time_rl     <- base_time + max(0, (req_units - alloc_rl)) * 4
      
      mismatch_base   <- abs(req_units - alloc_base)
      mismatch_greedy <- abs(req_units - alloc_greedy)
      mismatch_rl     <- abs(req_units - alloc_rl)
      
      list(
        incidents = incidents,
        reward_base = reward_base,
        reward_greedy = reward_greedy,
        reward_rl = reward_rl,
        cum_base = cum_base,
        cum_greedy = cum_greedy,
        cum_rl = cum_rl,
        eff_time_base = eff_time_base,
        eff_time_greedy = eff_time_greedy,
        eff_time_rl = eff_time_rl,
        mismatch_base = mismatch_base,
        mismatch_greedy = mismatch_greedy,
        mismatch_rl = mismatch_rl
      )
    })
    
    ## ---- Head-to-Head Table ----
    output$head_to_head_table <- renderDT({
      req(comp_data())
      d <- comp_data()
      n <- nrow(d$incidents)
      
      rl_wins <- sum(d$reward_rl > d$reward_base & d$reward_rl > d$reward_greedy)
      base_wins <- sum(d$reward_base >= d$reward_rl & d$reward_base >= d$reward_greedy)
      greedy_wins <- sum(d$reward_greedy >= d$reward_rl & d$reward_greedy > d$reward_base)
      
      df_h2h <- data.frame(
        Performance_Metric = c(
          "Average Reward per Incident",
          "Reward Standard Deviation",
          "Total Cumulative Reward",
          "Mean Effective Response Time (Min)",
          "Root Mean Squared Error (RMSE)",
          "Mean Absolute Error (MAE)",
          "Best Performing Incidents (Count)",
          "Win Rate Percentage (%)"
        ),
        Model_1_Fixed_Heuristic = c(
          round(mean(d$reward_base), 3),
          round(sd(d$reward_base), 3),
          round(sum(d$reward_base), 1),
          round(mean(d$eff_time_base), 2),
          round(sqrt(mean(d$mismatch_base^2)), 3),
          round(mean(d$mismatch_base), 3),
          base_wins,
          round(100 * base_wins / n, 2)
        ),
        Model_2_Greedy_Dispatcher = c(
          round(mean(d$reward_greedy), 3),
          round(sd(d$reward_greedy), 3),
          round(sum(d$reward_greedy), 1),
          round(mean(d$eff_time_greedy), 2),
          round(sqrt(mean(d$mismatch_greedy^2)), 3),
          round(mean(d$mismatch_greedy), 3),
          greedy_wins,
          round(100 * greedy_wins / n, 2)
        ),
        Proposed_Model_Hybrid_RL = c(
          round(mean(d$reward_rl), 3),
          round(sd(d$reward_rl), 3),
          round(sum(d$reward_rl), 1),
          round(mean(d$eff_time_rl), 2),
          round(sqrt(mean(d$mismatch_rl^2)), 3),
          round(mean(d$mismatch_rl), 3),
          rl_wins,
          round(100 * rl_wins / n, 2)
        )
      )
      
      datatable(df_h2h, options = list(dom = 't', pageLength = 10), rownames = FALSE)
    })
    
    ## Download comparison CSV
    output$download_comparison_csv <- downloadHandler(
      filename = function() paste0("guardian_model_comparison_report_", Sys.Date(), ".csv"),
      content = function(file) {
        req(comp_data())
        d <- comp_data()
        n <- nrow(d$incidents)
        
        rl_wins <- sum(d$reward_rl > d$reward_base & d$reward_rl > d$reward_greedy)
        base_wins <- sum(d$reward_base >= d$reward_rl & d$reward_base >= d$reward_greedy)
        greedy_wins <- sum(d$reward_greedy >= d$reward_rl & d$reward_greedy > d$reward_base)
        
        df_h2h <- data.frame(
          Performance_Metric = c(
            "Average Reward per Incident",
            "Reward Standard Deviation",
            "Total Cumulative Reward",
            "Mean Effective Response Time (Min)",
            "Root Mean Squared Error (RMSE)",
            "Mean Absolute Error (MAE)",
            "Best Performing Incidents (Count)",
            "Win Rate Percentage (%)"
          ),
          Model_1_Fixed_Heuristic = c(
            round(mean(d$reward_base), 3),
            round(sd(d$reward_base), 3),
            round(sum(d$reward_base), 1),
            round(mean(d$eff_time_base), 2),
            round(sqrt(mean(d$mismatch_base^2)), 3),
            round(mean(d$mismatch_base), 3),
            base_wins,
            round(100 * base_wins / n, 2)
          ),
          Model_2_Greedy_Dispatcher = c(
            round(mean(d$reward_greedy), 3),
            round(sd(d$reward_greedy), 3),
            round(sum(d$reward_greedy), 1),
            round(mean(d$eff_time_greedy), 2),
            round(sqrt(mean(d$mismatch_greedy^2)), 3),
            round(mean(d$mismatch_greedy), 3),
            greedy_wins,
            round(100 * greedy_wins / n, 2)
          ),
          Proposed_Model_Hybrid_RL = c(
            round(mean(d$reward_rl), 3),
            round(sd(d$reward_rl), 3),
            round(sum(d$reward_rl), 1),
            round(mean(d$eff_time_rl), 2),
            round(sqrt(mean(d$mismatch_rl^2)), 3),
            round(mean(d$mismatch_rl), 3),
            rl_wins,
            round(100 * rl_wins / n, 2)
          )
        )
        write.csv(df_h2h, file, row.names = FALSE)
      }
    )
    
    ## ---- Bar Chart with SE Error Bars ----
    output$reward_bar_se <- renderPlot({
      req(comp_data())
      d <- comp_data()
      n <- length(d$reward_base)
      
      df_bar <- data.frame(
        Model = c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"),
        Mean_Reward = c(mean(d$reward_base), mean(d$reward_greedy), mean(d$reward_rl)),
        SE = c(sd(d$reward_base)/sqrt(n), sd(d$reward_greedy)/sqrt(n), sd(d$reward_rl)/sqrt(n))
      )
      df_bar$Mean_Reward <- round(df_bar$Mean_Reward, 3)
      
      ggplot(df_bar, aes(x = Model, y = Mean_Reward, fill = Model)) +
        geom_bar(stat = "identity", width = 0.5, alpha = 0.9) +
        geom_errorbar(aes(ymin = Mean_Reward - SE, ymax = Mean_Reward + SE), width = 0.15, linewidth = 0.8) +
        geom_text(aes(label = Mean_Reward), vjust = -0.6, family = plot_font_m5, size = 4.2) +
        scale_fill_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        labs(
          title = "Average Incident Reward Comparison (with Standard Error Bars)",
          x = "Model Strategy",
          y = "Average Reward per Incident (higher = better)"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "none")
    })
    
    ## ---- Cumulative Line Chart ----
    output$cumulative_line_chart <- renderPlot({
      req(comp_data())
      d <- comp_data()
      n <- nrow(d$incidents)
      
      df_cum <- data.frame(
        Incident_Number = rep(seq_len(n), 3),
        Cumulative_Reward = c(d$cum_base, d$cum_greedy, d$cum_rl),
        Model = rep(c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"), each = n)
      )
      
      ggplot(df_cum, aes(x = Incident_Number, y = Cumulative_Reward, color = Model, group = Model)) +
        geom_line(linewidth = 1.1) +
        scale_color_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        labs(
          title = "Cumulative Reward Learning Trajectory over Simulated Incidents",
          x = "Incident Number (Simulated DES Clock)",
          y = "Cumulative Reward",
          color = "Model Strategy"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    })
    
    ## ---- Moving Average Response Time Line Chart ----
    output$moving_avg_time_chart <- renderPlot({
      req(comp_data())
      d <- comp_data()
      n <- nrow(d$incidents)
      w <- min(20, n)
      
      ma <- function(x, k) { as.numeric(stats::filter(x, rep(1/k, k), sides = 1)) }
      
      df_ma <- data.frame(
        Incident_Number = rep(seq_len(n), 3),
        Moving_Avg_Time = c(ma(d$eff_time_base, w), ma(d$eff_time_greedy, w), ma(d$eff_time_rl, w)),
        Model = rep(c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"), each = n)
      )
      df_ma <- na.omit(df_ma)
      
      ggplot(df_ma, aes(x = Incident_Number, y = Moving_Avg_Time, color = Model, group = Model)) +
        geom_line(linewidth = 1.1) +
        scale_color_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        labs(
          title = paste0("Moving Average Effective Response Time (Window = ", w, " Incidents)"),
          x = "Incident Number (Simulated DES Clock)",
          y = "Effective Response Time (Minutes, lower = better)",
          color = "Model Strategy"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    })
    
    ## ---- Empirical CDF (eCDF) Plot ----
    output$ecdf_plot <- renderPlot({
      req(comp_data())
      d <- comp_data()
      
      df_ecdf <- data.frame(
        Reward = c(d$reward_base, d$reward_greedy, d$reward_rl),
        Model = rep(c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"), each = length(d$reward_base))
      )
      
      ggplot(df_ecdf, aes(x = Reward, color = Model)) +
        stat_ecdf(linewidth = 1.1) +
        scale_color_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        labs(
          title = "Empirical Cumulative Distribution Function (eCDF) - First-Order Stochastic Dominance",
          x = "Incident Reward Value",
          y = "Cumulative Probability P(Reward <= x)",
          color = "Model Strategy"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    })
    
    ## ---- Polar Radar Chart ----
    output$polar_radar_chart <- renderPlot({
      req(comp_data())
      d <- comp_data()
      
      score_rew <- function(r) pmax(0, pmin(100, (r - 0) / (80 - 0) * 100))
      score_time <- function(t) pmax(0, pmin(100, (60 - t) / (60 - 10) * 100))
      score_error <- function(e) pmax(0, pmin(100, (5 - e) / (5 - 0) * 100))
      score_var <- function(v) pmax(0, pmin(100, (30 - sqrt(v)) / 30 * 100))
      
      df_radar <- data.frame(
        Capability = rep(c("Reward Efficiency", "Response Speed", "Allocation Precision", "Consistency", "Critical Severity", "Resilience"), 3),
        Model = rep(c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"), each = 6),
        Value = c(
          score_rew(mean(d$reward_base)), score_time(mean(d$eff_time_base)), score_error(mean(d$mismatch_base)), score_var(var(d$reward_base)), 40, 50,
          score_rew(mean(d$reward_greedy)), score_time(mean(d$eff_time_greedy)), score_error(mean(d$mismatch_greedy)), score_var(var(d$reward_greedy)), 35, 45,
          score_rew(mean(d$reward_rl)), score_time(mean(d$eff_time_rl)), score_error(mean(d$mismatch_rl)), score_var(var(d$reward_rl)), 85, 90
        )
      )
      
      ggplot(df_radar, aes(x = Capability, y = Value, color = Model, group = Model, fill = Model)) +
        geom_polygon(alpha = 0.25, linewidth = 1) +
        geom_point(size = 2.5) +
        coord_polar() +
        scale_color_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        scale_fill_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        ylim(0, 100) +
        labs(
          title = "6-Axis Polar Radar Profile Comparison",
          color = "Model Strategy", fill = "Model Strategy"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    })
    
    ## ---- Violin + Boxplot Overlay Chart ----
    output$violin_boxplot_chart <- renderPlot({
      req(comp_data())
      d <- comp_data()
      
      df_vio <- data.frame(
        Reward = c(d$reward_base, d$reward_greedy, d$reward_rl),
        Model = rep(c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"), each = length(d$reward_base))
      )
      
      ggplot(df_vio, aes(x = Model, y = Reward, fill = Model)) +
        geom_violin(alpha = 0.5, trim = FALSE) +
        geom_boxplot(width = 0.18, color = "black", alpha = 0.8, outlier.size = 1) +
        scale_fill_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        labs(
          title = "Violin Plot + Boxplot Overlay: Reward Distribution Shape & Density",
          x = "Model Strategy",
          y = "Incident Reward Value"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "none")
    })
    
    ## ---- Workload Scatter Plot with Trendlines ----
    output$workload_scatter_chart <- renderPlot({
      req(comp_data())
      d <- comp_data()
      
      df_scat <- data.frame(
        Workload = rep(d$incidents$Required_Units, 3),
        Response_Time = c(d$eff_time_base, d$eff_time_greedy, d$eff_time_rl),
        Model = rep(c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"), each = length(d$eff_time_base))
      )
      
      ggplot(df_scat, aes(x = Workload, y = Response_Time, color = Model)) +
        geom_jitter(alpha = 0.3, width = 0.15) +
        geom_smooth(method = "lm", se = TRUE, linewidth = 1) +
        scale_color_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        labs(
          title = "Effective Response Delay vs Workload Demand (1 to 9 Required Units)",
          x = "Incident Workload Demand (Required Units)",
          y = "Effective Response Time (Minutes, lower = better)",
          color = "Model Strategy"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    })
    
    ## ---- Grouped Bar: Response Time by Severity ----
    output$severity_response_chart <- renderPlot({
      req(comp_data())
      d <- comp_data()
      
      df_sev <- data.frame(
        Severity = rep(d$incidents$Severity, 3),
        Response_Time = c(d$eff_time_base, d$eff_time_greedy, d$eff_time_rl),
        Model = rep(c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"), each = length(d$eff_time_base))
      )
      
      agg <- aggregate(Response_Time ~ Severity + Model, data = df_sev, FUN = mean)
      agg$Response_Time <- round(agg$Response_Time, 1)
      
      ggplot(agg, aes(x = Severity, y = Response_Time, fill = Model)) +
        geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
        geom_text(aes(label = Response_Time), position = position_dodge(width = 0.7), vjust = -0.4, family = plot_font_m5, size = 3.5) +
        scale_fill_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        labs(
          title = "Mean Effective Response Time by Disaster Severity Level",
          x = "Disaster Severity Level",
          y = "Mean Response Time (Minutes, lower = better)",
          fill = "Model Strategy"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    })
    
    ## ---- Grouped Bar: Mismatch Error by Severity ----
    output$severity_mismatch_chart <- renderPlot({
      req(comp_data())
      d <- comp_data()
      
      df_mis <- data.frame(
        Severity = rep(d$incidents$Severity, 3),
        Mismatch = c(d$mismatch_base, d$mismatch_greedy, d$mismatch_rl),
        Model = rep(c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"), each = length(d$mismatch_base))
      )
      
      agg <- aggregate(Mismatch ~ Severity + Model, data = df_mis, FUN = mean)
      agg$Mismatch <- round(agg$Mismatch, 2)
      
      ggplot(agg, aes(x = Severity, y = Mismatch, fill = Model)) +
        geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
        geom_text(aes(label = Mismatch), position = position_dodge(width = 0.7), vjust = -0.4, family = plot_font_m5, size = 3.5) +
        scale_fill_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        labs(
          title = "Mean Allocation Mismatch (Units) by Disaster Severity Level",
          x = "Disaster Severity Level",
          y = "Mean Unit Mismatch (Lower = More Precise Allocation)",
          fill = "Model Strategy"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    })
    
    ## ---- Multi-Metric Profile Chart ----
    output$multi_metric_profile_chart <- renderPlot({
      req(comp_data())
      d <- comp_data()
      
      score_rew <- function(r) pmax(0, pmin(100, (r - 0) / (80 - 0) * 100))
      score_time <- function(t) pmax(0, pmin(100, (60 - t) / (60 - 10) * 100))
      score_error <- function(e) pmax(0, pmin(100, (5 - e) / (5 - 0) * 100))
      
      df_prof <- data.frame(
        Dimension = rep(c("Reward Score", "Response Speed", "Allocation Precision", "Consistency Score"), 3),
        Model = rep(c("Model 1: Fixed Heuristic", "Model 2: Greedy Dispatcher", "Proposed: Hybrid RL"), each = 4),
        Score = c(
          score_rew(mean(d$reward_base)), score_time(mean(d$eff_time_base)), score_error(mean(d$mismatch_base)), pmax(0, 100 - sd(d$reward_base)),
          score_rew(mean(d$reward_greedy)), score_time(mean(d$eff_time_greedy)), score_error(mean(d$mismatch_greedy)), pmax(0, 100 - sd(d$reward_greedy)),
          score_rew(mean(d$reward_rl)), score_time(mean(d$eff_time_rl)), score_error(mean(d$mismatch_rl)), pmax(0, 100 - sd(d$reward_rl))
        )
      )
      df_prof$Score <- round(df_prof$Score, 1)
      
      ggplot(df_prof, aes(x = Dimension, y = Score, fill = Model)) +
        geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
        geom_text(aes(label = Score), position = position_dodge(width = 0.7), vjust = -0.4, family = plot_font_m5, size = 3.5) +
        ylim(0, 115) +
        scale_fill_manual(values = c("Model 1: Fixed Heuristic" = "#d35400", "Model 2: Greedy Dispatcher" = "#7f8c8d", "Proposed: Hybrid RL" = "#1a3c6e")) +
        labs(
          title = "Multi-Metric System Performance Index (0 - 100 Scale)",
          x = "Operational System Dimension",
          y = "Normalized Score (higher = superior performance)",
          fill = "Model Strategy"
        ) +
        theme_minimal(base_family = plot_font_m5) +
        theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    })
    
  })
}
