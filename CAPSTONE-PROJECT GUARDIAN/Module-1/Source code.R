#############################################################################
# PROJECT GUARDIAN
# Intelligent Disaster Response and Emergency Resource Coordination
# using Reinforcement Learning and Hybrid Agent-Based Simulation
#
# TASK 1 : SYNTHETIC DATA GENERATION MODULE (R Shiny Desktop App)
#
# Method   : Gaussian Mixture Model (GMM) based synthetic generation.
#            Each mixture component = one disaster severity regime
#            (Low / Medium / High / Critical). Continuous features are
#            drawn from a multivariate normal per component (MASS::mvrnorm),
#            which is exactly what a GMM generator does once mixture
#            weights, means and covariances are fixed.
#
# STRUCTURE (as requested):
#   MODULE 1 : Data Generation  -> Raw Preview, Distribution Graphs,
#              Descriptive Statistics, Normality Tests (Shapiro-Wilk),
#              ANOVA across severity groups, Correlation matrix + p-values.
#              Always visible.
#   MODULE 2 : Preprocessing & Export -> stays HIDDEN until a dataset has
#              been generated in Module 1. Unlocks automatically the
#              moment "Generate Dataset" is clicked.
#
# Required packages:
# install.packages(c("shiny","shinyjs","MASS","ggplot2","gridExtra",
#                     "DT","psych","reshape2"))
# Optional (for true Cambria font rendering inside plots):
# install.packages("extrafont")   # then run extrafont::font_import() once
#############################################################################

library(shiny)
library(shinyjs)
library(MASS)        # mvrnorm() -> multivariate normal draws per GMM component
library(ggplot2)
library(gridExtra)
library(DT)
library(psych)        # describe(), corr.test() -> descriptive & correlation stats
library(reshape2)     # melt() for faceted / long-format plots

## ---------------------------------------------------------------------
## Attempt to register Cambria for plot rendering. Falls back to serif
## silently if the font isn't installed, so the app never crashes.
## ---------------------------------------------------------------------
plot_font <- "serif"
if (requireNamespace("extrafont", quietly = TRUE)) {
  tryCatch({
    extrafont::loadfonts(device = "win", quiet = TRUE)
    if ("Cambria" %in% extrafont::fonts()) plot_font <- "Cambria"
  }, error = function(e) NULL)
}

FEATURE_NAMES <- c(
  "Population_Affected",
  "Available_Rescue_Units",
  "Available_Medical_Units",
  "Response_Time_Minutes",
  "Distance_to_Hospital_KM",
  "Infrastructure_Damage_Percent",
  "Weather_Severity_Index",
  "Communication_Network_Status_Percent",
  "Resource_Allocation_Efficiency_Score"
)

## ---------------------------------------------------------------------
## Dynamic p-value formatter: NEVER uses scientific notation, and never
## collapses a tiny-but-nonzero p-value into a flat "0.00000". It grows
## the number of decimal places shown just far enough to reveal the
## first significant digit(s) of the real value, capped at max_digits
## purely so the table stays readable (double-precision floats can't
## carry more than ~17 significant digits of real information anyway).
## Example: 0.00000032 -> "0.000000320000" instead of "0.00000".
## ---------------------------------------------------------------------
format_pval <- function(p, min_digits = 5, max_digits = 20) {
  vapply(p, function(x) {
    if (is.na(x)) return(NA_character_)
    if (x <= 0) return(formatC(0, format = "f", digits = max_digits))
    needed <- max(min_digits, ceiling(-log10(x)) + 4)
    needed <- min(needed, max_digits)
    formatC(x, format = "f", digits = needed)
  }, character(1))
}

#############################################################################
## ------------------------- GMM DATA GENERATOR -------------------------
#############################################################################
generate_guardian_data <- function(n_records, seed = 123) {
  
  set.seed(seed)
  p <- length(FEATURE_NAMES)
  
  mix_weights <- c(Low = 0.30, Medium = 0.35, High = 0.25, Critical = 0.10)
  
  means <- list(
    Low      = c(500,   60, 40,  15,  8,  10, 2, 90, 85),
    Medium   = c(3000,  45, 30,  30, 15,  35, 4, 70, 65),
    High     = c(12000, 25, 15,  55, 25,  60, 7, 45, 45),
    Critical = c(40000, 10,  6,  90, 40,  85, 9, 20, 25)
  )
  
  base_corr <- matrix(c(
    1.00,-0.55,-0.55, 0.60, 0.50, 0.65, 0.40,-0.60,-0.55,
    -0.55, 1.00, 0.70,-0.45,-0.35,-0.50,-0.30, 0.45, 0.50,
    -0.55, 0.70, 1.00,-0.45,-0.35,-0.50,-0.30, 0.45, 0.50,
    0.60,-0.45,-0.45, 1.00, 0.55, 0.55, 0.35,-0.55,-0.50,
    0.50,-0.35,-0.35, 0.55, 1.00, 0.50, 0.30,-0.45,-0.40,
    0.65,-0.50,-0.50, 0.55, 0.50, 1.00, 0.45,-0.60,-0.55,
    0.40,-0.30,-0.30, 0.35, 0.30, 0.45, 1.00,-0.35,-0.30,
    -0.60, 0.45, 0.45,-0.55,-0.45,-0.60,-0.35, 1.00, 0.65,
    -0.55, 0.50, 0.50,-0.50,-0.40,-0.55,-0.30, 0.65, 1.00
  ), nrow = p, byrow = TRUE)
  
  sd_scale <- list(
    Low      = c(150,  8, 6,  4, 2,  4, 0.6,  6, 6),
    Medium   = c(800, 10, 7,  6, 3,  6, 0.8,  8, 7),
    High     = c(2500,12, 5,  8, 4,  7, 0.9,  9, 8),
    Critical = c(6000, 4, 2, 10, 5,  6, 0.5,  7, 6)
  )
  
  n_each <- round(n_records * mix_weights)
  diff <- n_records - sum(n_each)
  n_each[which.max(n_each)] <- n_each[which.max(n_each)] + diff
  
  pieces <- list()
  for (lvl in names(mix_weights)) {
    D <- diag(sd_scale[[lvl]])
    Sigma <- D %*% base_corr %*% D
    Sigma <- (Sigma + t(Sigma)) / 2
    draw <- MASS::mvrnorm(n = n_each[[lvl]], mu = means[[lvl]], Sigma = Sigma)
    draw <- as.data.frame(draw)
    colnames(draw) <- FEATURE_NAMES
    draw$Disaster_Severity_Level <- lvl
    pieces[[lvl]] <- draw
  }
  
  df <- do.call(rbind, pieces)
  
  df$Population_Affected                     <- pmax(round(df$Population_Affected), 10)
  df$Available_Rescue_Units                  <- pmax(round(df$Available_Rescue_Units), 0)
  df$Available_Medical_Units                 <- pmax(round(df$Available_Medical_Units), 0)
  df$Response_Time_Minutes                   <- pmax(round(df$Response_Time_Minutes, 1), 2)
  df$Distance_to_Hospital_KM                 <- pmax(round(df$Distance_to_Hospital_KM, 1), 0.5)
  df$Infrastructure_Damage_Percent           <- pmin(pmax(round(df$Infrastructure_Damage_Percent, 1), 0), 100)
  df$Weather_Severity_Index                  <- pmin(pmax(round(df$Weather_Severity_Index, 1), 0), 10)
  df$Communication_Network_Status_Percent    <- pmin(pmax(round(df$Communication_Network_Status_Percent, 1), 0), 100)
  df$Resource_Allocation_Efficiency_Score    <- pmin(pmax(round(df$Resource_Allocation_Efficiency_Score, 1), 0), 100)
  
  df$Event_ID <- paste0("EVT-", sprintf("%05d", seq_len(nrow(df))))
  df$Disaster_Severity_Level <- factor(df$Disaster_Severity_Level,
                                       levels = c("Low","Medium","High","Critical"))
  
  df <- df[, c("Event_ID","Disaster_Severity_Level", FEATURE_NAMES)]
  rownames(df) <- NULL
  df <- df[sample(nrow(df)), ]
  rownames(df) <- NULL
  
  df
}

#############################################################################
## ----------------------------- USER INTERFACE ---------------------------
#############################################################################

ui <- fluidPage(
  
  useShinyjs(),
  
  tags$head(
    tags$style(HTML("
      @import url('https://fonts.cdnfonts.com/css/cambria');
      * { font-family: 'Cambria', Georgia, serif !important; }
      body { background-color: #f5f6fa; }
      h2, h3, h4 { color: #1a3c6e; font-weight: bold; }
      .well { background-color: #ffffff; border: 1px solid #d0d7e5; }
      .btn-generate { background-color: #1a3c6e; color: white; font-weight: bold; }
      .btn-arrow    { background-color: #d35400; color: white; font-weight: bold; font-size: 16px; }
      .btn-export   { background-color: #1e8449; color: white; font-weight: bold; }
      .module-badge { display:inline-block; background:#1a3c6e; color:#fff; padding:2px 10px;
                       border-radius: 10px; font-size: 13px; margin-bottom:6px; }
      .module-badge-2 { background:#d35400; }
      table.dataTable, table.dataTable td, table.dataTable th {
        font-family: 'Cambria', Georgia, serif !important;
      }
    "))
  ),
  
  titlePanel(
    div(
      h2("Project Guardian - Synthetic Disaster Response Dataset Generator"),
      h4("Reinforcement Learning & Hybrid Agent-Based Simulation | GMM-based Data Engine")
    )
  ),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      div(class = "module-badge", "MODULE 1"),
      h4("Data Generation"),
      
      radioButtons("record_choice", "Number of records to generate:",
                   choices = c("500" = "500", "600" = "600",
                               "1000" = "1000", "Custom" = "custom"),
                   selected = "600"),
      
      conditionalPanel(
        condition = "input.record_choice == 'custom'",
        numericInput("custom_n", "Enter custom record count (500 - 1000):",
                     value = 700, min = 500, max = 1000, step = 10)
      ),
      
      numericInput("seed_val", "Random seed (for reproducibility):",
                   value = 123, min = 1, step = 1),
      
      actionButton("gen_btn", "Generate Dataset", icon = icon("cogs"),
                   class = "btn-generate", width = "100%"),
      
      hr(),
      
      ## ---- Module 2 sidebar: stays hidden until data is generated ----
      shinyjs::hidden(
        div(id = "module2_sidebar",
            div(class = "module-badge module-badge-2", "MODULE 2"),
            h4("Modelling"),
            
            p("Run preprocessing (missing-value handling, min-max normalization,
            severity label-encoding), then export either file."),
            actionButton("preprocess_btn", HTML('Preprocess &nbsp; <i class="fa fa-arrow-right"></i>'),
                         class = "btn-arrow", width = "100%"),
            br(), br(),
            downloadButton("download_raw", "Export Raw Dataset (.csv)", class = "btn-export"),
            br(), br(),
            downloadButton("download_preprocessed", "Export Preprocessed Dataset (.csv)", class = "btn-export")
        )
      )
    ),
    
    mainPanel(
      width = 9,
      tabsetPanel(
        id = "top_tabs",
        
        tabPanel("Module 1: Data Generation",
                 tabsetPanel(
                   tabPanel("Dataset Summary",
                            br(),
                            h4("Quick Dataset Summary"),
                            p("A snapshot of the generated dataset before you dive into the detailed views."),
                            DTOutput("summary_table")),
                   
                   tabPanel("Raw Data Preview",
                            br(), DTOutput("raw_table")),
                   
                   tabPanel("Distribution Graphs",
                            br(),
                            h4("Distribution of each numeric parameter"),
                            p("X-axis = feature name & value range. Y-axis = density (proportion of records)."),
                            plotOutput("dist_plots", height = "900px")),
                   
                   tabPanel("Descriptive Statistics",
                            br(),
                            h4("Full statistical summary (per numeric variable)"),
                            p("Includes mean, SD, variance, median, MAD, trimmed mean, min, max,
                       range, IQR, skewness, kurtosis, standard error, and 95% confidence interval."),
                            DTOutput("stats_table"),
                            br(),
                            h4("Severity-level distribution (counts & proportion)"),
                            DTOutput("severity_table")),
                   
                   tabPanel("Normality Tests",
                            br(),
                            h4("Shapiro-Wilk Test of Normality (per variable)"),
                            p("H0: the variable is normally distributed. p < 0.05 rejects normality."),
                            DTOutput("shapiro_table")),
                   
                   tabPanel("ANOVA (Severity Effect)",
                            br(),
                            h4("One-way ANOVA: does Disaster_Severity_Level significantly affect each variable?"),
                            p("H0: group means are equal across severity levels. p < 0.05 = significant effect."),
                            DTOutput("anova_table")),
                   
                   tabPanel("Correlation Analysis",
                            br(),
                            h4("Pearson correlation matrix with significance (p-values)"),
                            p("r = correlation coefficient (-1 to 1). p < 0.05 marked significant.
                       Axes on the heatmap show the two variables being compared."),
                            plotOutput("corr_heatmap", height = "550px"),
                            br(),
                            DTOutput("corr_table"))
                 )
        ),
        
        tabPanel("Module 2: Modelling",
                 br(),
                 uiOutput("module2_main_ui"))
      )
    )
  )
)

#############################################################################
## ------------------------------- SERVER ----------------------------------
#############################################################################

server <- function(input, output, session) {
  
  raw_data          <- reactiveVal(NULL)
  preprocessed_data <- reactiveVal(NULL)
  
  ## Module 2 starts locked
  hideTab(inputId = "top_tabs", target = "Module 2: Modelling")
  
  ## ---- STEP 1: Generate data (Module 1) ----
  observeEvent(input$gen_btn, {
    n <- if (input$record_choice == "custom") input$custom_n else as.numeric(input$record_choice)
    n <- max(500, min(1000, round(n)))
    
    df <- generate_guardian_data(n_records = n, seed = input$seed_val)
    raw_data(df)
    preprocessed_data(NULL)
    
    ## Unlock Module 2
    shinyjs::show("module2_sidebar")
    showTab(inputId = "top_tabs", target = "Module 2: Modelling")
    
    showNotification(paste("Generated", nrow(df), "records. Module 2 is now unlocked."),
                     type = "message", duration = 4)
  })
  
  ## ---- Quick Dataset Summary (key characteristics, not raw records) ----
  output$summary_table <- renderDT({
    req(raw_data())
    df <- raw_data()
    
    n <- nrow(df)
    train_n <- round(0.8 * n)
    test_n  <- n - train_n
    
    missing_count <- sum(is.na(df))
    ## Duplicate check excludes Event_ID (which is unique-by-design) so it
    ## reflects true duplicate *feature* rows, not just duplicate IDs.
    dup_count <- sum(duplicated(df[, setdiff(names(df), "Event_ID")]))
    
    sev_counts <- table(df$Disaster_Severity_Level)
    sev_breakdown <- paste0(names(sev_counts), " (", as.integer(sev_counts), ")", collapse = ", ")
    
    summary_df <- data.frame(
      Parameter = c(
        "Total Incidents (Records)",
        "Generation Method",
        "Disaster Severity Levels",
        "Severity Level Breakdown (count)",
        "Total Columns (ID + Label + Features)",
        "Numeric Features",
        "Categorical Features",
        "Target Variable (for Modelling)",
        "Missing Values",
        "Duplicate Records",
        "Random Seed Used",
        "Planned Training Samples (80%)",
        "Planned Testing Samples (20%)",
        "Mean Population Affected",
        "Mean Response Time (Minutes)",
        "Mean Resource Allocation Efficiency Score",
        "Dataset Generated On"
      ),
      Value = c(
        format(n, big.mark = ","),
        "Gaussian Mixture Model (4 severity components)",
        paste(levels(df$Disaster_Severity_Level), collapse = ", "),
        sev_breakdown,
        ncol(df),
        length(FEATURE_NAMES),
        1,
        "Resource_Allocation_Efficiency_Score",
        missing_count,
        dup_count,
        input$seed_val,
        format(train_n, big.mark = ","),
        format(test_n, big.mark = ","),
        format(round(mean(df$Population_Affected), 1), big.mark = ","),
        round(mean(df$Response_Time_Minutes), 2),
        round(mean(df$Resource_Allocation_Efficiency_Score), 2),
        as.character(Sys.Date())
      ),
      stringsAsFactors = FALSE
    )
    
    datatable(summary_df, options = list(dom = 't', pageLength = 20), rownames = FALSE)
  })
  
  output$raw_table <- renderDT({
    req(raw_data())
    datatable(raw_data(), options = list(pageLength = 10, scrollX = TRUE))
  })
  
  ## ---- Distribution plots ----
  output$dist_plots <- renderPlot({
    req(raw_data())
    df <- raw_data()
    
    plots <- lapply(FEATURE_NAMES, function(col) {
      ggplot(df, aes(x = .data[[col]], fill = Disaster_Severity_Level)) +
        geom_histogram(aes(y = after_stat(density)), bins = 25,
                       alpha = 0.6, position = "identity", color = "white") +
        geom_density(alpha = 0.25, linewidth = 0.6) +
        labs(
          title = gsub("_", " ", col),
          x = paste0(gsub("_", " ", col), " (value)"),
          y = "Density (proportion of records)",
          fill = "Severity Level"
        ) +
        theme_minimal(base_family = plot_font) +
        theme(plot.title = element_text(face = "bold", size = 12),
              axis.title = element_text(size = 10),
              legend.position = "bottom")
    })
    
    do.call(grid.arrange, c(plots, ncol = 2))
  })
  
  ## ---- Descriptive statistics (extended) ----
  output$stats_table <- renderDT({
    req(raw_data())
    df <- raw_data()
    stats <- psych::describe(df[, FEATURE_NAMES])
    stats$variance <- stats$sd^2
    stats$IQR <- sapply(df[, FEATURE_NAMES], IQR)
    stats$CI95_lower <- stats$mean - 1.96 * stats$se
    stats$CI95_upper <- stats$mean + 1.96 * stats$se
    stats <- round(stats, 3)
    stats <- cbind(Variable = rownames(stats), stats)
    rownames(stats) <- NULL
    datatable(stats, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$severity_table <- renderDT({
    req(raw_data())
    df <- raw_data()
    tab <- as.data.frame(table(df$Disaster_Severity_Level))
    colnames(tab) <- c("Severity_Level", "Count")
    tab$Proportion_Percent <- round(100 * tab$Count / sum(tab$Count), 2)
    datatable(tab, options = list(dom = 't'))
  })
  
  ## ---- Shapiro-Wilk normality test per variable ----
  output$shapiro_table <- renderDT({
    req(raw_data())
    df <- raw_data()
    res <- lapply(FEATURE_NAMES, function(col) {
      x <- df[[col]]
      if (length(x) > 5000) x <- sample(x, 5000)  # shapiro.test max n = 5000
      test <- shapiro.test(x)
      data.frame(
        Variable = col,
        W_Statistic = round(unname(test$statistic), 4),
        P_Value = format_pval(test$p.value),
        Normally_Distributed = ifelse(test$p.value > 0.05, "Yes (p > 0.05)", "No (p <= 0.05)")
      )
    })
    out <- do.call(rbind, res)
    datatable(out, options = list(pageLength = 10, dom = 't'))
  })
  
  ## ---- One-way ANOVA: variable ~ severity level ----
  output$anova_table <- renderDT({
    req(raw_data())
    df <- raw_data()
    res <- lapply(FEATURE_NAMES, function(col) {
      fmla <- as.formula(paste(col, "~ Disaster_Severity_Level"))
      a <- aov(fmla, data = df)
      s <- summary(a)[[1]]
      data.frame(
        Variable = col,
        F_Value = round(s$`F value`[1], 4),
        DF_Between = s$Df[1],
        DF_Within = s$Df[2],
        P_Value = format_pval(s$`Pr(>F)`[1]),
        Significant_Severity_Effect = ifelse(s$`Pr(>F)`[1] < 0.05, "Yes (p < 0.05)", "No (p >= 0.05)")
      )
    })
    out <- do.call(rbind, res)
    datatable(out, options = list(pageLength = 10, dom = 't'))
  })
  
  ## ---- Correlation matrix (r) with significance (p-values) ----
  corr_result <- reactive({
    req(raw_data())
    df <- raw_data()
    psych::corr.test(df[, FEATURE_NAMES], adjust = "none")
  })
  
  output$corr_heatmap <- renderPlot({
    req(corr_result())
    r_mat <- corr_result()$r
    melted <- reshape2::melt(r_mat, varnames = c("Variable_X", "Variable_Y"), value.name = "r")
    
    ggplot(melted, aes(x = Variable_X, y = Variable_Y, fill = r)) +
      geom_tile(color = "white") +
      geom_text(aes(label = round(r, 2)), size = 3, family = plot_font) +
      scale_fill_gradient2(low = "#c0392b", mid = "white", high = "#1a5276",
                           midpoint = 0, limits = c(-1, 1), name = "Pearson r") +
      labs(x = "Variable (X axis)", y = "Variable (Y axis)",
           title = "Correlation Heatmap (Pearson r)") +
      theme_minimal(base_family = plot_font) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            axis.text.y = element_text(size = 8),
            plot.title = element_text(face = "bold"))
  })
  
  output$corr_table <- renderDT({
    req(corr_result())
    r_mat <- corr_result()$r
    p_mat <- corr_result()$p
    
    r_long <- reshape2::melt(r_mat, varnames = c("Variable_1", "Variable_2"), value.name = "r")
    p_long <- reshape2::melt(p_mat, varnames = c("Variable_1", "Variable_2"), value.name = "p_value")
    
    merged <- merge(r_long, p_long, by = c("Variable_1", "Variable_2"))
    merged <- merged[merged$Variable_1 != merged$Variable_2, ]
    ## keep each unique pair only once
    merged$pair_key <- apply(merged[, c("Variable_1","Variable_2")], 1, function(x) paste(sort(x), collapse = "_"))
    merged <- merged[!duplicated(merged$pair_key), ]
    merged$pair_key <- NULL
    
    merged$r <- round(merged$r, 3)
    merged$Significant <- ifelse(merged$p_value < 0.05, "Yes (p < 0.05)", "No (p >= 0.05)")
    merged <- merged[order(-abs(merged$r)), ]
    merged$p_value <- format_pval(merged$p_value)
    rownames(merged) <- NULL
    
    datatable(merged, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  ## ---- STEP 2: Preprocess (Module 2, triggered by arrow button) ----
  observeEvent(input$preprocess_btn, {
    req(raw_data())
    df <- raw_data()
    proc <- df
    
    for (col in FEATURE_NAMES) {
      if (any(is.na(proc[[col]]))) {
        proc[[col]][is.na(proc[[col]])] <- median(proc[[col]], na.rm = TRUE)
      }
    }
    
    for (col in FEATURE_NAMES) {
      rng <- range(proc[[col]])
      proc[[paste0(col, "_norm")]] <- if (diff(rng) == 0) 0 else
        (proc[[col]] - rng[1]) / (rng[2] - rng[1])
    }
    
    proc$Disaster_Severity_Level_Encoded <- as.integer(proc$Disaster_Severity_Level) - 1
    
    preprocessed_data(proc)
    showNotification("Modelling step complete: normalized + encoded.",
                     type = "message", duration = 4)
  })
  
  ## ---- Module 2 main panel: locked / ready / done states ----
  output$module2_main_ui <- renderUI({
    if (is.null(raw_data())) {
      tagList(h4("Module 2 is locked"),
              p("Generate a dataset in Module 1 first — this tab unlocks automatically."))
    } else if (is.null(preprocessed_data())) {
      tagList(h4("Dataset ready for preprocessing"),
              p("Click 'Preprocess' in the sidebar (Module 2) to run missing-value imputation,
                min-max normalization, and severity label-encoding."))
    } else {
      tagList(
        h4("Preprocessed Dataset Preview"),
        p("Normalized columns are suffixed with _norm (0-1 scale). Severity level is
          also encoded as an integer for direct use in the RL environment."),
        DTOutput("preprocessed_table"),
        
        hr(),
        h4("Existing System vs Proposed System - Performance Comparison"),
        p("Existing System = a simple baseline regression using only two basic resource-count
          predictors (mimicking how a traditional, non-RL system would allocate resources).
          Proposed System = a regression using the full normalized feature set. Bars show RMSE
          (lower is better) separately for the training split and the held-out test split."),
        plotOutput("model_compare_bar", height = "420px"),
        
        hr(),
        h4("Overfitting / Underfitting Diagnostic (Learning Curve)"),
        p("Model complexity is increased by fitting higher-degree polynomial terms on a key
          predictor and re-measuring error on train vs test data. Where the training-error line
          keeps falling but the test-error line turns back up, the model is overfitting; where
          both lines are still high together, the model is underfitting."),
        plotOutput("overfit_line", height = "460px"),
        verbatimTextOutput("overfit_summary")
      )
    }
  })
  
  output$preprocessed_table <- renderDT({
    req(preprocessed_data())
    datatable(preprocessed_data(), options = list(pageLength = 10, scrollX = TRUE))
  })
  
  ## =========================================================================
  ## Module 2 - Existing vs Proposed model comparison (train/test RMSE)
  ## =========================================================================
  train_test_split <- reactive({
    req(preprocessed_data())
    df <- preprocessed_data()
    set.seed(input$seed_val + 1)
    n <- nrow(df)
    idx <- sample(seq_len(n), size = round(0.8 * n))
    list(train = df[idx, ], test = df[-idx, ])
  })
  
  model_metrics <- reactive({
    split <- train_test_split()
    train <- split$train
    test  <- split$test
    
    target <- "Resource_Allocation_Efficiency_Score_norm"
    
    ## Existing System: baseline uses only raw resource-count predictors
    existing_fmla <- as.formula(paste(target, "~ Available_Rescue_Units_norm + Available_Medical_Units_norm"))
    existing_model <- lm(existing_fmla, data = train)
    
    ## Proposed System: full normalized feature set (excluding the target)
    all_norm_cols <- paste0(FEATURE_NAMES, "_norm")
    proposed_predictors <- setdiff(all_norm_cols, target)
    proposed_fmla <- as.formula(paste(target, "~", paste(proposed_predictors, collapse = " + ")))
    proposed_model <- lm(proposed_fmla, data = train)
    
    rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
    
    data.frame(
      Model = rep(c("Existing System (Baseline)", "Proposed System (Full Feature Set)"), each = 2),
      Dataset = rep(c("Train", "Test"), times = 2),
      RMSE = c(
        rmse(train[[target]], predict(existing_model, train)),
        rmse(test[[target]],  predict(existing_model, test)),
        rmse(train[[target]], predict(proposed_model, train)),
        rmse(test[[target]],  predict(proposed_model, test))
      )
    )
  })
  
  output$model_compare_bar <- renderPlot({
    req(model_metrics())
    m <- model_metrics()
    m$RMSE <- round(m$RMSE, 4)
    
    ggplot(m, aes(x = Model, y = RMSE, fill = Dataset)) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.6), width = 0.55) +
      geom_text(aes(label = RMSE), position = position_dodge(width = 0.6),
                vjust = -0.4, size = 4, family = plot_font) +
      labs(
        title = "Existing vs Proposed System: Train & Test RMSE",
        x = "Model Type (X axis)",
        y = "RMSE - Root Mean Squared Error (Y axis, lower = better)",
        fill = "Dataset Split"
      ) +
      theme_minimal(base_family = plot_font) +
      theme(plot.title = element_text(face = "bold"),
            axis.text.x = element_text(size = 10))
  })
  
  ## =========================================================================
  ## Module 2 - Overfitting / Underfitting diagnostic (polynomial complexity)
  ## =========================================================================
  complexity_curve <- reactive({
    split <- train_test_split()
    train <- split$train
    test  <- split$test
    
    target    <- "Resource_Allocation_Efficiency_Score_norm"
    predictor <- "Infrastructure_Damage_Percent_norm"
    degrees   <- 1:10
    
    rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
    
    results <- lapply(degrees, function(d) {
      fmla <- as.formula(paste(target, "~ poly(", predictor, ",", d, ")"))
      model <- tryCatch(lm(fmla, data = train), error = function(e) NULL)
      if (is.null(model)) return(NULL)
      data.frame(
        Degree = d,
        Train_RMSE = rmse(train[[target]], predict(model, train)),
        Test_RMSE  = rmse(test[[target]],  predict(model, test))
      )
    })
    do.call(rbind, results[!sapply(results, is.null)])
  })
  
  output$overfit_line <- renderPlot({
    req(complexity_curve())
    cc <- complexity_curve()
    long <- reshape2::melt(cc, id.vars = "Degree",
                           variable.name = "Dataset", value.name = "RMSE")
    long$Dataset <- gsub("_RMSE", "", long$Dataset)
    
    best_degree <- cc$Degree[which.min(cc$Test_RMSE)]
    
    ggplot(long, aes(x = Degree, y = RMSE, color = Dataset, group = Dataset)) +
      geom_line(linewidth = 1) +
      geom_point(size = 2.2) +
      geom_vline(xintercept = best_degree, linetype = "dashed", color = "gray40") +
      annotate("text", x = best_degree, y = max(long$RMSE) * 0.98,
               label = paste("Best generalization\n(degree", best_degree, ")"),
               size = 3.3, family = plot_font, hjust = -0.05) +
      scale_x_continuous(breaks = cc$Degree) +
      labs(
        title = "Train vs Test RMSE across Model Complexity (Polynomial Degree)",
        x = "Model Complexity - Polynomial Degree on Infrastructure_Damage_Percent (X axis)",
        y = "RMSE - Root Mean Squared Error (Y axis, lower = better)",
        color = "Dataset Split"
      ) +
      theme_minimal(base_family = plot_font) +
      theme(plot.title = element_text(face = "bold"))
  })
  
  output$overfit_summary <- renderText({
    req(complexity_curve())
    cc <- complexity_curve()
    best_degree <- cc$Degree[which.min(cc$Test_RMSE)]
    low_degree_gap  <- cc$Test_RMSE[cc$Degree == min(cc$Degree)] - cc$Train_RMSE[cc$Degree == min(cc$Degree)]
    high_degree_gap <- cc$Test_RMSE[cc$Degree == max(cc$Degree)] - cc$Train_RMSE[cc$Degree == max(cc$Degree)]
    
    paste0(
      "Diagnosis:\n",
      "- Lowest test RMSE occurs at polynomial degree ", best_degree,
      " (Test RMSE = ", round(min(cc$Test_RMSE), 4), "). This is the best-generalizing complexity.\n",
      "- At degree ", min(cc$Degree), " (simplest model): Train RMSE = ", round(cc$Train_RMSE[cc$Degree == min(cc$Degree)], 4),
      ", Test RMSE = ", round(cc$Test_RMSE[cc$Degree == min(cc$Degree)], 4),
      " -> both errors are relatively high together, indicating UNDERFITTING.\n",
      "- At degree ", max(cc$Degree), " (most complex model): Train RMSE = ", round(cc$Train_RMSE[cc$Degree == max(cc$Degree)], 4),
      ", Test RMSE = ", round(cc$Test_RMSE[cc$Degree == max(cc$Degree)], 4),
      " -> if Test RMSE is clearly higher than Train RMSE here, the model is OVERFITTING beyond degree ", best_degree, "."
    )
  })
  
  ## ---- STEP 3: Export ----
  output$download_raw <- downloadHandler(
    filename = function() paste0("guardian_raw_dataset_", Sys.Date(), ".csv"),
    content = function(file) {
      req(raw_data())
      write.csv(raw_data(), file, row.names = FALSE)
    }
  )
  
  output$download_preprocessed <- downloadHandler(
    filename = function() paste0("guardian_preprocessed_dataset_", Sys.Date(), ".csv"),
    content = function(file) {
      req(preprocessed_data())
      write.csv(preprocessed_data(), file, row.names = FALSE)
    }
  )
}

#############################################################################
shinyApp(ui = ui, server = server)
#############################################################################