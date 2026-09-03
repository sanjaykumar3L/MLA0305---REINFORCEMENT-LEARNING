#############################################################################
# PROJECT GUARDIAN - METHODOLOGY FLOW DIAGRAM GENERATOR
#
# Draws the 7-stage pipeline diagram (GMM Data Generation -> Statistical
# Validation -> Preprocessing -> Existing vs Proposed Modelling ->
# Hybrid ABM+DES Simulation -> Q-Learning RL Dispatch Agent -> Performance
# Comparison & Visualization) exactly as used in the Methodology slide.
#
# KEY FEATURES
#   - Every box's height/width is computed FROM the actual rendered text
#     extent (via grid::grobWidth/grobHeight), so labels always fit
#     perfectly inside their box with consistent padding - no manual
#     trial-and-error sizing.
#   - All text (box labels, arrow labels) uses Cambria. Falls back to
#     serif silently if Cambria isn't installed, so the script never errors.
#   - Output is a high-resolution PNG ready to drop straight into PowerPoint.
#
# Required packages: grid (base R, no install needed)
# Optional: extrafont, for guaranteed Cambria embedding on non-Windows
#           systems -> install.packages("extrafont"); extrafont::font_import()
#############################################################################

library(grid)

## ---------------------------------------------------------------------
## Font setup: try to register Cambria; fall back to serif if unavailable.
## ---------------------------------------------------------------------
font_family <- "serif"

if (.Platform$OS.type == "windows") {
  tryCatch({
    windowsFonts(Cambria = windowsFont("Cambria"))
    font_family <- "Cambria"
  }, error = function(e) NULL)
} else if (requireNamespace("extrafont", quietly = TRUE)) {
  tryCatch({
    extrafont::loadfonts(device = "pdf", quiet = TRUE)
    if ("Cambria" %in% extrafont::fonts()) font_family <- "Cambria"
  }, error = function(e) NULL)
}

## ---------------------------------------------------------------------
## Pipeline stages: label, fill color (kept in the same order/palette
## as the original slide diagram), and the arrow label beneath each.
## ---------------------------------------------------------------------
stages <- list(
  list(label = "GMM Synthetic\nData Generation",              fill = "#8ecbe8"),
  list(label = "Statistical Validation\n(Normality, ANOVA, Correlation)", fill = "#9bdfa9"),
  list(label = "Preprocessing &\nMin-Max Normalization",       fill = "#f2cf6b"),
  list(label = "Existing vs Proposed\nRegression Modelling",   fill = "#f0908a"),
  list(label = "Hybrid ABM + DES\nSimulation",                 fill = "#c6a3e0"),
  list(label = "Q-Learning RL\nDispatch Agent",                fill = "#f3b672"),
  list(label = "Performance Comparison\n& Visualization",      fill = "#7fd6d6")
)

arrow_labels <- c(
  "Generate Data", "Validate Quality", "Clean & Scale",
  "Benchmark Models", "Simulate Incidents", "Learn Policy"
)

## ---------------------------------------------------------------------
## Layout parameters (inches)
## ---------------------------------------------------------------------
canvas_w   <- 6.5
canvas_h   <- 15.5
box_w      <- 5.6                 # fixed box width, generous for all labels
pad_x      <- 0.35                # horizontal padding inside each box
pad_y      <- 0.28                # vertical padding inside each box
gap        <- 0.62                # vertical gap between boxes (for arrows)
label_size <- 15                  # box label font size (pt)
arrow_size <- 10.5                # arrow caption font size (pt)
title_size <- 13                  # (unused placeholder, kept for future titles)

## ---------------------------------------------------------------------
## STEP 1: Measure each box's required height from its actual text,
## so every label fits perfectly with consistent padding - never
## overflowing, never leaving excess empty space.
## ---------------------------------------------------------------------
measure_box_height <- function(label, fontsize, fontfamily, width_in) {
  g <- textGrob(label, gp = gpar(fontsize = fontsize, fontfamily = fontfamily,
                                 fontface = "bold"))
  h_in <- convertHeight(grobHeight(g), "inches", valueOnly = TRUE)
  h_in + 2 * pad_y
}

box_heights <- sapply(stages, function(s) {
  measure_box_height(s$label, label_size, font_family, box_w)
})

## ---------------------------------------------------------------------
## STEP 2: Compute vertical center-Y position for every box (top to
## bottom), then render everything to a PNG device.
## ---------------------------------------------------------------------
n <- length(stages)
total_height <- sum(box_heights) + gap * (n - 1) + 1.0   # + top/bottom margin
canvas_h <- total_height

y_top <- canvas_h - 0.5   # start 0.5" from the top
centers_y <- numeric(n)
cursor <- y_top
for (i in seq_len(n)) {
  cursor <- cursor - box_heights[i] / 2
  centers_y[i] <- cursor
  cursor <- cursor - box_heights[i] / 2 - gap
}

png("methodology_flow_diagram.png", width = canvas_w, height = canvas_h,
    units = "in", res = 300, bg = "white")

grid.newpage()
pushViewport(viewport(width = unit(1, "npc"), height = unit(1, "npc"),
                      xscale = c(0, canvas_w), yscale = c(0, canvas_h)))

x_center <- canvas_w / 2

## ---- Draw arrows + arrow labels first (so boxes sit cleanly on top) ----
for (i in seq_len(n - 1)) {
  y_from <- centers_y[i]     - box_heights[i]     / 2
  y_to   <- centers_y[i + 1] + box_heights[i + 1] / 2
  
  grid.lines(
    x = unit(c(x_center, x_center), "native"),
    y = unit(c(y_from, y_to), "native"),
    arrow = arrow(type = "closed", length = unit(0.12, "inches"), angle = 20),
    gp = gpar(col = "#1a3c6e", lwd = 2.6, fill = "#1a3c6e")
  )
  
  mid_y <- (y_from + y_to) / 2
  grid.text(
    arrow_labels[i],
    x = unit(x_center + 0.18, "native"), y = unit(mid_y, "native"),
    just = "left",
    gp = gpar(fontsize = arrow_size, fontfamily = font_family,
              fontface = "italic", col = "#333333")
  )
}

## ---- Draw each box with its perfectly-fitted text ----
for (i in seq_len(n)) {
  cy <- centers_y[i]
  bh <- box_heights[i]
  
  grid.roundrect(
    x = unit(x_center, "native"), y = unit(cy, "native"),
    width  = unit(box_w, "inches"), height = unit(bh, "inches"),
    r = unit(0.12, "inches"),
    gp = gpar(fill = stages[[i]]$fill, col = "#2c3e50", lwd = 1.8)
  )
  
  grid.text(
    stages[[i]]$label,
    x = unit(x_center, "native"), y = unit(cy, "native"),
    just = "center",
    gp = gpar(fontsize = label_size, fontfamily = font_family,
              fontface = "bold", col = "#1a2634", lineheight = 1.25)
  )
}

popViewport()
dev.off()

cat("Saved: methodology_flow_diagram.png (", canvas_w, "x", canvas_h, "in @300dpi)\n")