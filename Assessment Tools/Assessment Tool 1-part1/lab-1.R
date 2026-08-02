#---------------------------------------
# Markov Decision Process Demo
#---------------------------------------

library(ggplot2)

grid_size <- 4
start <- c(1,1)
goal  <- c(4,4)

state <- start
step <- 1
total_reward <- 0

# Track path
path <- matrix("", grid_size, grid_size)
path[start[1], start[2]] <- "S"
path[goal[1], goal[2]]   <- "G"

trajectory <- data.frame(step=1, row=start[1], col=start[2])

cat("Initial Grid:\n")
print(path)

cat("\nStarting Simulation\n\n")

while(!(all(state==goal))){
  
  row <- state[1]
  col <- state[2]
  
  possible <- list()
  if(row>1) possible$Up <- c(row-1,col)
  if(row<grid_size) possible$Down <- c(row+1,col)
  if(col>1) possible$Left <- c(row,col-1)
  if(col<grid_size) possible$Right <- c(row,col+1)
  
  best_action <- NULL
  best_distance <- 100
  state_next <- NULL
  
  for(a in names(possible)){
    next_state <- possible[[a]]
    distance <- abs(goal[1]-next_state[1]) + abs(goal[2]-next_state[2])
    if(distance < best_distance){
      best_distance <- distance
      best_action <- a
      state_next <- next_state
    }
  }
  
  # Exploration: 20% chance to pick a random action
  if(runif(1) < 0.2){
    rand_action <- sample(names(possible),1)
    state_next <- possible[[rand_action]]
    best_action <- rand_action
  }
  
  reward <- -1
  if(all(state_next==goal)) reward <- 100
  total_reward <- total_reward + reward
  
  cat("Step:", step, "\n")
  cat("Current State :", state, "\n")
  cat("Action Taken  :", best_action, "\n")
  cat("Next State    :", state_next, "\n")
  cat("Reward        :", reward, "\n\n")
  
  state <- state_next
  step <- step+1
  
  trajectory <- rbind(trajectory, data.frame(step=step, row=state[1], col=state[2]))
}

cat("Simulation Complete!\n")
cat("Total Steps :", step-1, "\n")
cat("Total Reward:", total_reward, "\n")

# ==========================
# Visualization of Path
# ==========================

ggplot(trajectory, aes(x=col, y=row)) +
  geom_tile(fill="white", color="black", width=1, height=1) +
  geom_path(color="blue", size=1.2) +
  geom_point(color="red", size=3) +
  geom_text(aes(label=step), vjust=-1, size=3) +
  scale_y_reverse() +
  coord_fixed() +
  ggtitle("MDP Grid Path Simulation") +
  theme_minimal()
