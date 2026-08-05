###################################################
# Reinforcement Learning Inventory Dataset
# Bellman Equation Case Study
###################################################

set.seed(123)

records <- 500

episode <- ceiling((1:records)/25)
step <- ((1:records)-1) %% 25 + 1

# Inventory state (0-100 units)
state <- sample(20:80, records, replace = TRUE)

# Customer demand
demand <- sample(5:40, records, replace = TRUE)

# Existing retailer policy
existing_policy <- sample(c("Order Low",
                            "Order Medium",
                            "Order High"),
                          records,
                          replace = TRUE)

# Replenishment action
action <- sample(c(0,10,20,30,40),
                 records,
                 replace = TRUE)

# Inventory after demand and replenishment
next_state <- pmax(0, state - demand + action)

# Holding Cost
holding_cost <- pmax(next_state * 0.5,0)

# Stockout Cost
stockout_cost <- pmax((demand - state) * 6,0)

# Ordering Cost
ordering_cost <- ifelse(action>0,40,0)

# Penalty
penalty <- holding_cost +
  stockout_cost +
  ordering_cost

# Reward
reward <- 200 - penalty

# Cumulative Reward
cumulative_reward <- ave(reward,
                         episode,
                         FUN=cumsum)

# End of episode
done <- ifelse(step==25,TRUE,FALSE)

dataset <- data.frame(
  
  Episode=episode,
  
  Step=step,
  
  State=state,
  
  Demand=demand,
  
  Action=action,
  
  Next_State=next_state,
  
  Reward=round(reward,2),
  
  Penalty=round(penalty,2),
  
  Done=done,
  
  Existing_Policy=existing_policy
  
)

write.csv(dataset,
          "RL_Inventory_Dataset.csv",
          row.names=FALSE)

cat("Dataset Generated Successfully\n")

cat("Rows :",nrow(dataset),"\n")

cat("Columns :",ncol(dataset),"\n")

summary(dataset)

head(dataset)
