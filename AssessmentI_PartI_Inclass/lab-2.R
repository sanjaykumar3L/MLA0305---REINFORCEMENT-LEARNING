#---------------------------------------
# Agent Path in Grid World (Enhanced)
#---------------------------------------

grid_size <- 4

# Example trajectory
x <- c(1,1,2,3,4,4,4)   # row positions
y <- c(1,2,2,2,2,3,4)   # column positions

plot(1:grid_size,
     1:grid_size,
     type="n",
     xlim=c(1,4),
     ylim=c(4,1),   # flip y-axis so row 1 is at bottom
     xlab="Column",
     ylab="Row",
     main="Agent Path in Grid World")

# Draw grid lines
abline(v=1:4, col="gray")
abline(h=1:4, col="gray")

# Mark start and goal
points(1,1,pch=19,col="blue",cex=2)
text(1,1,"S",pos=3,col="blue",cex=1.5)

points(4,4,pch=19,col="red",cex=2)
text(4,4,"G",pos=3,col="red",cex=1.5)

# Draw path with arrows
arrows(y[-length(y)], x[-length(x)],
       y[-1], x[-1],
       length=0.15, col="darkgreen", lwd=2)

# Mark steps along the path
points(y,x,pch=19,cex=1.2,col="darkorange")
text(y,x,labels=1:length(x),pos=1,cex=0.8,col="black")
