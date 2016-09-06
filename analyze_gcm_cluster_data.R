# TODO: Add comment
# 
# Author: jwo118
###############################################################################


anoms <- read.csv('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/cluster_anoms.csv', stringsAsFactors=FALSE, row.names=1)
pc <- prcomp(anoms,center=T,scale=T)
summary(pc)

# First 24 PCs (85% of variance)
comp <- data.frame(pc$x[,1:24])

wss <- (nrow(comp)-1)*sum(apply(comp,2,var))
for (i in 2:100) wss[i] <- sum(kmeans(comp, centers=i, iter.max=1000, nstart=25)$withinss)
plot(wss)

k <- kmeans(comp, centers=12, iter.max=1000, nstart=25)

clusters <- data.frame(k$cluster)