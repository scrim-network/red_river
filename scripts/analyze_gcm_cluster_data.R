# Script to analyze clustering of GCM realizations based on precipitation and
# temperature projections. Uses PCA and kmeans based on 10-year means of anomalies.
###############################################################################

source('esd_r/config.R')

anoms <- read.csv(file.path(cfg['path_cmip5_trends'],'cluster_anoms.csv'),
                  stringsAsFactors=FALSE, row.names=1)
pc <- prcomp(anoms,center=T,scale=T)
summary(pc)

# First 24 PCs (85% of variance)
comp <- data.frame(pc$x[,1:24])

# Determine # of clusters to use
wss <- (nrow(comp)-1)*sum(apply(comp,2,var))
for (i in 2:100) wss[i] <- sum(kmeans(comp, centers=i, iter.max=1000, nstart=25)$withinss)
plot(wss)

# Calculate final clusters and output cluster number for each model realization
k <- kmeans(comp, centers=12, iter.max=1000, nstart=25)
clusters <- data.frame(k$cluster)
write.csv(file.path(cfg['path_cmip5_trends'],'cluster_nums.csv'))