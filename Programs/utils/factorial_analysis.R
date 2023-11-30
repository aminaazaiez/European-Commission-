library(readr)
library(FactoMineR)
data <- read.csv("Documents/Cours/These/European Commission/Programs/data_test_R.csv", row.names = 'Name')
res <- MFA(data, group = c(4,1,1,1,3,4,41), type = c('s', 'n','s','n','s','n','n' ),  ncp = 111, name.group = c("centrality","country", "lobbying", "sector", "financial", "level", "field") ,num.group.sup = c(1))
#res <- PCA(data, quanti.sup = c(1,2,3,4))
#res <- FAMD(data, sup.var = c('Betweenness',  'Closeness', 'Degree', 'Strength') )

#Save ind coordinate in the new data base
coord_data <- as.data.frame(res$ind$coord)
write.csv(coord_data, file = 'Documents/Cours/These/European Commission/Programs/MFA_coord.csv', 
          row.names = TRUE)

fields = read.csv("Documents/Cours/These/European Commission/Programs/fields.csv")
res <- MCA(fields)

plot(res, invisible = "quali", select = "contrib 5", cex = 0.5, habillage = 'Trade')
plot(res, choix= 'var', select = "contrib 5")
dimdesc(res)
