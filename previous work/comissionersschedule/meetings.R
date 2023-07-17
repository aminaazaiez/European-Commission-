library(data.table)
library(countrycode)
library(stringdist)
library(reshape2)
#library(rJava)
#.jinit(parameters="-Xmx4g")
#library(openNLP)
#library(NLP)
#library(foreach)
#library(doSNOW)
#library(openNLPmodels.en)
#library(openNLPmodels.da)
#library(openNLPmodels.de)
#library(openNLPmodels.it)
#library(openNLPmodels.es)
#library(openNLPmodels.pt)
#library(openNLPmodels.nl)
#library(openNLPmodels.sv)


commissioner = data.table(read.csv("/home/andreaskarpf/Dropbox/comissionersschedule/all_meetings.csv",sep=","))
cabinet      = data.table(read.csv("/home/andreaskarpf/Dropbox/comissionersschedule/all_meetings_by_cabinet.csv",sep=","))
meetings1    = commissioner[,list(Date,Commissioner,Entity,Subject)]
meetings2    = cabinet[,list(Date,Commissioner,Entity,Subject)]
meetings     = rbind(meetings1,meetings2)
trim         = function (x) gsub("\\s+"," ",gsub("^\\s+|\\s+$","",x))
companies    = data.table(colsplit(as.character(meetings$Entity),pattern='\\|',names=paste0(rep("c",30),1:30)))
companies    = data.table(sapply(companies,trim))
netw         = cbind(meetings[,list(Date,Commissioner)],companies)
setkey(netw,Date,Commissioner,c1)
netw       = unique(netw)
netw[,Date:=NULL]
elist     = melt(netw,id.vars="Commissioner")
elist[,variable:=NULL]
setnames(elist,c("source","target"))
elist     = elist[!is.na(target)]
elist = elist[,.N,by=list(source,target)]
setnames(elist,"N","weight")
elist = elist[source!=target]
elist = elist[source!="" & target!=""]
#elist = elist[!(stringdist(source,target,method="jw")<0.2)]
nodtab = rbind(elist[,list(unique(source),"euc")],elist[,list(unique(target),"other")],use.names=F)
setnames(nodtab,c("source","type"))
setkey(nodtab,source)
nodtab = unique(nodtab)
nodtab[,id:=.I]
elist = merge(elist,nodtab,by="source")
setnames(nodtab,c("target","type","id"))
elist = merge(elist,nodtab,by="target")
elist  = elist[,list(id.x,id.y,weight)]
nodtab = nodtab[,list(id,target,type)]
setnames(elist,c("source","target","weight"))
elist[,Type:="Undirected"]
setnames(nodtab,c("id","name","type"))
nodtab[,acronyms:=gsub(".*\\((.*)\\).*","\\1",name)]


write.csv(elist,"/home/andreaskarpf/Dropbox/comissionersschedule/elist.csv")
write.csv(nodtab,"/home/andreaskarpf/Dropbox/comissionersschedule/nodetab.csv")


#############################################
#perf = function(s){
#  f    = as.String(s)
#  sent_token_annotator = Maxent_Sent_Token_Annotator()
#  word_token_annotator = Maxent_Word_Token_Annotator()
#  entity_annotator     = Maxent_Entity_Annotator()
#  ann                  = annotate(s,list(sent_token_annotator,word_token_annotator))
#  indi                 = entity_annotator(s,ann)
#  gc()
#  ifelse((as.character(indi)[1]=="integer(0)"),"NN",f[indi])}
#meetings[,Entity:=gsub("\\s*\\([^\\)]+\\)","",Entity)]
#meetings[,person:=paste(Entity,Subject,sep=" ")]
#meetings[,c("Entity","Subject"):=list(NULL,NULL)]
#cluster = makeCluster(30, type = "SOCK")
#registerDoSNOW(cluster)
#pckgs = c("data.table","openNLP","NLP","openNLPmodels.en","openNLPmodels.de","openNLPmodels.it","openNLPmodels.es","openNLPmodels.pt")
#expf  = c("perf","meetings")
#st = Sys.time()
#res  =  foreach(i=1:nrow(meetings),.combine="rbind",.packages=pckgs,.export=expf) %dopar% {
#data.table(perf(meetings[i]$Subject),perf(meetings[i]$Entity))
#}
#stopCluster(cluster)
#et = Sys.time()
#et - st


