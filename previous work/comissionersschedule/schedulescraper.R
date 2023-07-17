require(devtools)
require(httr)
require(rvest)
require(dplyr)
require(magrittr)

# extract the link to the president
links_raw <- read_html("http://ec.europa.eu/commission/2014-2019_en")
link_president <- links_raw %>%
html_node(".team-members-president") %>%
html_node("a") %>%
html_attr("href")
# extract the links of all commissioners
links <- links_raw %>%
html_node("#quicktabs-tabpage-team_members-2") %>%
html_nodes(".field-content.member-details") %>%
html_node("a") %>%
html_attr("href")

links <- c(links, link_president)
commissioners <- list()
for(link in links){
print(link)
member_page <- read_html(link)
member_name <- paste((member_page %>%
html_node("h1.page-header") %>%
html_node("span.first-name") %>% 
html_text()), (member_page %>%
html_node("h1.page-header") %>%
html_node("span.last-name")  %>% 
html_text()), sep = " ")
agenda_links = member_page %>%
html_node("#block-views-activities-block") %>%
html_nodes("ul") %>%
html_nodes("a") %>%
html_attr("href")
newCommissioner <- list(name = member_name, meetings_1 = agenda_links[1], meetings_2 = agenda_links[2])
commissioners[[length(commissioners) + 1]] <- newCommissioner
}

#########################commissioner#########################
all_data <- data.frame()
#commissioner = commissioners[[21]]

for(commissioner in commissioners){
print(commissioner[["name"]])
agenda_meetings_1_page <- read_html(commissioner[["meetings_1"]])
pagelinksContainer     <- tryCatch(html_node(agenda_meetings_1_page,"span.pagelinks"),error=function(e) e)
dataForCommissioner    <- data.frame()

if(!any(class(pagelinksContainer) %in% "error")){
pagelinks <- agenda_meetings_1_page %>% html_node("span.pagelinks") %>% html_nodes("a") %>% html_attr("href")
pagelink  <- tail(pagelinks,n=1)
pPosition <- regexpr("-p=",pagelink)[1]
lastPage <- substr(pagelink,  pPosition + 3, nchar(pagelink))

# loop over all pages
for(i in 1:lastPage){
data <- read_html(paste("http://ec.europa.eu", substr(pagelink, 1, pPosition), "p=", i, sep = "")) %>%  
as.character %>% gsub('<br/>','|',.) %>% read_html %>% html_node("table#listMeetingsTable") %>% html_table()

dataForCommissioner <- rbind(dataForCommissioner, data)
}
} else {
dataForCommissioner <- agenda_meetings_1_page %>%  
as.character %>% gsub('<br/>','|',.) %>% read_html %>% html_node("table#listMeetingsTable") %>% html_table()
}
  
dataForCommissioner %<>% mutate(Commissioner = commissioner[["name"]])
all_data <- rbind(all_data, dataForCommissioner)
}

all_data %<>% rename(Entity = `Entity/ies met`, Subject = `Subject(s)`) 
all_data %<>% select(Commissioner, Date, Location, Entity, Subject)
all_data$Entity <- gsub("\\t", "", all_data$Entity)
all_data$Entity <- gsub("\\r", "", all_data$Entity)
all_data$Entity <- gsub("\n", "", all_data$Entity)
write.csv(file = "/home/andreaskarpf/Dropbox/comissionersschedule/all_meetings.csv", all_data, row.names = F)

#########################cabinet#########################

all_data_cabinet <- data.frame()

for(commissioner in commissioners){
print(commissioner[["name"]])
agenda_meetings_2_page <- read_html(commissioner[["meetings_2"]])
pagelinksContainer     <- tryCatch(html_node(agenda_meetings_2_page,"span.pagelinks"),error=function(e) e)
dataForCommissioner    <- data.frame()

if(!any(class(pagelinksContainer) %in% "error")){
pagelinks <- agenda_meetings_2_page %>% html_node("span.pagelinks") %>% html_nodes("a") %>% html_attr("href")
pagelink  <- tail(pagelinks, n = 1)
pPosition <- regexpr("-p=",pagelink)[1]
lastPage <- substr(pagelink,  pPosition + 3, nchar(pagelink))

for(i in 1:lastPage){
data                <- read_html(paste("http://ec.europa.eu", substr(pagelink, 1, pPosition), "p=", i, sep = "")) %>%  
as.character %>% gsub('<br/>','|',.) %>% read_html %>% html_node("table#listMeetingsTable") %>% html_table()
dataForCommissioner <- rbind(dataForCommissioner, data)
}
} else {
dataForCommissioner <- agenda_meetings_2_page %>%  
as.character %>% gsub('<br/>','|',.) %>% read_html %>% html_node("table#listMeetingsTable") %>% html_table()
}

dataForCommissioner %<>% mutate(Commissioner = commissioner[["name"]])
all_data_cabinet    <- rbind(all_data_cabinet, dataForCommissioner)
}


# make data more tidy
all_data_cabinet %<>%
rename(Entity = `Entity/ies met`, Subject = `Subject(s)`, MemberName = Name) 
all_data_cabinet %<>%
select(Commissioner, MemberName, Date, Location, Entity, Subject)
all_data_cabinet$Entity <- gsub("\\t", "", all_data_cabinet$Entity)
all_data_cabinet$Entity <- gsub("\\r", "", all_data_cabinet$Entity)
all_data_cabinet$Entity <- gsub("\n", "", all_data_cabinet$Entity)
write.csv(file = "/home/andreaskarpf/Dropbox/comissionersschedule/all_meetings_by_cabinet.csv", all_data_cabinet, row.names = F)


