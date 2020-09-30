library(rcolgem)
library(phytools)
library(seqRFLP)
library(readr)
library(jsonlite)

demes <-  c('I', 'R_h', 'R_s')
births <- rbind(c('beta * S * I / (S + I + R_h + R_s)', '0', '0'),
                c('0', '0', '0'),
                c('0', '0', '0'))

rownames(births)=colnames(births) <- demes

migrations <- rbind(c('0', 'gamma * (1 - s) * I', 'gamma * s * I'),
                    c('0', '0', '0'),
                    c('0', '0', '0'))

rownames(migrations)=colnames(migrations) <- demes

deaths <- c('0.', '0.', '0.')
names(deaths) <- demes

nonDemeDynamics <- c( S = '-beta * S * I / (S + I + R_h + R_s)')

SIR_model <- build.demographic.process(births=births, 
                                       nonDemeDynamics=nonDemeDynamics, 
                                       migrations=migrations,
                                       deaths=deaths, 
                                       parameterNames = c('beta', 'gamma', 's'),
                                       rcpp = TRUE,
                                       sde=FALSE)

beta <- 0.75
gamma <- 0.3
s <- 0.1
theta <- c(beta=beta, gamma=gamma, s=s)
t0 <- 0
t1 <- 20
x0 <- c(S = 999, I = 1, R_h = 0, R_s = 0)
show.demographic.process(SIR_model, theta, x0, t0, t1)

res <- 1000
Rs_times <- SIR_model(theta, x0, t0, t1, res)[[5]][,'R_s']
diff <- floor(Rs_times[2:res]) - floor(Rs_times[1:res-1])
sample_times <- which(diff!=0) + 1
sample_times <- sample_times * t1/res
sample_states <- t(replicate(length(sample_times), c(0,0,1)))
tree <- sim.co.tree(theta, SIR_model, x0, t0, sample_times, sample_states)
plot.phylo(tree)
L <- 1000
#rate <- rgamma(n=L,shape=0.25,rate=0.25)
data <- genSeq(tree,l=L,format='matrix')
sequences <- matrix(apply(toupper(data), 1, function(x) paste(x, collapse = '')))
df <- data.frame(name=rownames(data), sequences)

N <- 1000
S <- SIR_model(theta, x0, t0, t1, res)[[5]][,'S']
Re <- beta / gamma * S / N
write_json(Re, "R.json")

write.tree(tree, "sim.tree")
dataframe2fas(df, file="sim.fa")

dates <- data.frame(name = names(tree$sampleTimes), date=tree$sampleTimes)
write_csv(dates, 'dates.csv')