using Distributions, CSV, Tables
a = MvNormal([0.; 0.],1.)
s = sampler(a)
data = cat([rand(s) for i=1:500]...;dims=2)
t = Tables.table(data')
CSV.write("data.csv",t)
