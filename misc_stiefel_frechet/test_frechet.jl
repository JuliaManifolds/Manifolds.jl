import Pkg
Pkg.add(path="/data/share/Manifolds.jl")
using Manifolds
using LinearAlgebra
using Random

using JLD2

max_norm = 10.


function linf(mat)
  return maximum(abs.(mat))
end  

function do_one(n, i, save_dir)
    ft = (rand() + .1)*max_norm/1.1
    A = rand(n, n)

    A = A / norm(A, 2)*ft
    E = rand(n, n)
    E = E / norm(E, 2)*ft
    buff = Array{Float64, 2}(undef, 16*n, n)
    @views begin
        expA = buff[1:n, :]
        expAE = buff[n+1:2*n, :]
    end
    Manifolds.expm_frechet!(buff, A, E)
    eA1, eAE1 = Manifolds.expm_frechet(A, E)
    # return A, E, expA[:, :], expAE[:, :]
    jldsave(joinpath(
        save_dir, "exp_"*string(i)*'_'*string(n)*".jld2"),
            A=A, E=E, expA=expA[:, :], expAE=expAE[:, :])
    dlt = 1e-7
    ndiff = linf((exp(A+dlt*E) .- exp(A))/dlt .- expAE)
    return linf(eA1 .- buff[1:n, :]), linf(eAE1 .- buff[n+1:2*n, :]), ndiff
end
save_dir = "test_expm_frechet"

Random.seed!(0)
NN = 500
check_vec = Array{Float64, 2}(undef, NN, 4)

for i in 1:NN
    n = Int(ceil(2^(10/1.02*(.02+rand()))))
    if n == 1
        n = 2
    end
    
    if i % 20 == 0
        println("Doing "*string(i)*" "*string(n))
    end
    check_vec[i, 1] = n
    check_vec[i, 2:end] .= do_one(n, i, save_dir)    
end        
println(maximum(check_vec[:, 2:end], dims=1))
println(median(check_vec, dims=1))
println(sort(check_vec[:, 4])[Int(ceil(.9*NN))])

using BenchmarkTools

function bench_one(n, n_samples)
    ft = (rand() + .1)*max_norm/1.1
    A = rand(n, n)

    A .= A / norm(A, 2)*ft
    
    E = rand(n, n)
    
    E .= E / norm(E, 2)*ft

    buff = Array{Float64, 2}(undef, 16*n, n)
    
    texp = @benchmark exp($A) samples=n_samples
    
    texpF = @benchmark Manifolds.expm_frechet($A, $E) samples=n_samples
    texpF_bf = @benchmark Manifolds.expm_frechet!($buff, $A, $E) samples=n_samples    
    return texp, texpF, texpF_bf
end

n = Int(ceil(2^(10/1.02*(.02+rand()))))
if n == 1
    n = 2
end
ret = bench_one(n, 5)
println(n," ", median(ret[1]), " ",  median(ret[2]), median(ret[3]))


