### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ b9aec7da-4210-11ec-3f51-71546dd47b1d
begin
    using Pkg
    Pkg.activate()# use global environment to have Manifolds in dev mode
    using Manifolds, BenchmarkTools, Random, Revise
end

# ╔═╡ aaed81b5-c309-4c7d-98b5-842113e92b36
begin
    n = 1000
    k = 40
end;

# ╔═╡ 1265db7d-f9f2-42b5-81af-99108c6521ae
M = SymplecticStiefel(n, k)

# ╔═╡ 0767ca5a-8784-4708-aab0-773c8cdbfd14
begin
    Random.seed!(42)
    p = rand(M)
    q = rand(eltype(p), n, k)
    r2 = zeros(k, k)
    X = rand(M, p)
    Y = rand(M, p)
end;

# ╔═╡ 42d56c68-1ae9-4744-9c76-ab681c0d8f54
r = inv(M, p) * q;

# ╔═╡ 9fcc168d-7a89-49fc-8c16-6af5449e9314
Manifolds.symplectic_inverse_times!(M, r2, p, q);

# ╔═╡ c710a52d-18fb-4609-bac9-c6b5f0ecfa61
norm(r - r2)

# ╔═╡ ef744311-f993-47b6-9858-9c45227c0e17
sizeof(p) / 10^3

# ╔═╡ 6ba126e2-3732-42e4-8e6e-a6e643c12539
@benchmark inv($M, $p) * $q

# ╔═╡ 48322c17-4752-45c2-a492-28f757885eed
@benchmark Manifolds.symplectic_inverse_times!($M, $r2, $p, $q)

# ╔═╡ 5b8ee8f9-6e83-4749-802e-dc399ee72a65
r2

# ╔═╡ Cell order:
# ╠═b9aec7da-4210-11ec-3f51-71546dd47b1d
# ╠═aaed81b5-c309-4c7d-98b5-842113e92b36
# ╠═1265db7d-f9f2-42b5-81af-99108c6521ae
# ╠═0767ca5a-8784-4708-aab0-773c8cdbfd14
# ╠═42d56c68-1ae9-4744-9c76-ab681c0d8f54
# ╠═9fcc168d-7a89-49fc-8c16-6af5449e9314
# ╠═c710a52d-18fb-4609-bac9-c6b5f0ecfa61
# ╠═ef744311-f993-47b6-9858-9c45227c0e17
# ╠═6ba126e2-3732-42e4-8e6e-a6e643c12539
# ╠═48322c17-4752-45c2-a492-28f757885eed
# ╠═5b8ee8f9-6e83-4749-802e-dc399ee72a65
