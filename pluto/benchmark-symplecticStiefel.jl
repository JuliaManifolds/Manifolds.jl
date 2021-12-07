### A Pluto.jl notebook ###
# v0.17.3

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
    q = allocate(p)
	pI = allocate(p')
    r2 = zeros(k,k)
    X = rand(M, p)
    Y = rand(M, p)
end;

# ╔═╡ 42d56c68-1ae9-4744-9c76-ab681c0d8f54
begin
	inv!(M, pI, p)
	r = pI*q
end

# ╔═╡ 6ba126e2-3732-42e4-8e6e-a6e643c12539
@benchmark begin
	inv!($M, $pI, $p)
	r = $pI*$q
end

# ╔═╡ 9fcc168d-7a89-49fc-8c16-6af5449e9314
Manifolds.symplectic_inverse_times!(M, r2, p, q);

# ╔═╡ 48322c17-4752-45c2-a492-28f757885eed
@benchmark Manifolds.symplectic_inverse_times!($M, $r2, $p, $q)

# ╔═╡ c710a52d-18fb-4609-bac9-c6b5f0ecfa61
norm(r - r2)

# ╔═╡ a7b1061a-1c5a-40c8-81f9-3d77f6530abe
begin
    Manifolds.retract_old!(M, q, p, X, m)
    retract!(M, r, p, X, m)
    norm(r - q) # since we do not have a distance, let's do the poor-mans check.
end

# ╔═╡ 2d30d83b-313e-4b06-aac8-1d2a236a4829
@benchmark retract!($M, $r, $p, $X, $m)

# ╔═╡ 2a5fb247-20af-4e68-8a5e-d4dd18e2cc3c
@benchmark Manifolds.retract_old!($M, $q, $p, $X, $m)

# ╔═╡ 73f1adf0-5c71-484c-b467-e4dd15911c0e
invp = similar(p');

# ╔═╡ 43e60e53-1924-4d65-8474-ba8051e3cc89
@benchmark inv!($M, $invp, $p)

# ╔═╡ 3d325328-123b-46cc-8cbc-e7c09be71bc0
invp2 = similar(p');

# ╔═╡ 556cfbc7-5c04-429e-ac7e-78b658ad9c7a
@benchmark Manifolds.old_inv!($M, $invp2, $p)

# ╔═╡ 8c2035c0-e8b7-4fd3-818e-cbbce4312765
norm(invp - invp2)

# ╔═╡ Cell order:
# ╠═b9aec7da-4210-11ec-3f51-71546dd47b1d
# ╠═aaed81b5-c309-4c7d-98b5-842113e92b36
# ╠═1265db7d-f9f2-42b5-81af-99108c6521ae
# ╠═0767ca5a-8784-4708-aab0-773c8cdbfd14
# ╠═42d56c68-1ae9-4744-9c76-ab681c0d8f54
# ╠═6ba126e2-3732-42e4-8e6e-a6e643c12539
# ╠═9fcc168d-7a89-49fc-8c16-6af5449e9314
# ╠═48322c17-4752-45c2-a492-28f757885eed
# ╠═c710a52d-18fb-4609-bac9-c6b5f0ecfa61
# ╠═a7b1061a-1c5a-40c8-81f9-3d77f6530abe
# ╠═2d30d83b-313e-4b06-aac8-1d2a236a4829
# ╠═2a5fb247-20af-4e68-8a5e-d4dd18e2cc3c
# ╠═73f1adf0-5c71-484c-b467-e4dd15911c0e
# ╠═43e60e53-1924-4d65-8474-ba8051e3cc89
# ╠═3d325328-123b-46cc-8cbc-e7c09be71bc0
# ╠═556cfbc7-5c04-429e-ac7e-78b658ad9c7a
# ╠═8c2035c0-e8b7-4fd3-818e-cbbce4312765
