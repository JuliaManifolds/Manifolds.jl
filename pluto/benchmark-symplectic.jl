### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ b9aec7da-4210-11ec-3f51-71546dd47b1d
begin
	using Pkg
	 Pkg.activate()# use global environment to have Manifolds in dev mode
	using Manifolds, PlutoUI, BenchmarkTools, Random
end

# ╔═╡ aaed81b5-c309-4c7d-98b5-842113e92b36
begin
	n=500
	k=20
end;

# ╔═╡ 1265db7d-f9f2-42b5-81af-99108c6521ae
M = SymplecticStiefel(n,k)

# ╔═╡ 0767ca5a-8784-4708-aab0-773c8cdbfd14
begin
    Random.seed!(42)
	p = rand(M)
	q = allocate(p)
	r = allocate(p)
    X = rand(M, p)
    Y = rand(M, p)
	m = CayleyRetraction()
end;

# ╔═╡ 42d56c68-1ae9-4744-9c76-ab681c0d8f54
inner(M, p, X, Y) - Manifolds.inner_old(M, p, X, Y)

# ╔═╡ 9fcc168d-7a89-49fc-8c16-6af5449e9314
@benchmark inner($M, $p, $X, $Y)

# ╔═╡ c710a52d-18fb-4609-bac9-c6b5f0ecfa61
@benchmark Manifolds.inner_old($M, $p, $X, $Y)

# ╔═╡ a7b1061a-1c5a-40c8-81f9-3d77f6530abe
begin
	Manifolds.retract_old!(M, q, p, X,m)
	retract!(M, r, p, X,m)
	norm(r-q)
end

# ╔═╡ 2d30d83b-313e-4b06-aac8-1d2a236a4829
@benchmark retract!(M, r, p, X,m)

# ╔═╡ 2a5fb247-20af-4e68-8a5e-d4dd18e2cc3c
@benchmark Manifolds.retract_old!(M, q, p, X,m)

# ╔═╡ Cell order:
# ╠═b9aec7da-4210-11ec-3f51-71546dd47b1d
# ╠═aaed81b5-c309-4c7d-98b5-842113e92b36
# ╠═1265db7d-f9f2-42b5-81af-99108c6521ae
# ╠═0767ca5a-8784-4708-aab0-773c8cdbfd14
# ╠═42d56c68-1ae9-4744-9c76-ab681c0d8f54
# ╠═9fcc168d-7a89-49fc-8c16-6af5449e9314
# ╠═c710a52d-18fb-4609-bac9-c6b5f0ecfa61
# ╠═a7b1061a-1c5a-40c8-81f9-3d77f6530abe
# ╠═2d30d83b-313e-4b06-aac8-1d2a236a4829
# ╠═2a5fb247-20af-4e68-8a5e-d4dd18e2cc3c
