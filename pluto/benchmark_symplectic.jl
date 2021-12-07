### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 85f0da8c-43e6-11ec-0fd5-ad40fcfed441
begin
    using Pkg
    Pkg.activate()# use global environment to have Manifolds in dev mode
    using Manifolds, BenchmarkTools, Random, Revise
    Random.seed!(42)
end

# ╔═╡ bbda868e-5a2b-4870-980c-1c68080bdd86
M = Symplectic(1000)

# ╔═╡ 1ea6d03e-9b0d-4081-945b-2b0dd861313f
p = rand(M);

# ╔═╡ 711e53e8-654e-49a9-b540-61424229e050
p1, p2 = copy(p), copy(p');

# ╔═╡ a1449cee-7f96-4af7-a0fc-3d754c3e1881
@benchmark Manifolds.inv(M, p1)

# ╔═╡ 2c58e4ef-3cfe-4d34-b1fc-f7fcb173fdb0
@benchmark Manifolds.inv!(M, p2, p1)

# ╔═╡ Cell order:
# ╠═85f0da8c-43e6-11ec-0fd5-ad40fcfed441
# ╠═bbda868e-5a2b-4870-980c-1c68080bdd86
# ╠═1ea6d03e-9b0d-4081-945b-2b0dd861313f
# ╠═711e53e8-654e-49a9-b540-61424229e050
# ╠═a1449cee-7f96-4af7-a0fc-3d754c3e1881
# ╠═2c58e4ef-3cfe-4d34-b1fc-f7fcb173fdb0
