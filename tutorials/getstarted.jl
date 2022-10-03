### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ c96935ca-6bda-466d-ad29-b40c19f55392
using Manifolds

# ╔═╡ 41cbc7c8-3a39-11ed-292e-0bb253a3b2f3
md"""
# 🚀 Get Started with `Manifolds.jl`

This is a short overview of [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/).

This tutorial is rendered from a pluto notebook, so you can also open the file [tutorials/getstarted.jl](https://github.com/JuliaManifolds/Manifolds.jl/tree/master/tutorials/getstarted.jl) in Pluto and work on this tutorial interactively.

As usual, if you want to install the package, just type

```
] add Manifolds
```

in Julia REPL or use

```
using Pkg; Pkg.add("Manifolds");
```

before the first use. Then load the package with
"""

# ╔═╡ 9d16efde-bd95-46d9-a659-5420fe860699
md"""
Since the packagae hevily depends on [`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/) we will sometimes also link to the interface definition of functions in the interface and mark this with 🔗.
"""

# ╔═╡ b34d2b6c-907e-45b3-9b62-445666413b26
md"""
## Contents
* [Using the library of manifolds](#using-the-library-of-manifolds)
* [implementing generic functions](#implementing-generic-functions)
* the exponential map and retractions
* the logarithmic map, parallel transport and its
"""

# ╔═╡ c1e139b0-7d39-4d20-81dc-5592fee831d0
md"""
## Using the Library of Manifolds

[`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/) is first of all a library of manifolds, see the list in the menu [here](https://juliamanifolds.github.io/Manifolds.jl/latest/) under “basic manifolds”.

Let's look at three examples together with the first few functions on manifolds.
"""

# ╔═╡ 7a3d7f18-75b2-4c0b-ac4f-8c5d5e27b4f6
md"#### 1. [The Euclidean space](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/euclidean.html)

[The Euclidean space](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/euclidean.html#Manifolds.Euclidean) brings us (back) into linear case of vectors, so in terms of manifolds, this is a very simple one. It is often useful to compare to classical algorithms, or implementations."

# ╔═╡ 554a8a25-92bd-4603-9f23-1afd18dfc658
M₁ = Euclidean(3)

# ╔═╡ 6341255c-f641-4b91-a7a9-e052183a5791
md"""
Note that since a manifold is a type in Julia, we write it in CamelCase. Its parameters are first a dimension or size parameter of the manifold, sometimes optional is a field the manifold is defined over.

For example the above definition is the same as the real-valued case
"""

# ╔═╡ fef3b6a6-b19b-4fac-9ffe-aa45a4bc547a
M₁ === Euclidean(3, field=ℝ)

# ╔═╡ 338465ed-3055-45b7-a7e1-304a7ac856b5
md"But we even introduced a short hand notation, since ℝ is also just a symbol/variable to use"

# ╔═╡ 6360598f-5280-4327-ab0c-50bd401ed5d6
M₁ === ℝ^3

# ╔═╡ 088293e9-ebff-49e3-868a-ed824de857fa
md"And similarly here are two ways to create the manifold of vectors of length two with complex entries – or mathematically the space ``\mathbb C^2``"

# ╔═╡ 657bce13-5cf2-438f-9c12-4434fa1850ac
Euclidean(2, field=ℂ) === ℂ^2

# ╔═╡ 57c6fb90-03fc-487d-a8e7-02108097cc78
md"""
The easiest to check is the dimension of a manifold. Here we have three “directions to walk into” at every point ``p\in \mathbb R
^3`` so [`manifold_dimension`]() ([🔗](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.maniold_dimension-Tuple{AbstractManifold})) is
"""

# ╔═╡ 316b2d4f-984c-4969-b515-0772ec89a745
manifold_dimension(M₁)

# ╔═╡ 78f1ae49-a973-4b39-a058-720e12532283
md"""
#### 2. [The hyperpolic space](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/hyperbolic.html)

The ``d``-dimensional [hyperbolic space](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/hyperbolic.html#Manifolds.Hyperbolic) is usually represented in ``\mathbb R^{d+1}`` as the set of points ``p\in\mathbb R^3`` fulfilling
```math
p_1^2+p_2^2+\cdots+p_d^2-p_{d+1}^2 = -1.
```
"""

# ╔═╡ 1025b30d-3433-4335-8751-658e7731d424
M₂ = Hyperbolic(2)

# ╔═╡ 77101edd-4870-4b45-88f2-20b48a07fd57
manifold_dimension(M₂)

# ╔═╡ 588e67af-8335-47e5-ba34-ad1cfd22a69d
md"""
Here, a useful function is to check, whether some ``p∈\mathbb R^3`` is a point on the manifold. We can check
"""

# ╔═╡ dcce82a5-f7bb-4ebb-89cb-a66900c873fd
is_point(M₂, [0, 0, 1])

# ╔═╡ c07a05df-9d0c-4810-9539-a5fdd7640f45
is_point(M₂, [1, 0, 1])

# ╔═╡ 908d0ee4-73c0-4f8a-b9b4-5b42aec8559b
md"Keyword arguments are passed on to any numerical checks, for example an absolute tolerance when checking the above equiality."

# ╔═╡ 4880eaaf-6cf0-4250-8056-6d5b220e963c
md"""
But in an interactive session an error message might be helpful. A positional (third) argument is present to activate this. Here we illustrate this with try-catch to keep the notebook as valid running code.
"""

# ╔═╡ d3caea7a-89ff-4f04-94e9-922048ad0bb1
try
    is_point(M₂, [0, 0, 1.001], true)
catch e #We just have to trick a litte to display the Domain error here
    if isa(e, DomainError)
        Markdown.parse("""```
        $(e)
        ```""")
    else
        rethrow(e)
    end
end

# ╔═╡ 19cbc8c5-4c2c-4594-bbb5-30f268c046cc
md"""
#### 3. [The sphere](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/sphere.html)

[The sphere](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/sphere.html#Manifolds.Sphere) ``\mathbb S^d`` is the ``d``-dimensional sphere represented in its embedded form, that is unit vectors ``p \in \mathbb R^{d+1}`` with unit norm ``\lVert p \rVert_2 = 1``.
"""

# ╔═╡ f689ac55-7c5d-4197-90b6-6c32591482d7
M₃ = Sphere(2)

# ╔═╡ 0066a636-2a06-4891-b807-8b354827ad0a
is_point(M₃, [0, 0, 1.001]; atol=1e-3)

# ╔═╡ 6d8a6b23-2ab8-4a70-b303-eda3f490efee
md"""
Here we can show a last nice check: [`is_vector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.is_vector) to check whether a tangent vector `X` is a representation of a tangent vector ``X∈T_p\mathcal M`` to a point `p` on the manifold.

This function has two positional asrguments, the first to again indicate whether to throw an error, the second to disable the check that `p` is a valid point on the manifold. Usually this validity is essential for the tangent check, but if it was for example performed before, it can be turned off to spare time.

For example in our first example the point is not of unit norm
"""

# ╔═╡ 9f1482b8-d345-4e65-a2a0-9fd38a251df5
is_vector(M₃, [2, 0, 0], [0, 1, 1])

# ╔═╡ e8f068e4-f11c-480e-86ab-9934263d1c06
md"But the orthogonality of `p` and `X` is still valid, so we get"

# ╔═╡ ff2cf9f6-712b-4c67-9d65-92412558b6e4
is_vector(M₃, [2, 0, 0], [0, 1, 1], true, false)

# ╔═╡ 082e751c-eaa5-4c31-9589-aada0d417a66
md"But of course it is better to use a valid point in the first place"

# ╔═╡ 90832504-4eaf-49d3-9c59-6b219121c6ef
is_vector(M₃, [1, 0, 0], [0, 1, 1])

# ╔═╡ ba9320d3-a340-4b36-95ac-2a9935803f44
try
    is_vector(M₃, [1, 0, 0], [0.1, 1, 1], true)
catch e #We just have to trick a litte to display the Domain error here
    if isa(e, DomainError)
        Markdown.parse("""```
        $(e)
        ```""")
    else
        rethrow(e)
    end
end

# ╔═╡ a9883394-e1bb-4cef-bae5-ce34f6e821d8
md"To learn about how to define a manifold youself check out the [How to define your own manifold](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/example.html) tutorial of [`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/)."

# ╔═╡ 1c3c993c-4c49-4baa-b84f-eb42cd481620
md"""
### Building more advanced manifolds

Based on these basic manifolds we can directly build more advanced manifolds.

The first one concerns vectors or matrices of data on a manifold, the [PowerManifold](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#sec-power-manifold).
"""

# ╔═╡ 114b46c3-654d-4b1c-b8a9-3acc5939a25e
M₄ = M₂^2

# ╔═╡ 1c7deff8-f9bf-4e11-9c40-47a6970ee502
md"Then points are represented by arrays, where the power manifold dimension is added in the end. In other words – for the hyperbolic manifold here, we have a matrix with 2 columns, where each column is a valid point on hyperbolic space."

# ╔═╡ 741ceb0c-f5e8-438e-b3f1-857cae2d451d
p = [0 0; 0 1; 1 sqrt(2)]

# ╔═╡ 59a99512-cb9c-4106-ad1b-69bf947d604b
[is_point(M₂, p[:, 1]), is_point(M₂, p[:, 2])]

# ╔═╡ ff62cb96-d1ae-4fed-9101-e728d9f2fe18
md"But of course the method we used previously also works for power manifolds:"

# ╔═╡ 81e0cc2b-1e79-411c-aee1-d1761d0b95d2
is_point(M₄, p)

# ╔═╡ 8e85742c-fa06-4212-bace-81479b31d9e9
md"Note that nested power manifolds are combined into one as in"

# ╔═╡ 262f0ca4-4ef5-4670-a3d2-c86a74884d97
M₄₂ = M₄^4

# ╔═╡ 0c46f29e-126b-42c0-a96c-24f369c5fd80
md"which represents ``2\times 4`` – matrices of hyperbolic points represented in ``3\times 2\times 4`` arrays."

# ╔═╡ 0d56dddc-0f52-48e1-b81f-bc08fbcdfddf
md"We can – alternatively – use a power manifold with nested arrays"

# ╔═╡ 01cae5a1-0d0d-4163-8713-52efb79d5043
M₅ = PowerManifold(M₃, NestedPowerRepresentation(), 2)

# ╔═╡ 33bab279-4258-4402-a4e5-da6fec1a88bf
md"which emphasizes that we have vectors of length 2 that contain points, so we store them that way."

# ╔═╡ e20ec2bc-7488-4d09-9a3a-17c06d6a4bfa
p₂ = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

# ╔═╡ 58eb08aa-1963-4b01-bbee-48ca9ae68e9e
md"Top unify both representations, elements of the power manifold can also be accessed in the classical indexing fashion, if we start with the corresponding manifold first. This way one can implement algorithms also independent of which representation is used."

# ╔═╡ bd676b86-060d-4e94-af01-fa1fe88a4cc7
p[M₄, 1]

# ╔═╡ 44840536-f75e-4db9-ad70-ded60beac987
p₂[M₅, 2]

# ╔═╡ 5570facf-5d04-42fc-bc40-28acad1fbc40
md"""
Another construtor is the [ProductManifold](https://juliamanifolds.github.io/Manifolds.jl/previews/PR534/manifolds/product.html) to combine different manifolds. Here of course the order matters.
First we construct these using ``×``
"""

# ╔═╡ a02f085b-eaef-4a8f-a8fe-e306e956dd2c
M₆ = M₂ × M₃

# ╔═╡ 0f135d94-25c8-499d-a9e0-169db76cb901
md"Since now the representations might differ from element to element, we have to encapsulate these in their own type."

# ╔═╡ fa0baedb-636e-4ac8-9779-039625ca8267
p₃ = ProductRepr([0, 0, 1], [0, 1, 0])

# ╔═╡ 3e1e665f-ae85-4a4c-9f7d-4db6b9beada8
is_point(M₆, p₃, true)

# ╔═╡ 82b06119-d5f6-4aca-bd6c-45251144e4bf
md"But accessing single components still works the same."

# ╔═╡ b253ac94-d09a-4f5e-87d6-61275b81f8f4
p₃[M₆, 1]

# ╔═╡ 0e8778d6-b550-4d3d-9788-15c2c52c502c
md"Finally, also the [TangentBundle](https://juliamanifolds.github.io/Manifolds.jl/previews/PR534/manifolds/vector_bundle.html#Manifolds.TangentBundle), the manifold collecting all tangent spaces on a manifold is available as"

# ╔═╡ ab650dfb-f156-4a57-b85b-419b4b65c9c4
M₇ = TangentBundle(M₃)

# ╔═╡ a68af8e4-82d0-4d55-ad39-461688c86b95
md"""
## Implementing generic functions

In this section
"""

# ╔═╡ 592549a7-5de7-452d-9dfa-fc748afc8b04

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"

[compat]
Manifolds = "~0.8.29"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.1"
manifest_format = "2.0"
project_hash = "71acc07ae88ce70be87ae3e5f6370e01ce20d018"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5bb0f8292405a516880a3809954cb832ae7a31c5"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.20"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "a1e2cf6ced6505cbad2490532388683f1e88c3ed"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "3c8de95b4e932d76ec8960e12d681eba580e9674"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.8"

[[deps.DataAPI]]
git-tree-sha1 = "1106fa7e1256b402a86a8e7b15c00c85036fef49"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.11.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0d7d213133d948c56e8c2d9f4eab0293491d8e4a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.75"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "87519eb762f85534445f5cda35be12e32759ee14"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.4"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ba2d094a88b6b287bd25cfa86f301e7693ffae2f"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.4"

[[deps.HybridArrays]]
deps = ["LinearAlgebra", "Requires", "StaticArrays"]
git-tree-sha1 = "898f2ae8e920084a7e05963b263b0d9c6ceafc23"
uuid = "1baab800-613f-4b0a-84e4-9cd3431bfbb9"
version = "0.4.12"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.Kronecker]]
deps = ["LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "78d9909daf659c901ae6c7b9de7861ba45a743f4"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "d1b46faefb7c2f48fdec69e6f3cc34857769bc15"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.8.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Manifolds]]
deps = ["Colors", "Distributions", "Einsum", "Graphs", "HybridArrays", "Kronecker", "LinearAlgebra", "ManifoldsBase", "Markdown", "MatrixEquations", "Quaternions", "Random", "RecipesBase", "RecursiveArrayTools", "Requires", "SimpleWeightedGraphs", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "f2d8872d3202d2ef34cc687b0c5bca44fc82d432"
uuid = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
version = "0.8.29"

[[deps.ManifoldsBase]]
deps = ["LinearAlgebra", "Markdown"]
git-tree-sha1 = "3e0dc4ae917310b1d7952c8741ecfd9942c93540"
uuid = "3362f125-f0bb-47a3-aa74-596ffd7ef2fb"
version = "0.13.20"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixEquations]]
deps = ["LinearAlgebra", "LinearMaps"]
git-tree-sha1 = "3b284e9c98f645232f9cf07d4118093801729d43"
uuid = "99c1a7ee-ab34-5fd5-8076-27c950a045f4"
version = "2.2.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "cb8ebcee2b4e07b72befb9def593baef8aa12f07"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.50"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "3c009334f45dfd546a16a57960a821a1a023d241"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.5.0"

[[deps.Quaternions]]
deps = ["DualNumbers", "LinearAlgebra", "Random"]
git-tree-sha1 = "4ab19353944c46d65a10a75289d426ef57b0a40c"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.5.7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "612a4d76ad98e9722c8ba387614539155a59e30c"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.0"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "Tables", "ZygoteRules"]
git-tree-sha1 = "3004608dc42101a944e44c1c68b599fa7c669080"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.32.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "f86b3a049e5d05227b10e15dbb315c5b90f14988"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.9"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "2d7164f7b8a066bcfa6224e67736ce0eb54aef5b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.9.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─41cbc7c8-3a39-11ed-292e-0bb253a3b2f3
# ╠═c96935ca-6bda-466d-ad29-b40c19f55392
# ╟─9d16efde-bd95-46d9-a659-5420fe860699
# ╟─b34d2b6c-907e-45b3-9b62-445666413b26
# ╟─c1e139b0-7d39-4d20-81dc-5592fee831d0
# ╟─7a3d7f18-75b2-4c0b-ac4f-8c5d5e27b4f6
# ╠═554a8a25-92bd-4603-9f23-1afd18dfc658
# ╟─6341255c-f641-4b91-a7a9-e052183a5791
# ╠═fef3b6a6-b19b-4fac-9ffe-aa45a4bc547a
# ╟─338465ed-3055-45b7-a7e1-304a7ac856b5
# ╠═6360598f-5280-4327-ab0c-50bd401ed5d6
# ╟─088293e9-ebff-49e3-868a-ed824de857fa
# ╠═657bce13-5cf2-438f-9c12-4434fa1850ac
# ╟─57c6fb90-03fc-487d-a8e7-02108097cc78
# ╠═316b2d4f-984c-4969-b515-0772ec89a745
# ╟─78f1ae49-a973-4b39-a058-720e12532283
# ╠═1025b30d-3433-4335-8751-658e7731d424
# ╠═77101edd-4870-4b45-88f2-20b48a07fd57
# ╟─588e67af-8335-47e5-ba34-ad1cfd22a69d
# ╠═dcce82a5-f7bb-4ebb-89cb-a66900c873fd
# ╠═c07a05df-9d0c-4810-9539-a5fdd7640f45
# ╟─908d0ee4-73c0-4f8a-b9b4-5b42aec8559b
# ╠═0066a636-2a06-4891-b807-8b354827ad0a
# ╟─4880eaaf-6cf0-4250-8056-6d5b220e963c
# ╠═d3caea7a-89ff-4f04-94e9-922048ad0bb1
# ╟─19cbc8c5-4c2c-4594-bbb5-30f268c046cc
# ╠═f689ac55-7c5d-4197-90b6-6c32591482d7
# ╟─6d8a6b23-2ab8-4a70-b303-eda3f490efee
# ╠═9f1482b8-d345-4e65-a2a0-9fd38a251df5
# ╟─e8f068e4-f11c-480e-86ab-9934263d1c06
# ╠═ff2cf9f6-712b-4c67-9d65-92412558b6e4
# ╟─082e751c-eaa5-4c31-9589-aada0d417a66
# ╠═90832504-4eaf-49d3-9c59-6b219121c6ef
# ╠═ba9320d3-a340-4b36-95ac-2a9935803f44
# ╟─a9883394-e1bb-4cef-bae5-ce34f6e821d8
# ╟─1c3c993c-4c49-4baa-b84f-eb42cd481620
# ╠═114b46c3-654d-4b1c-b8a9-3acc5939a25e
# ╟─1c7deff8-f9bf-4e11-9c40-47a6970ee502
# ╠═741ceb0c-f5e8-438e-b3f1-857cae2d451d
# ╠═59a99512-cb9c-4106-ad1b-69bf947d604b
# ╟─ff62cb96-d1ae-4fed-9101-e728d9f2fe18
# ╠═81e0cc2b-1e79-411c-aee1-d1761d0b95d2
# ╟─8e85742c-fa06-4212-bace-81479b31d9e9
# ╠═262f0ca4-4ef5-4670-a3d2-c86a74884d97
# ╟─0c46f29e-126b-42c0-a96c-24f369c5fd80
# ╠═0d56dddc-0f52-48e1-b81f-bc08fbcdfddf
# ╠═01cae5a1-0d0d-4163-8713-52efb79d5043
# ╠═33bab279-4258-4402-a4e5-da6fec1a88bf
# ╠═e20ec2bc-7488-4d09-9a3a-17c06d6a4bfa
# ╟─58eb08aa-1963-4b01-bbee-48ca9ae68e9e
# ╠═bd676b86-060d-4e94-af01-fa1fe88a4cc7
# ╠═44840536-f75e-4db9-ad70-ded60beac987
# ╠═5570facf-5d04-42fc-bc40-28acad1fbc40
# ╠═a02f085b-eaef-4a8f-a8fe-e306e956dd2c
# ╠═0f135d94-25c8-499d-a9e0-169db76cb901
# ╠═fa0baedb-636e-4ac8-9779-039625ca8267
# ╠═3e1e665f-ae85-4a4c-9f7d-4db6b9beada8
# ╠═82b06119-d5f6-4aca-bd6c-45251144e4bf
# ╠═b253ac94-d09a-4f5e-87d6-61275b81f8f4
# ╟─0e8778d6-b550-4d3d-9788-15c2c52c502c
# ╟─ab650dfb-f156-4a57-b85b-419b4b65c9c4
# ╟─a68af8e4-82d0-4d55-ad39-461688c86b95
# ╠═592549a7-5de7-452d-9dfa-fc748afc8b04
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
