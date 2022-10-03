### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

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

# ╔═╡ c96935ca-6bda-466d-ad29-b40c19f55392
using Manifolds

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

# ╔═╡ 0066a636-2a06-4891-b807-8b354827ad0a
is_point(M₃, [0, 0, 1.001]; atol=1e-3)

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
"""

# ╔═╡ 114b46c3-654d-4b1c-b8a9-3acc5939a25e

# ╔═╡ a68af8e4-82d0-4d55-ad39-461688c86b95
md"""
## Implementing generic functions

In this section
"""

# ╔═╡ 592549a7-5de7-452d-9dfa-fc748afc8b04

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
# ╠═6d8a6b23-2ab8-4a70-b303-eda3f490efee
# ╠═9f1482b8-d345-4e65-a2a0-9fd38a251df5
# ╟─e8f068e4-f11c-480e-86ab-9934263d1c06
# ╠═ff2cf9f6-712b-4c67-9d65-92412558b6e4
# ╟─082e751c-eaa5-4c31-9589-aada0d417a66
# ╠═90832504-4eaf-49d3-9c59-6b219121c6ef
# ╠═ba9320d3-a340-4b36-95ac-2a9935803f44
# ╟─a9883394-e1bb-4cef-bae5-ce34f6e821d8
# ╠═1c3c993c-4c49-4baa-b84f-eb42cd481620
# ╠═114b46c3-654d-4b1c-b8a9-3acc5939a25e
# ╟─a68af8e4-82d0-4d55-ad39-461688c86b95
# ╠═592549a7-5de7-452d-9dfa-fc748afc8b04
