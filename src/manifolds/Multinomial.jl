@doc raw"""
    MultinomialMatrices{n,m} <: AbstractPowerManifold{ℝ}

The multinomial manifold consists of `m` column vectors, where each column is of length
`n` and unit norm, i.e.

````math
\mathcal{MN}(n,m) \coloneqq \bigl\{ p ∈ ℝ^{n×m}\ \big|\ p_{i,j} > 0 \text{ for all } i=1,…,n, j=1,…,m \text{ and } p^{\mathrm{T}}\mathbb{1}_m = \mathbb{1}_n\bigr\},
````
where $\mathbb{1}_k$ is the vector of length $k$ containing ones.

This yields exactly the same metric as
considering the product metric of the probablity vectors, i.e. [`PowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.PowerManifold) of the
$(n-1)$-dimensional [`ProbabilitySimplex`](@ref).

The [`ProbabilitySimplex`](@ref) is stored internally within `M.manifold`, such that all functions of
[`AbstractPowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractPowerManifold)  can be used directly.

# Constructor

    MultinomialMatrices(n, m)

Generate the manifold of matrices $\mathbb R^{n×m}$ such that the $m$ columns are
discrete probability distributions, i.e. sum up to one.
"""
struct MultinomialMatrices{N,M,S} <:
       AbstractPowerManifold{ℝ,ProbabilitySimplex{S},ArrayPowerRepresentation} where {N,M}
    manifold::ProbabilitySimplex{S}
end

function MultinomialMatrices(n::Int, m::Int)
    return MultinomialMatrices{n,m,n - 1}(ProbabilitySimplex(n - 1))
end

function Base.:^(M::ProbabilitySimplex{N}, m::Int) where {N}
    return MultinomialMatrices{manifold_dimension(M) + 1,m,N}(M)
end

@doc raw"""
    check_point(M::MultinomialMatrices, p)

Checks whether `p` is a valid point on the [`MultinomialMatrices`](@ref)`(m,n)` `M`, i.e. is a matrix
of `m` discrete probability distributions as columns from $\mathbb R^{n}$, i.e. each column is a point from
[`ProbabilitySimplex`](@ref)`(n-1)`.
"""
check_point(::MultinomialMatrices, ::Any)
function check_point(M::MultinomialMatrices{n,m}, p; kwargs...) where {n,m}
    return check_point(PowerManifold(M.manifold, m), p; kwargs...)
end

@doc raw"""
    check_vector(M::MultinomialMatrices p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialMatrices`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and columnswise
a tangent vector to the columns of `p` on the [`ProbabilitySimplex`](@ref).
"""
function check_vector(M::MultinomialMatrices{n,m}, p, X; kwargs...) where {n,m}
    return check_vector(PowerManifold(M.manifold, m), p, X; kwargs...)
end

get_iterator(::MultinomialMatrices{n,m}) where {n,m} = Base.OneTo(m)

@generated manifold_dimension(::MultinomialMatrices{n,m}) where {n,m} = (n - 1) * m
@generated power_dimensions(::MultinomialMatrices{n,m}) where {n,m} = (m,)

@generated representation_size(::MultinomialMatrices{n,m}) where {n,m} = (n, m)

function Base.show(io::IO, ::MultinomialMatrices{n,m}) where {n,m}
    return print(io, "MultinomialMatrices($(n),$(m))")
end
