@doc raw"""
    MultinomialMatrices{n,m} <: AbstractPowerManifold{ℝ}

The multinomial manifold consists of `m` column vectors, where each column is of length
`n` and unit norm, i.e.

````math
\mathcal{MN}(n,m) \coloneqq \bigl\{
    p ∈ ℝ^{n×m}\ \big|\ p_{i,j} > 0 \text{ for all } i=1,…,n, j=1,…,m
    \text{ and } p^{\mathrm{T}}\mathbb{1}_m = \mathbb{1}_n\bigr\},
````

where ``\mathbb{1}_k`` is the vector of length ``k`` containing ones.

This yields exactly the same metric as
considering the product metric of the probablity vectors, i.e. [`PowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.PowerManifold) of the
``(n-1)``-dimensional [`ProbabilitySimplex`](@ref).

The [`ProbabilitySimplex`](@ref) is stored internally within `M.manifold`, such that all functions of
[`AbstractPowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractPowerManifold)  can be used directly.

# Constructor

    MultinomialMatrices(n::Int, m::Int; parameter::Symbol=:type)

Generate the manifold of matrices ``ℝ^{n×m}`` such that the ``m`` columns are
discrete probability distributions, i.e. sum up to one.

`parameter`: whether a type parameter should be used to store `n` and `m`. By default size
is stored in type. Value can either be `:field` or `:type`.
"""
struct MultinomialMatrices{T,TPM<:ProbabilitySimplex} <:
       AbstractPowerManifold{ℝ,TPM,ArrayPowerRepresentation}
    size::T
    manifold::TPM
end

function MultinomialMatrices(n::Int, m::Int; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n, m))
    MPS = ProbabilitySimplex(n - 1; parameter=parameter)
    return MultinomialMatrices{typeof(size),typeof(MPS)}(size, MPS)
end

function Base.:^(::ProbabilitySimplex{TypeParameter{Tuple{N}}}, m::Int) where {N}
    return MultinomialMatrices(N + 1, m)
end
function Base.:^(M::ProbabilitySimplex{Tuple{Int}}, m::Int)
    n = get_parameter(M.size)[1]
    return MultinomialMatrices(n + 1, m; parameter=:field)
end

@doc raw"""
    check_point(M::MultinomialMatrices, p)

Checks whether `p` is a valid point on the [`MultinomialMatrices`](@ref)`(m,n)` `M`, i.e. is a matrix
of `m` discrete probability distributions as columns from ``ℝ^n``, i.e. each column is a point from
[`ProbabilitySimplex`](@ref)`(n-1)`.
"""
check_point(::MultinomialMatrices, ::Any)
function check_point(M::MultinomialMatrices, p; kwargs...)
    n, m = get_parameter(M.size)
    return check_point(PowerManifold(M.manifold, m), p; kwargs...)
end

@doc raw"""
    check_vector(M::MultinomialMatrices p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialMatrices`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and columnswise
a tangent vector to the columns of `p` on the [`ProbabilitySimplex`](@ref).
"""
function check_vector(M::MultinomialMatrices, p, X; kwargs...)
    n, m = get_parameter(M.size)
    return check_vector(PowerManifold(M.manifold, m), p, X; kwargs...)
end

function get_iterator(M::MultinomialMatrices)
    n, m = get_parameter(M.size)
    return Base.OneTo(m)
end

function manifold_dimension(M::MultinomialMatrices)
    n, m = get_parameter(M.size)
    return (n - 1) * m
end
function power_dimensions(M::MultinomialMatrices)
    n, m = get_parameter(M.size)
    return (m,)
end

@doc raw"""
    riemannian_gradient(M::MultinomialMatrices, p, Y; kwargs...)

Let ``Y`` denote the Euclidean gradient of a function ``\tilde f`` defined in the
embedding neighborhood of `M`, then the Riemannian gradient is given by
Equation 5 of [DouikHassibi:2019](@cite) as

```math
  \operatorname{grad} f(p) = \proj_{T_p\mathcal M}(Y⊙p)
```

where ``⊙`` denotes the Hadamard or elementwise product.

"""
riemannian_gradient(M::MultinomialMatrices, p, Y; kwargs...)

function riemannian_gradient!(M::MultinomialMatrices, X, p, Y; kwargs...)
    X .= p .* Y
    project!(M, X, p, X)
    return X
end

representation_size(M::MultinomialMatrices) = get_parameter(M.size)

function Base.show(io::IO, ::MultinomialMatrices{TypeParameter{Tuple{n,m}}}) where {n,m}
    return print(io, "MultinomialMatrices($(n), $(m))")
end
function Base.show(io::IO, M::MultinomialMatrices{Tuple{Int,Int}})
    n, m = get_parameter(M.size)
    return print(io, "MultinomialMatrices($(n), $(m); parameter=:field)")
end
