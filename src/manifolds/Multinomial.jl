@doc raw"""
    MultinomialMatrices{n,m} <: AbstractPowerManifold{ℝ}

The multinomial manifold consists of `m` column vectors, where each column is of length
`n` and unit norm, i.e.

````math
\mathcal{MN}(n,m) \coloneqq \bigl\{ p ∈ ℝ^{n×m}\ \big|\ p_{i,j} > 0 \text{ for all } i=1,…,n, j=1,…,m \text{ and } p^{\mathrm{T}}\mathbb{1}_m = \mathbb{1}_n\bigr\},
````
where $\mathbb{1}_k$ is the vector of length $k$ containing ones.

This yields exactly the same metric as
considering the product metric of the probablity vectors, i.e. [`PowerManifold`](@ref) of the
$(n-1)$-dimensional [`ProbabilitySimplex`](@ref).

The [`ProbabilitySimplex`](@ref) is stored internally within `M.manifold`, such that all functions of
[`AbstractPowerManifold`](@ref) can be used directly.

# Constructor

    MultinomialMatrices(n,m)

Generate the manifold of matrices $\mathbb R^{n×m}$ such that the $m$ columns are
discrete probability distributions, i.e. sum up to one.
"""
struct MultinomialMatrices{N,M,S} <:
       AbstractPowerManifold{ℝ,ProbabilitySimplex{S},ArrayPowerRepresentation} where {N,M}
    manifold::ProbabilitySimplex{S}
end

MultinomialMatrices(n::Int, m::Int) =
    MultinomialMatrices{n,m,n - 1}(ProbabilitySimplex(n - 1))

^(M::ProbabilitySimplex{N}, m::Int) where {N} =
    MultinomialMatrices{manifold_dimension(M) + 1,m,N}(M)

@doc raw"""
    check_manifold_point(M::MultinomialMatrices,p)

Checks whether `p` is a valid point on the [`MultinomialMatrices`](@ref)`{m,n}` `M`, i.e. is a matrix
of `m` discrete probability distributions as columns from $\mathbb R^{n}$, i.e. each column is a point from
[`ProbabilitySimplex`](@ref)`(n-1)`.
"""
check_manifold_point(::MultinomialMatrices, ::Any)
function check_manifold_point(M::MultinomialMatrices{n,m}, p; kwargs...) where {n,m}
    if size(p) != (n, m)
        return DomainError(
            length(p),
            "The matrix in `p` ($(size(p))) does not match the dimensions of $(M).",
        )
    end
    return check_manifold_point(PowerManifold(M.manifold, m), p; kwargs...)
end
@doc raw"""
    check_tangent_vector(M::MultinomialMatrices p, X; check_base_point = true, kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialMatrices`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and columnswise
a tangent vector to the columns of `p` on the [`ProbabilitySimplex`](@ref).
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
"""
function check_tangent_vector(
    M::MultinomialMatrices{n,m},
    p,
    X;
    check_base_point = true,
    kwargs...,
) where {n,m}
    if check_base_point && size(p) != (n, m)
        return DomainError(
            length(p),
            "The matrix `p` ($(size(p))) does not match the dimension of $(M).",
        )
    end
    if size(X) != (n, m)
        return DomainError(
            length(X),
            "The matrix `X` ($(size(X))) does not match the dimension of $(M).",
        )
    end
    return check_tangent_vector(
        PowerManifold(M.manifold, m),
        p,
        X;
        check_base_point = check_base_point,
        kwargs...,
    )
end

get_iterator(M::MultinomialMatrices{n,m}) where {n,m} = Base.OneTo(m)

@generated manifold_dimension(::MultinomialMatrices{n,m}) where {n,m} = (n - 1) * m

@generated representation_size(::MultinomialMatrices{n,m}) where {n,m} = (n, m)

show(io::IO, ::MultinomialMatrices{n,m}) where {n,m} =
    print(io, "MultinomialMatrices($(n),$(m))")
