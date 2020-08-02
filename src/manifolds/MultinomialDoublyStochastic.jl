@doc raw"""
    MultinomialDoublyStochastic{n} <: AbstractPowerManifold{ℝ}

The multinomial manifold consists of `m` column vectors, where each column is of length
`n` and unit norm, i.e.

````math
\mathcal{PS}(n) \coloneqq \bigl\{ p ∈ ℝ^{n×n}\ \big|\ p_{i,j} > 0 \text{ for all } i=1,…,n, j=1,…,m, p^\mathrm{T} = p, p\text{ is symmetric positive definite} \text{ and } p\mathbb{1}_n = p^{\mathrm{T}}\mathbb{1}_n = \mathbb{1}_n\bigr\},
````
where $\mathbb{1}_n$ is the vector of length $n$ containing ones.

The tangent space can be written as
````math
T_p\mathcal{PS}(n) \coloneqq \bigl\{
X ∈ ℝ^{n×n}\ \big|\
X = X^{\mathrm{T}} \text{ and } X \mathbb{1}_n = \mathbb{0}_n
\bigr\},
````
where $\mathbb{0}_n$ is the vector of length $n$ containing zeros.

# Constructor

    MultinomialSymmetricDoubleStochasticMatrices(n)

Generate the manifold of matrices $\mathbb R^{n×n}$ that are doubly stochastic and symmetric.
"""
struct MultinomialSymmetricDoubleStochasticMatrices{N} <:
       AbstractEmbeddedManifold{ℝ,DefaultEmbedding} where {N}
end

function MultinomialSymmetricDoubleStochasticMatrices(n::Int)
    return MultinomialSymmetricDoubleStochasticMatrices{n}()
end

@doc raw"""
    check_manifold_point(M::MultinomialSymmetricDoubleStochasticMatrices, p)

Checks whether `p` is a valid point on the [`MultinomialSymmetricDoubleStochasticMatrices`](@ref)`(m,n)` `M`, i.e. is a matrix
of `m` discrete probability distributions as columns from $\mathbb R^{n}$, i.e. each column is a point from
[`ProbabilitySimplex`](@ref)`(n-1)`.
"""
check_manifold_point(::MultinomialSymmetricDoubleStochasticMatrices, ::Any)
function check_manifold_point(M::MultinomialSymmetricDoubleStochasticMatrices{n,m}, p; kwargs...) where {n,m}
    if size(p) != (n, m)
        return DomainError(
            length(p),
            "The matrix in `p` ($(size(p))) does not match the dimensions of $(M).",
        )
    end
    return check_manifold_point(PowerManifold(M.manifold, m), p; kwargs...)
end
@doc raw"""
    check_tangent_vector(M::MultinomialSymmetricDoubleStochasticMatrices p, X; check_base_point = true, kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialSymmetricDoubleStochasticMatrices`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and columnswise
a tangent vector to the columns of `p` on the [`ProbabilitySimplex`](@ref).
The optional parameter `check_base_point` indicates, whether to call
[`check_manifold_point`](@ref check_manifold_point(::MultinomialSymmetricDoubleStochasticMatrices, ::Any))  for `p`.
"""
function check_tangent_vector(
    M::MultinomialSymmetricDoubleStochasticMatrices{n,m},
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

decorated_manifold(::MultinomialSymmetricDoubleStochasticMatrices{N}) where {N} = Euclidean(N, N; field = ℝ)

function distance(::MultinomialSymmetricDoubleStochasticMatrices, p, q)
    return sum( 2 * acos.(p.*q) )
end

embed!(::MultinomialSymmetricDoubleStochasticMatrices, q, p) = copyto!(q, p)
embed!(::MultinomialSymmetricDoubleStochasticMatrices, Y, p, X) = copyto!(Y, X)

function inner(::MultinomialSymmetricDoubleStochasticMatrices, p, X, Y)
    d = zero(Base.promote_eltype(p, X, Y))
    @inbounds for i in eachindex(p, X, Y)
        d += X[i] * Y[i] / p[i]
    end
    return d
end


@generated manifold_dimension(::MultinomialSymmetricDoubleStochasticMatrices{n}) where {n} = (n - 1)^2

@doc raw"""
    project(;;MultinomialSymmetricDoubleStochasticMatrices{n}, X, p, Y) where {n}

Project `Y` onto the tangent space at `p` on the [`MultinomialSymmetricDoubleStochasticMatrices`](@ref) `M`, return the result in `X`.
The formula reads
````math
    \operatorname{proj}_p)(Y) = Y - (α\mathbf{1}_n^{\mathrm{T}} + \mathbf{1}_nβ^{\mathrm{T}}) ⊙ p,
````
where $⊙$ denotes the Hadamard or elementwise product and $\mathbb{1}_n$ is the vector of length $n$ containing ones.
The two vectors $α, β$ are computed as
````math
    α = (1-pp^{\mathrm{T}})^\dagger(Y-pY^{\mathrm{T}})\mathbf{1}_n
    \text{ and }
    β = Y^{\mathrm{T}}\mathbf{1}_n - p^{\mathrm{T}}α,
````
where $\circ^{\dagger}$ denotes the left pseude inverse.
"""
function project(::MultinomialSymmetricDoubleStochasticMatrices{n}, X, p, Y) where {n}
    α = (I-p*p') \ sum( Y-p*Y', dims=2)
    β = sum(Y',dims=2) - p'*α
    X .= Y .-(repeat(α,1,3) .+ repeat(β',3,1) ) .* p
end

@generated representation_size(::MultinomialSymmetricDoubleStochasticMatrices{n}) where {n} = (n, n)

function Base.show(io::IO, ::MultinomialSymmetricDoubleStochasticMatrices{n}) where {n}
    return print(io, "MultinomialSymmetricDoubleStochasticMatrices($(n))")
end
