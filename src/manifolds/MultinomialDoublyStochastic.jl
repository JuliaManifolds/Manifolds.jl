@doc raw"""
    AbstractMultinomialDoublyStochastic <: AbstractDecoratorManifold{ℝ}

A common type for manifolds that are doubly stochastic, for example by direct constraint
[`MultinomialDoubleStochastic`](@ref) or by symmetry [`MultinomialSymmetric`](@ref),
or additionally by symmetric positive definiteness [`MultinomialSymmetricPositiveDefinite`](@ref)
as long as they are also modeled as [`IsIsometricEmbeddedManifold`](@extref `ManifoldsBase.IsIsometricEmbeddedManifold`).

That way they share the inner product (just by restriction), and even the Riemannian gradient
"""
abstract type AbstractMultinomialDoublyStochastic <: AbstractDecoratorManifold{ℝ} end

@doc raw"""
    representation_size(M::AbstractMultinomialDoublyStochastic)

return the representation size of doubly stochastic matrices, which are embedded
in the ``ℝ^{n×n}`` matrices and hence the answer here is ``
"""
function representation_size(M::AbstractMultinomialDoublyStochastic)
    n = get_parameter(M.size)[1]
    return (n, n)
end

@doc raw"""
    riemannian_gradient(M::AbstractMultinomialDoublyStochastic, p, Y; kwargs...)

Let ``Y`` denote the Euclidean gradient of a function ``\tilde f`` defined in the
embedding neighborhood of `M`, then the Riemannian gradient is given by
Lemma 1 [DouikHassibi:2019](@cite) as

```math
  \operatorname{grad} f(p) = \proj_{T_p\mathcal M}(Y⊙p)
```

where ``⊙`` denotes the Hadamard or elementwise product, and the projection
is the projection onto the tangent space of the corresponding manifold.

"""
riemannian_gradient(M::AbstractMultinomialDoublyStochastic, p, Y; kwargs...)

function riemannian_gradient!(M::AbstractMultinomialDoublyStochastic, X, p, Y; kwargs...)
    X .= p .* Y
    project!(M, X, p, X)
    return X
end

@doc raw"""
    MultinomialDoublyStochastic{T} <: AbstractMultinomialDoublyStochastic

The set of doubly stochastic multinomial matrices consists of all ``n×n`` matrices with
stochastic columns and rows, i.e.

````math
\begin{aligned}
\mathcal{DP}(n) \coloneqq \bigl\{p ∈ ℝ^{n×n}\ \big|\ &p_{i,j} > 0 \text{ for all } i=1,…,n, j=1,…,m,\\
& p\mathbf{1}_n = p^{\mathrm{T}}\mathbf{1}_n = \mathbf{1}_n
\bigr\},
\end{aligned}
````

where ``\mathbf{1}_n`` is the vector of length ``n`` containing ones.

The tangent space can be written as

````math
T_p\mathcal{DP}(n) \coloneqq \bigl\{
X ∈ ℝ^{n×n}\ \big|\ X = X^{\mathrm{T}} \text{ and }
X\mathbf{1}_n = X^{\mathrm{T}}\mathbf{1}_n = \mathbf{0}_n
\bigr\},
````

where ``\mathbf{0}_n`` is the vector of length ``n`` containing zeros.

More details can be found in Section III [DouikHassibi:2019](@cite).

# Constructor

    MultinomialDoubleStochastic(n::Int; parameter::Symbol=:type)

Generate the manifold of matrices ``ℝ^{n×n}`` that are doubly stochastic and symmetric.
"""
struct MultinomialDoubleStochastic{T} <: AbstractMultinomialDoublyStochastic
    size::T
end

function MultinomialDoubleStochastic(n::Int; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n,))
    return MultinomialDoubleStochastic{typeof(size)}(size)
end

@doc raw"""
    check_point(M::MultinomialDoubleStochastic, p)

Checks whether `p` is a valid point on the [`MultinomialDoubleStochastic`](@ref)`(n)` `M`,
i.e. is a  matrix with positive entries whose rows and columns sum to one.
"""
function check_point(M::MultinomialDoubleStochastic, p; kwargs...)
    n = get_parameter(M.size)[1]
    s = check_point(MultinomialMatrices(n, n), p; kwargs...)
    !isnothing(s) && return s
    s2 = check_point(MultinomialMatrices(n, n), p'; kwargs...)
    return s2
end

@doc raw"""
    check_vector(M::MultinomialDoubleStochastic p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialDoubleStochastic`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and sums to zero along any
column or row.
"""
function check_vector(M::MultinomialDoubleStochastic, p, X; kwargs...)
    n = get_parameter(M.size)[1]
    s = check_vector(MultinomialMatrices(n, n), p, X; kwargs...)
    !isnothing(s) && return s
    s2 = check_vector(MultinomialMatrices(n, n), p, X'; kwargs...)
    return s2
end

function get_embedding(::MultinomialDoubleStochastic{TypeParameter{Tuple{n}}}) where {n}
    return MultinomialMatrices(n, n)
end
function get_embedding(M::MultinomialDoubleStochastic{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return MultinomialMatrices(n, n; parameter=:field)
end

function ManifoldsBase.get_embedding_type(::MultinomialDoubleStochastic)
    return ManifoldsBase.IsometricallyEmbeddedManifoldType()
end

"""
    is_flat(::MultinomialDoubleStochastic)

Return false. [`MultinomialDoubleStochastic`](@ref) is not a flat manifold.
"""
is_flat(M::MultinomialDoubleStochastic) = false

@doc raw"""
    manifold_dimension(M::MultinomialDoubleStochastic)

returns the dimension of the [`MultinomialDoubleStochastic`](@ref) manifold
namely
````math
\operatorname{dim}_{\mathcal{DP}(n)} = (n-1)^2.
````
"""
function manifold_dimension(M::MultinomialDoubleStochastic)
    n = get_parameter(M.size)[1]
    return (n - 1)^2
end

@doc raw"""
    project(M::MultinomialDoubleStochastic, p, Y)

Project `Y` onto the tangent space at `p` on the [`MultinomialDoubleStochastic`](@ref) `M`, return the result in `X`.
The formula reads

````math
    \operatorname{proj}_p(Y) = Y - (α\mathbf{1}_n^{\mathrm{T}} + \mathbf{1}_nβ^{\mathrm{T}}) ⊙ p,
````

where ``⊙`` denotes the Hadamard or elementwise product and ``\mathbb{1}_n`` is the vector of length ``n`` containing ones.
The two vectors ``α,β ∈ ℝ^{n×n}`` are computed as a solution (typically using the left pseudo inverse) of

````math
    \begin{pmatrix} I_n & p\\p^{\mathrm{T}} & I_n \end{pmatrix}
    \begin{pmatrix} α\\ β\end{pmatrix}
    =
    \begin{pmatrix} Y\mathbf{1}\\Y^{\mathrm{T}}\mathbf{1}\end{pmatrix},
````
where ``I_n`` is the ``n×n`` unit matrix and ``\mathbf{1}_n`` is the vector of length ``n`` containing ones.

"""
project(::MultinomialDoubleStochastic, ::Any, ::Any)

function project!(M::MultinomialDoubleStochastic, X, p, Y)
    n = get_parameter(M.size)[1]
    ζ = [I p; p' I] \ [sum(Y, dims=2); sum(Y, dims=1)'] # Formula (25) from 1802.02628
    return X .= Y .- (repeat(ζ[1:n], 1, n) .+ repeat(ζ[(n + 1):end]', n, 1)) .* p
end

@doc raw"""
    project(
        M::AbstractMultinomialDoublyStochastic,
        p;
        maxiter = 100,
        tolerance = eps(eltype(p))
    )

project a matrix `p` with positive entries applying Sinkhorn's algorithm.
Note that this project method – different from the usual case, accepts keywords.
"""
function project(M::AbstractMultinomialDoublyStochastic, p; kwargs...)
    q = allocate_result(M, project, p)
    project!(M, q, p; kwargs...)
    return q
end

function project!(
    ::AbstractMultinomialDoublyStochastic,
    q,
    p;
    maxiter::Int=100,
    tolerance::Real=eps(eltype(p)),
)
    any(p .<= 0) && throw(
        DomainError(
            "The matrix $p can not be projected, since it has nonpositive entries.",
        ),
    )
    iter = 0
    d1 = sum(p, dims=1)
    d2 = 1 ./ (p * d1')
    row = d2' * p
    gap = 2 * tolerance
    while iter < maxiter && (gap >= tolerance)
        iter += 1
        row .= d2' * p
        gap = maximum(abs.(row .* d1 .- 1))
        d1 .= 1 ./ row
        d2 .= 1 ./ (p * d1')
    end
    q .= p .* (d2 * d1)
    return q
end

@doc raw"""
    rand(::MultinomialDoubleStochastic; vector_at=nothing, σ::Real=1.0, kwargs...)

Generate random points on the [`MultinomialDoubleStochastic`](@ref) manifold
or tangent vectors at the point `vector_at` if that is not `nothing`.

Let ``n×n`` denote the matrix dimension of the [`MultinomialDoubleStochastic`](@ref).

When `vector_at` is nothing, this is done by generating a random matrix`rand(n,n)`
with positive entries and projecting it onto the manifold. The `kwargs...` are
passed to this projection.

When `vector_at` is not `nothing`, a random matrix in the ambient space is generated
and projected onto the tangent space
"""
rand(::MultinomialDoubleStochastic; σ::Real=1.0)

function Random.rand!(
    rng::AbstractRNG,
    M::MultinomialDoubleStochastic,
    pX;
    vector_at=nothing,
    σ::Real=one(real(eltype(pX))),
    kwargs...,
)
    rand!(rng, pX)
    pX .*= σ
    if vector_at === nothing
        project!(M, pX, pX; kwargs...)
    else
        project!(M, pX, vector_at, pX)
    end
    return pX
end

@doc raw"""
    retract(M::MultinomialDoubleStochastic, p, X, ::ProjectionRetraction)

compute a projection based retraction by projecting ``p\odot\exp(X⨸p)`` back onto the manifold,
where ``⊙,⨸`` are elementwise multiplication and division, respectively. Similarly, ``\exp``
refers to the elementwise exponentiation.
"""
retract(::MultinomialDoubleStochastic, ::Any, ::Any, ::ProjectionRetraction)

function ManifoldsBase.retract_project!(M::MultinomialDoubleStochastic, q, p, X)
    return project!(M, q, p .* exp.(X ./ p))
end
function ManifoldsBase.retract_project_fused!(
    M::MultinomialDoubleStochastic,
    q,
    p,
    X,
    t::Number,
)
    return project!(M, q, p .* exp.(t .* X ./ p))
end

function Base.show(io::IO, ::MultinomialDoubleStochastic{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "MultinomialDoubleStochastic($(n))")
end
function Base.show(io::IO, M::MultinomialDoubleStochastic{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "MultinomialDoubleStochastic($(n); parameter=:field)")
end
