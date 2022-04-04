@doc raw"""
    AbstractMultinomialDoublyStochastic{N} <: AbstractDecoratorManifold{ℝ}

A common type for manifolds that are doubly stochastic, for example by direct constraint
[`MultinomialDoubleStochastic`](@ref) or by symmetry [`MultinomialSymmetric`](@ref),
as long as they are also modeled as [`IsIsometricEmbeddedManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/decorator.html#ManifoldsBase.IsIsometricEmbeddedManifold).
"""
abstract type AbstractMultinomialDoublyStochastic{N} <: AbstractDecoratorManifold{ℝ} end

function active_traits(f, ::AbstractMultinomialDoublyStochastic, args...)
    return merge_traits(IsIsometricEmbeddedManifold())
end

@doc raw"""
    MultinomialDoublyStochastic{n} <: AbstractMultinomialDoublyStochastic{N}

The set of doubly stochastic multinomial matrices consists of all $n×n$ matrices with
stochastic columns and rows, i.e.
````math
\begin{aligned}
\mathcal{DP}(n) \coloneqq \bigl\{p ∈ ℝ^{n×n}\ \big|\ &p_{i,j} > 0 \text{ for all } i=1,…,n, j=1,…,m,\\
& p\mathbf{1}_n = p^{\mathrm{T}}\mathbf{1}_n = \mathbf{1}_n
\bigr\},
\end{aligned}
````

where $\mathbf{1}_n$ is the vector of length $n$ containing ones.

The tangent space can be written as

````math
T_p\mathcal{DP}(n) \coloneqq \bigl\{
X ∈ ℝ^{n×n}\ \big|\ X = X^{\mathrm{T}} \text{ and }
X\mathbf{1}_n = X^{\mathrm{T}}\mathbf{1}_n = \mathbf{0}_n
\bigr\},
````

where $\mathbf{0}_n$ is the vector of length $n$ containing zeros.

More details can be found in Section III[^DouikHassibi2019].

# Constructor

    MultinomialDoubleStochastic(n)

Generate the manifold of matrices $\mathbb R^{n×n}$ that are doubly stochastic and symmetric.

[^DouikHassibi2019]:
    > A. Douik, B. Hassibi:
    > AbstractManifold Optimization Over the Set of Doubly Stochastic Matrices: A Second-Order Geometry,
    > IEEE Transactions on Signal Processing 67(22), pp. 5761–5774, 2019.
    > doi: [10.1109/tsp.2019.2946024](http://doi.org/10.1109/tsp.2019.2946024),
    > arXiv: [1802.02628](https://arxiv.org/abs/1802.02628).
"""
struct MultinomialDoubleStochastic{N} <: AbstractMultinomialDoublyStochastic{N} end

function MultinomialDoubleStochastic(n::Int)
    return MultinomialDoubleStochastic{n}()
end

@doc raw"""
    check_point(M::MultinomialDoubleStochastic, p)

Checks whether `p` is a valid point on the [`MultinomialDoubleStochastic`](@ref)`(n)` `M`,
i.e. is a  matrix with positive entries whose rows and columns sum to one.
"""
function check_point(M::MultinomialDoubleStochastic{n}, p; kwargs...) where {n}
    r = sum(p, dims=2)
    if !isapprox(norm(r - ones(n, 1)), 0.0; kwargs...)
        return DomainError(
            r,
            "The point $(p) does not lie on $M, since its rows do not sum up to one.",
        )
    end
    return nothing
end
@doc raw"""
    check_vector(M::MultinomialDoubleStochastic p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialDoubleStochastic`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and sums to zero along any
column or row.
"""
function check_vector(M::MultinomialDoubleStochastic{n}, p, X; kwargs...) where {n}
    r = sum(X, dims=2) # check for stochastic rows
    if !isapprox(norm(r), 0.0; kwargs...)
        return DomainError(
            r,
            "The matrix $(X) is not a tangent vector to $(p) on $(M), since its rows do not sum up to zero.",
        )
    end
    return nothing
end

function get_embedding(::MultinomialDoubleStochastic{N}) where {N}
    return MultinomialMatrices(N, N)
end

@doc raw"""
    manifold_dimension(M::MultinomialDoubleStochastic{n}) where {n}

returns the dimension of the [`MultinomialDoubleStochastic`](@ref) manifold
namely
````math
\operatorname{dim}_{\mathcal{DP}(n)} = (n-1)^2.
````
"""
@generated function manifold_dimension(::MultinomialDoubleStochastic{n}) where {n}
    return (n - 1)^2
end

@doc raw"""
    project(M::MultinomialDoubleStochastic{n}, p, Y) where {n}

Project `Y` onto the tangent space at `p` on the [`MultinomialDoubleStochastic`](@ref) `M`, return the result in `X`.
The formula reads
````math
    \operatorname{proj}_p(Y) = Y - (α\mathbf{1}_n^{\mathrm{T}} + \mathbf{1}_nβ^{\mathrm{T}}) ⊙ p,
````
where $⊙$ denotes the Hadamard or elementwise product and $\mathbb{1}_n$ is the vector of length $n$ containing ones.
The two vectors $α,β ∈ ℝ^{n×n}$ are computed as a solution (typically using the left pseudo inverse) of
````math
    \begin{pmatrix} I_n & p\\p^{\mathrm{T}} & I_n \end{pmatrix}
    \begin{pmatrix} α\\ β\end{pmatrix}
    =
    \begin{pmatrix} Y\mathbf{1}\\Y^{\mathrm{T}}\mathbf{1}\end{pmatrix},
````
where $I_n$ is the $n×n$ unit matrix and $\mathbf{1}_n$ is the vector of length $n$ containing ones.

"""
project(::MultinomialDoubleStochastic, ::Any, ::Any)

function project!(::MultinomialDoubleStochastic{n}, X, p, Y) where {n}
    ζ = [I p; p I] \ [sum(Y, dims=2); sum(Y, dims=1)'] # Formula (25) from 1802.02628
    return X .= Y .- (repeat(ζ[1:n], 1, 3) .+ repeat(ζ[(n + 1):end]', 3, 1)) .* p
end

@doc raw"""
    project(
        M::AbstractMultinomialDoublyStochastic,
        p;
        maxiter = 100,
        tolerance = eps(eltype(p))
    )

project a matrix `p` with positive entries applying Sinkhorn's algorithm.
Note that this projct method – different from the usual case, accepts keywords.
"""
function project(M::AbstractMultinomialDoublyStochastic, p; kwargs...)
    q = allocate_result(M, project, p)
    project!(M, q, p; kwargs...)
    return q
end

function project!(
    ::AbstractMultinomialDoublyStochastic{n},
    q,
    p;
    maxiter=100,
    tolerance=eps(eltype(p)),
) where {n}
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

@generated function representation_size(::MultinomialDoubleStochastic{n}) where {n}
    return (n, n)
end

@doc raw"""
    retract(M::MultinomialDoubleStochastic, p, X, ::ProjectionRetraction)

compute a projection based retraction by projecting $p\odot\exp(X⨸p)$ back onto the manifold,
where $⊙,⨸$ are elementwise multiplication and division, respectively. Similarly, $\exp$
refers to the elementwise exponentiation.
"""
retract(::MultinomialDoubleStochastic, ::Any, ::Any, ::ProjectionRetraction)

function retract_project!(M::MultinomialDoubleStochastic, q, p, X)
    return project!(M, q, p .* exp.(X ./ p))
end

"""
    vector_transport_to(M::MultinomialDoubleStochastic, p, X, q)

transport the tangent vector `X` at `p` to `q` by projecting it onto the tangent space
at `q`.
"""
vector_transport_to(
    ::MultinomialDoubleStochastic,
    ::Any,
    ::Any,
    ::Any,
    ::ProjectionTransport,
)

function vector_transport_to!(
    M::MultinomialDoubleStochastic,
    Y,
    p,
    X,
    q,
    ::ProjectionTransport,
)
    project!(M, Y, q, X)
    return Y
end

function Base.show(io::IO, ::MultinomialDoubleStochastic{n}) where {n}
    return print(io, "MultinomialDoubleStochastic($(n))")
end
