@doc raw"""
    AbstractMultinomialDoublyStochastic{N} <: AbstractEmbeddedManifold{ℝ, DefaultIsometricEmbeddingType}

A comon type for manifolds that are doubly stochastic, for example by direct constraint
[`MultinomialDoublyStochastic`](@ref) or by symmetry [`MultionialSymmetric`](@ref),
as long as they are also modeled as [`DefaultIsometricEmbeddingType`](@ref)
[`AbstractEmbeddedManifold`](@ref)s.
"""
abstract type AbstractMultinomialDoublyStochastic{N} <:
              AbstractEmbeddedManifold{ℝ,DefaultIsometricEmbeddingType} end

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

    MultinomialDoubleStochasticMatrices(n)

Generate the manifold of matrices $\mathbb R^{n×n}$ that are doubly stochastic and symmetric.

[^DouikHassibi2019]:
    > A. Douik, B. Hassibi:
    > Manifold Optimization Over the Set of Doubly Stochastic Matrices: A Second-Order Geometry,
    > IEEE Transactions on Signal Processing 67(22), pp. 5761–5774, 2019.
    > doi: [10.1109/tsp.2019.2946024](http://doi.org/10.1109/tsp.2019.2946024),
    > arXiv: [1802.02628](https://arxiv.org/abs/1802.02628).
"""
struct MultinomialDoubleStochasticMatrices{N} <: AbstractMultinomialDoublyStochastic{N} end

function MultinomialDoubleStochasticMatrices(n::Int)
    return MultinomialDoubleStochasticMatrices{n}()
end

@doc raw"""
    check_manifold_point(M::MultinomialDoubleStochasticMatrices, p)

Checks whether `p` is a valid point on the [`MultinomialDoubleStochasticMatrices`](@ref)`(n)` `M`,
i.e. is a  matrix with positive entries whose rows and columns sum to one.
"""
function check_manifold_point(
    M::MultinomialDoubleStochasticMatrices{n},
    p;
    kwargs...,
) where {n}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    # positivity and columns are checked in the embedding, we further check
    r = sum(p, dims = 2)
    if !isapprox(norm(r - ones(n, 1)), 0.0; kwargs...)
        return DomainError(
            r,
            "The point $(p) does not lie on $M, since its rows do not sum up to one.",
        )
    end
    return nothing
end
@doc raw"""
    check_tangent_vector(M::MultinomialDoubleStochasticMatrices p, X; check_base_point = true, kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialDoubleStochasticMatrices`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and sums to zero along any
column or row.

The optional parameter `check_base_point` indicates, whether to call
[`check_manifold_point`](@ref check_manifold_point(::MultinomialDoubleStochasticMatrices, ::Any))  for `p`.
"""
function check_tangent_vector(
    M::MultinomialDoubleStochasticMatrices{n},
    p,
    X;
    check_base_point = true,
    kwargs...,
) where {n}
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)),typeof(p),typeof(X)},
        M,
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...,
    )
    mpv === nothing || return mpv
    # columns are checked in the embedding, we further check
    r = sum(X, dims = 2) # check for stochastic rows
    if !isapprox(norm(r), 0.0; kwargs...)
        return DomainError(
            r,
            "The matrix $(X) is not a tangent vector to $(p) on $(M), since its rows do not sum up to zero.",
        )
    end
    return nothing
end

function decorated_manifold(::MultinomialDoubleStochasticMatrices{N}) where {N}
    return MultinomialMatrices(N, N)
end

embed!(::MultinomialDoubleStochasticMatrices, q, p) = copyto!(q, p)
embed!(::MultinomialDoubleStochasticMatrices, Y, p, X) = copyto!(Y, X)

@doc raw"""
    manifold_dimension(M::MultinomialDoubleStochasticMatrices{n}) where {n}

returns the dimension of the [`MultinomialDoubleStochasticMatrices`](@ref) manifold
namely
````math
\operatorname{dim}_{\mathcal{DP}(n)} = (n-1)^2.
````
"""
@generated function manifold_dimension(::MultinomialDoubleStochasticMatrices{n}) where {n}
    return (n - 1)^2
end

@doc raw"""
    project(M::MultinomialDoubleStochasticMatrices{n}, p, Y) where {n}

Project `Y` onto the tangent space at `p` on the [`MultinomialDoubleStochasticMatrices`](@ref) `M`, return the result in `X`.
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
where $I_n$ is teh $n×n$ unit matrix and $\mathbf{1}_n$ is the vector of length $n$ containing ones.

"""
project(::MultinomialDoubleStochasticMatrices, ::Any, ::Any)

function project!(::MultinomialDoubleStochasticMatrices{n}, X, p, Y) where {n}
    ζ = [I p; p I] \ [sum(Y, dims = 2); sum(Y, dims = 1)'] # Formula (25) from 1802.02628
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
    maxiter = 100,
    tolerance = eps(eltype(p)),
) where {n}
    any(p .<= 0) &&
        throw(DomainError("The matrix $p can not be projected, since it has nonpositive entries."))
    iter = 0
    d1 = sum(p, dims = 1)
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

@generated function representation_size(::MultinomialDoubleStochasticMatrices{n}) where {n}
    return (n, n)
end

@doc raw"""
    retract(M::MultinomialDoubleStochasticMatrices, p, X, ::ProjectionRetraction)

compute a projection based retraction by projecting $p\odot\exp(X⨸p)$ back onto the manifold,
where $⊙,⨸$ are elementwise multiplication and division, respectively. Similarly, $\exp$
refers to the elementwise exponentiation.
"""
retract(::MultinomialDoubleStochasticMatrices, ::Any, ::Any, ::ProjectionRetraction)

function retract!(M::MultinomialDoubleStochasticMatrices, q, p, X, ::ProjectionRetraction)
    return project!(M, q, p .* exp.(X ./ p))
end

"""
    vector_transport_to(M::MultinomialDoubleStochasticMatrices, p, X, q)

transport the tangent vector `X` at `p` to `q` by projecting it onto the tangent space
at `q`.
"""
vector_transport_to(
    ::MultinomialDoubleStochasticMatrices,
    ::Any,
    ::Any,
    ::Any,
    ::ProjectionTransport,
)

function vector_transport_to!(
    M::MultinomialDoubleStochasticMatrices,
    Y,
    p,
    X,
    q,
    ::ProjectionTransport,
)
    project!(M, Y, q, X)
    return Y
end

function Base.show(io::IO, ::MultinomialDoubleStochasticMatrices{n}) where {n}
    return print(io, "MultinomialDoubleStochasticMatrices($(n))")
end
