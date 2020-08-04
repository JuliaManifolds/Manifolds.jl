@doc raw"""
    MultinomialDoublyStochastic{n} <: AbstractEmbeddedManifold{ℝ, DefaultEmbeddingType}

The multinomial manifold consists of `m` column vectors, where each column is of length
`n` and unit norm, i.e.

````math
\begin{aligned}
\mathcal{PS}(n) \coloneqq \bigl\{p ∈ ℝ^{n×n}\ \big|\ &p_{i,j} > 0 \text{ for all } i=1,…,n, j=1,…,m,\\
& p^\mathrm{T} = p,\\
& p\text{ is symmetric positive definite} \text{ and }\\
& p\mathbf{1}_n = p^{\mathrm{T}}\mathbf{1}_n = \mathbf{1}_n
\bigr\},
\end{aligned}
````

where $\mathbf{1}_n$ is the vector of length $n$ containing ones.

The tangent space can be written as

````math
T_p\mathcal{PS}(n) \coloneqq \bigl\{
X ∈ ℝ^{n×n}\ \big|\ X = X^{\mathrm{T}} \text{ and }
X\mathbf{1}_n = X^{\mathrm{T}}\mathbf{1}_n = \mathbf{0}_n
\bigr\},
````

where $\mathbf{0}_n$ is the vector of length $n$ containing zeros.

More details can be found in[^DouikHassibi2019].

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
struct MultinomialDoubleStochasticMatrices{N} <:
       AbstractEmbeddedManifold{ℝ,DefaultEmbeddingType} where {N} end

function MultinomialDoubleStochasticMatrices(n::Int)
    return MultinomialDoubleStochasticMatrices{n}()
end

@doc raw"""
    check_manifold_point(M::MultinomialDoubleStochasticMatrices, p)

Checks whether `p` is a valid point on the [`MultinomialDoubleStochasticMatrices`](@ref)`(m,n)` `M`,
i.e. is a symmetric matrix whose rows and columns sum to one.
"""
check_manifold_point(::MultinomialDoubleStochasticMatrices, ::Any)
function check_manifold_point(
    M::MultinomialDoubleStochasticMatrices{n},
    p;
    kwargs...,
) where {n}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    c = sum(p, dims = 1) # due to symmetry we only have to check the cols
    if !isapprox(norm(c - ones(1, n)), 0.0; kwargs...)
        return DomainError(
            c,
            "The point $(p) does not lie on $M, since its columns/rows do not sum up to one.",
        )
    end
    if !(minimum(p) > 0)
        return DomainError(
            minimum(p),
            "The point $(p) does not lie on $M, since at least one of its entries is nonpositive.",
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
    c = sum(X, dims = 1) # due to symmetry, we only have to check columns
    if !isapprox(norm(c), 0.0; kwargs...)
        return DomainError(
            c,
            "The matrix $(X) is not a tangent vector to $(p) on $(M), since its columns/rows do not sum up to zero.",
        )
    end
    return nothing
end

function decorated_manifold(::MultinomialDoubleStochasticMatrices{N}) where {N}
    return SymmetricMatrices(N, ℝ)
end

embed!(::MultinomialDoubleStochasticMatrices, q, p) = copyto!(q, p)
embed!(::MultinomialDoubleStochasticMatrices, Y, p, X) = copyto!(Y, X)

@doc raw"""
    inner(M::MultinomialDoubleStochasticMatrices{n}, p, X, Y) where {n}

Compute the inner product on the tangent space at $p$, which is the elementwise
inner product similar to the [`Hyperbolic`](@ref) space, i.e.

````math
    \langle X, Y \rangle_p = \sum_{i,j=1}^n \frac{X_{ij}Y_{ij}}{p_{ij}}.
````
"""
function inner(::MultinomialDoubleStochasticMatrices, p, X, Y)
    # to avoid memory allocations, we sum in a single number
    d = zero(Base.promote_eltype(p, X, Y))
    @inbounds for i in eachindex(p, X, Y)
        d += X[i] * Y[i] / p[i]
    end
    return d
end

@doc raw"""
    manifold_dimension(M::MultinomialDoubleStochasticMatrices)

returns the dimension of the [`MultinomialDoubleStochasticMatrices`](@ref) manifold
namely
````math
\operatorname{dim}_{\mathcal{PS}(n)} = (n-1)^2.
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
    \begin{pmatrix} I & p\\p^{\mathrm{T}} & I \end{pmatrix}
    \begin{pmatrix} α\\ β\end{pmatrix}
    =
    \begin{pmatrix} Y\mathbf{1}\\Y^{\mathrm{T}}\mathbf{1}\end{pmatrix},
````
where $\mathbf{1}_n$ is the vector of length $n$ containing ones.

"""
project(::MultinomialDoubleStochasticMatrices, ::Any, ::Any)

function project!(::MultinomialDoubleStochasticMatrices{n}, X, p, Y) where {n}
    ζ = [I p; p I] \ [sum(Y, dims = 2); sum(Y, dims = 1)'] # Formula (25) from 1802.02628
    return X .= Y .- (repeat(ζ[1:n], 1, 3) .+ repeat(ζ[(n + 1):end]', 3, 1)) .* p
end

@doc raw"""
    project(
        M::MultinomialDoubleStochasticMatrices,
        p;
        maxiter = 100,
        tolerance = eps(eltype(p))
    )

project a matrix `p` with positive entries applying Sinkhorn's algorithm.
"""
function project(M::MultinomialDoubleStochasticMatrices, p; kwargs...)
    q = allocate_result(M, project, p)
    project!(M, q, p; kwargs...)
    return q
end

function project!(
    ::MultinomialDoubleStochasticMatrices{n},
    q,
    p;
    maxiter = 100,
    tolerance = eps(eltype(p)),
) where {n}
    any(p .<= 0) && throw(DomainError(
        "The matrix $p can not be projected, since it has nonpositive entries."
    ))
    iter = 0
    d1 = sum(p, dims = 1)
    d2 = 1 ./ (p * d1')
    row = d2' * p
    gap = 2*tolerance
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
    retract(M::Spectrahedron, p, X, ::ProjectionRetraction)

compute a projection based retraction by projecting $p+X$ back onto the manifold.
"""
retract(::MultinomialDoubleStochasticMatrices, ::Any, ::Any, ::ProjectionRetraction)

function retract!(M::MultinomialDoubleStochasticMatrices, q, p, X, ::ProjectionRetraction)
    return project!(M, q, p + X)
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
