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

    MultinomialSymmetricDoubleStochasticMatrices(n)

Generate the manifold of matrices $\mathbb R^{n×n}$ that are doubly stochastic and symmetric.

[^DouikHassibi2019]:
    > A. Douik, B. Hassibi:
    > Manifold Optimization Over the Set of Doubly Stochastic Matrices: A Second-Order Geometry,
    > IEEE Transactions on Signal Processing 67(22), pp. 5761–5774, 2019.
    > doi: [10.1109/tsp.2019.2946024](http://doi.org/10.1109/tsp.2019.2946024),
    > arXiv: [1802.02628](https://arxiv.org/abs/1802.02628).
"""
struct MultinomialSymmetricDoubleStochasticMatrices{N} <:
       AbstractEmbeddedManifold{ℝ,DefaultEmbeddingType} where {N} end

function MultinomialSymmetricDoubleStochasticMatrices(n::Int)
    return MultinomialSymmetricDoubleStochasticMatrices{n}()
end

@doc raw"""
    check_manifold_point(M::MultinomialSymmetricDoubleStochasticMatrices, p)

Checks whether `p` is a valid point on the [`MultinomialSymmetricDoubleStochasticMatrices`](@ref)`(m,n)` `M`,
i.e. is a symmetric matrix whose rows and columns sum to one.
"""
check_manifold_point(::MultinomialSymmetricDoubleStochasticMatrices, ::Any)
function check_manifold_point(
    M::MultinomialSymmetricDoubleStochasticMatrices{n},
    p;
    kwargs...,
) where {n}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    c = sum(p, dims = 1)
    if !isapprox(norm(c - ones(1, n)), 0.0; kwargs...)
        return DomainError(
            c,
            "The point $(p) does not lie on $M, since its columns do not sum up to one.",
        )
    end
    r = sum(p, dims = 2)
    if !isapprox(norm(r - ones(n, 1)), 0.0; kwargs...)
        return DomainError(
            r,
            "The point $(p) does not lie on $M, since its columns do not sum up to one.",
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
    check_tangent_vector(M::MultinomialSymmetricDoubleStochasticMatrices p, X; check_base_point = true, kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialSymmetricDoubleStochasticMatrices`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and sums to zero along any
column or row.

The optional parameter `check_base_point` indicates, whether to call
[`check_manifold_point`](@ref check_manifold_point(::MultinomialSymmetricDoubleStochasticMatrices, ::Any))  for `p`.
"""
function check_tangent_vector(
    M::MultinomialSymmetricDoubleStochasticMatrices{n},
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
    c = sum(X, dims = 1)
    if !isapprox(norm(c), 0.0; kwargs...)
        return DomainError(
            c,
            "The matrix $(X) is not a tangent vector to $(p) on $(M), since its columns do not sum up to zero.",
        )
    end
    r = sum(p, dims = 2)
    if !isapprox(norm(r), 0.0; kwargs...)
        return DomainError(
            r,
            "The point $(p) does not lie on $M, since its columns do not sum up to one.",
        )
    end
    return nothing
end

function decorated_manifold(::MultinomialSymmetricDoubleStochasticMatrices{N}) where {N}
    return SymmetricMatrices(N,ℝ)
end

embed!(::MultinomialSymmetricDoubleStochasticMatrices, q, p) = copyto!(q, p)
embed!(::MultinomialSymmetricDoubleStochasticMatrices, Y, p, X) = copyto!(Y, X)

@doc raw"""
    inner(M::MultinomialSymmetricDoubleStochasticMatrices{n}, p, X, Y) where {n}

Compute the inner product on the tangent space at $p$, which is the elementwise
inner product similar to the [`Hyperbolic`](@ref) space, i.e.

````math
    \langle X, Y \rangle_p = \sum_{i,j=1}^n \frac{X_{ij}Y_{ij}}{p_{ij}}.
````
"""
function inner(::MultinomialSymmetricDoubleStochasticMatrices, p, X, Y)
    # to avoid memory allocations, we sum in a single number
    d = zero(Base.promote_eltype(p, X, Y))
    @inbounds for i in eachindex(p, X, Y)
        d += X[i] * Y[i] / p[i]
    end
    return d
end


@generated function manifold_dimension(
    ::MultinomialSymmetricDoubleStochasticMatrices{n},
) where {n}
    return (n - 1)^2
end

@doc raw"""
    project(M::MultinomialSymmetricDoubleStochasticMatrices{n}, X, p, Y) where {n}

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
    ζ = [I p; p I] \ [sum(B, dims = 2); sum(B, dims = 1)'] # Formula (25) from 1802.02628
    return X .= Y .- (repeat(ζ[1:n], 1, 3) .+ repeat(ζ[n:end]', 3, 1)) .* p
end

@doc raw"""
    project(
        M::MultinomialSymmetricDoubleStochasticMatrices,
        p;
        maxiter=100,
        tol=eps(eltype(p))
    )

    project a matrix `p` with positive entries applying sinkhorns algorithm,
    transcribed from the matlab code ot [manopt](http://manopt.org).
"""
function project!(
    ::MultinomialSymmetricDoubleStochasticMatrices{n},
    q,
    p;
    maxiter = 100,
    tolerance = eps(eltype(p)),
) where {n}
    iter = 0
    d1 = sum(p, dims = 1)
    any(p * d1' .== 0) && throw(DomainError(
        p * d1',
        "The matrix p ($p)  can not be projected, since p*d1' ($(p*d1')).",
    ))
    d2 = 1 ./ (p * d1')
    row = d2' * p
    gap = maximum(abs.(row .* d1 .- 1))
    while iter < maxiter && (gap >= tolerance)
        iter += 1
        row .= d2' * p
        any(row .== 0) && throw(DomainError(
            row,
            "projection sinkhorn failed with row $row containing a zero.",
        ))
        d1 .= 1 ./ row
        any(d1 .== 0) && throw(DomainError(
            d1,
            "projection sinkhorn failed with d1 $d1 containing a zero.",
        ))
        d2 .= 1 ./ (p * d1')
        any(d2 .== 0) && throw(DomainError(
            d2,
            "projection sinkhorn failed with d2 $d2 containing a zero.",
        ))
        gap = maximum(abs.(row .* d1 .- 1))
    end
    q .= p .* (d2 * d1)
    return q
end

@generated function representation_size(
    ::MultinomialSymmetricDoubleStochasticMatrices{n},
) where {n}
    return (n, n)
end

@doc raw"""
    retract(M::Spectrahedron, q, Y, ::ProjectionRetraction)

compute a projection based retraction by projecting $q+Y$ back onto the manifold.
"""
retract(
    ::MultinomialSymmetricDoubleStochasticMatrices,
    ::Any,
    ::Any,
    ::ProjectionRetraction,
)

function retract!(
    M::MultinomialSymmetricDoubleStochasticMatrices,
    q,
    p,
    X,
    ::ProjectionRetraction,
)
    return project!(M, q, p + Y)
end

"""
    vector_transport_to(M::MultinomialSymmetricDoubleStochasticMatrices, p, X, q)

transport the tangent vector `X` at `p` to `q` by projecting it onto the tangent space
at `q`.
"""
vector_transport_to(
    ::MultinomialSymmetricDoubleStochasticMatrices,
    ::Any,
    ::Any,
    ::Any,
    ::ProjectionTransport,
)

function vector_transport_to!(
    M::MultinomialSymmetricDoubleStochasticMatrices,
    Y,
    p,
    X,
    q,
    ::ProjectionTransport,
)
    project!(M, Y, q, X)
    return Y
end

function Base.show(io::IO, ::MultinomialSymmetricDoubleStochasticMatrices{n}) where {n}
    return print(io, "MultinomialSymmetricDoubleStochasticMatrices($(n))")
end
