@doc raw"""
    MultinomialSymmetric{n} <: AbstractMultinomialDoublyStochastic{N}

The multinomial symmetric matrices manifold consists of all symmetric $n×n$ matrices with
positive entries such that each column sums to one, i.e.

````math
\begin{aligned}
\mathcal{SP}(n) \coloneqq \bigl\{p ∈ ℝ^{n×n}\ \big|\ &p_{i,j} > 0 \text{ for all } i=1,…,n, j=1,…,m,\\
& p^\mathrm{T} = p,\\
& p\mathbf{1}_n = \mathbf{1}_n
\bigr\},
\end{aligned}
````

where $\mathbf{1}_n$ is the vector of length $n$ containing ones.

It is modeled as an [`DefaultIsometricEmbeddingType`](@ref), [`AbstractEmbeddedManifold`](@ref)
via the [`AbstractMultinomialDoublyStochastic`](@ref) type, since it shares a few functions
also with [`AbstractMultinomialDoublyStochastic`](@ref), most and foremost projection of
a point from the embedding onto the manifold.

The tangent space can be written as

````math
T_p\mathcal{SP}(n) \coloneqq \bigl\{
X ∈ ℝ^{n×n}\ \big|\ X = X^{\mathrm{T}} \text{ and }
X\mathbf{1}_n = \mathbf{0}_n
\bigr\},
````

where $\mathbf{0}_n$ is the vector of length $n$ containing zeros.

More details can be found in Section IV[^DouikHassibi2019].

# Constructor

    MultinomialSymmetric(n)

Generate the manifold of matrices $\mathbb R^{n×n}$ that are doubly stochastic and symmetric.

[^DouikHassibi2019]:
    > A. Douik, B. Hassibi:
    > AbstractManifold Optimization Over the Set of Doubly Stochastic Matrices: A Second-Order Geometry,
    > IEEE Transactions on Signal Processing 67(22), pp. 5761–5774, 2019.
    > doi: [10.1109/tsp.2019.2946024](http://doi.org/10.1109/tsp.2019.2946024),
    > arXiv: [1802.02628](https://arxiv.org/abs/1802.02628).
"""
struct MultinomialSymmetric{N} <: AbstractMultinomialDoublyStochastic{N} end

function MultinomialSymmetric(n::Int)
    return MultinomialSymmetric{n}()
end

@doc raw"""
    check_point(M::MultinomialSymmetric, p)

Checks whether `p` is a valid point on the [`MultinomialSymmetric`](@ref)`(m,n)` `M`,
i.e. is a symmetric matrix with positive entries whose rows sum to one.
"""
function check_point(M::MultinomialSymmetric{n}, p; kwargs...) where {n}
    mpv = invoke(check_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    # the embedding checks for positivity and unit sum columns, by symmetry we would get
    # the same for the rows, so checking symmetry is the only thing left, we can just use
    # the corresponding manifold for that
    return check_point(SymmetricMatrices(n, ℝ), p)
end
@doc raw"""
    check_vector(M::MultinomialSymmetric p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialSymmetric`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension, symmetric, and sums to zero
along any row.
"""
function check_vector(M::MultinomialSymmetric{n}, p, X; kwargs...) where {n}
    mpv = invoke(
        check_vector,
        Tuple{supertype(typeof(M)),typeof(p),typeof(X)},
        M,
        p,
        X;
        kwargs...,
    )
    mpv === nothing || return mpv
    # from the embedding we know that columns sum to zero, only symmety is left, i.e.
    return check_vector(SymmetricMatrices(n, ℝ), p, X; kwargs...)
end

function decorated_manifold(::MultinomialSymmetric{N}) where {N}
    return MultinomialMatrices(N, N)
end

embed!(::MultinomialSymmetric, q, p) = copyto!(q, p)
embed!(::MultinomialSymmetric, Y, ::Any, X) = copyto!(Y, X)

@doc raw"""
    manifold_dimension(M::MultinomialSymmetric{n}) where {n}

returns the dimension of the [`MultinomialSymmetric`](@ref) manifold
namely
````math
\operatorname{dim}_{\mathcal{SP}(n)} = \frac{n(n-1)}{2}.
````
"""
@generated function manifold_dimension(::MultinomialSymmetric{n}) where {n}
    return div(n * (n - 1), 2)
end

@doc raw"""
    project(M::MultinomialSymmetric{n}, p, Y) where {n}

Project `Y` onto the tangent space at `p` on the [`MultinomialSymmetric`](@ref) `M`, return the result in `X`.
The formula reads
````math
    \operatorname{proj}_p(Y) = Y - (α\mathbf{1}_n^{\mathrm{T}} + \mathbf{1}_n α^{\mathrm{T}}) ⊙ p,
````
where $⊙$ denotes the Hadamard or elementwise product and $\mathbb{1}_n$ is the vector of length $n$ containing ones.
The two vector $α ∈ ℝ^{n×n}$ is given by solving
````math
    (I_n+p)α =  Y\mathbf{1},
````
where $I_n$ is teh $n×n$ unit matrix and $\mathbf{1}_n$ is the vector of length $n$ containing ones.

"""
project(::MultinomialSymmetric, ::Any, ::Any)

function project!(::MultinomialSymmetric{n}, X, p, Y) where {n}
    α = (I + p) \ sum(Y, dims=2) # Formula (49) from 1802.02628
    return X .= Y .- (repeat(α, 1, 3) .+ repeat(α', 3, 1)) .* p
end

@generated function representation_size(::MultinomialSymmetric{n}) where {n}
    return (n, n)
end

@doc raw"""
    retract(M::MultinomialSymmetric, p, X, ::ProjectionRetraction)

compute a projection based retraction by projecting $p\odot\exp(X⨸p)$ back onto the manifold,
where $⊙,⨸$ are elementwise multiplication and division, respectively. Similarly, $\exp$
refers to the elementwise exponentiation.
"""
retract(::MultinomialSymmetric, ::Any, ::Any, ::ProjectionRetraction)

function retract!(M::MultinomialSymmetric, q, p, X, ::ProjectionRetraction)
    return project!(M, q, p .* exp.(X ./ p))
end

"""
    vector_transport_to(M::MultinomialSymmetric, p, X, q)

transport the tangent vector `X` at `p` to `q` by projecting it onto the tangent space
at `q`.
"""
vector_transport_to(::MultinomialSymmetric, ::Any, ::Any, ::Any, ::ProjectionTransport)

function vector_transport_to!(M::MultinomialSymmetric, Y, p, X, q, ::ProjectionTransport)
    project!(M, Y, q, X)
    return Y
end

function Base.show(io::IO, ::MultinomialSymmetric{n}) where {n}
    return print(io, "MultinomialSymmetric($(n))")
end
