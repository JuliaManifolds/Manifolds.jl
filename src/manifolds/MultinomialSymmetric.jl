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

It is modeled as [`IsIsometricEmbeddedManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/decorator.html#ManifoldsBase.IsIsometricEmbeddedManifold).
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

More details can be found in Section IV [DouikHassibi:2019](@cite).

# Constructor

    MultinomialSymmetric(n)

Generate the manifold of matrices $\mathbb R^{n×n}$ that are doubly stochastic and symmetric.
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
    return check_point(SymmetricMatrices(n, ℝ), p)
end
@doc raw"""
    check_vector(M::MultinomialSymmetric p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`MultinomialSymmetric`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension, symmetric, and sums to zero
along any row.
"""
function check_vector(M::MultinomialSymmetric{n}, p, X; kwargs...) where {n}
    return check_vector(SymmetricMatrices(n, ℝ), p, X; kwargs...)
end

function get_embedding(::MultinomialSymmetric{N}) where {N}
    return MultinomialMatrices(N, N)
end

embed!(::MultinomialSymmetric, q, p) = copyto!(q, p)
embed!(::MultinomialSymmetric, Y, ::Any, X) = copyto!(Y, X)

"""
    is_flat(::MultinomialSymmetric)

Return false. [`MultinomialSymmetric`](@ref) is not a flat manifold.
"""
is_flat(M::MultinomialSymmetric) = false

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

function retract_project!(M::MultinomialSymmetric, q, p, X, t::Number)
    return project!(M, q, p .* exp.(t .* X ./ p))
end

function Base.show(io::IO, ::MultinomialSymmetric{n}) where {n}
    return print(io, "MultinomialSymmetric($(n))")
end
