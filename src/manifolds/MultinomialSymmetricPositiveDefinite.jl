@doc raw"""
    MultinomialSymmetricPositiveDefinite <: AbstractMultinomialDoublyStochastic

The symmetric positive definite multinomial matrices manifold consists of all
symmetric ``n×n`` matrices with positive eigenvalues, and
positive entries such that each column sums to one, i.e.

````math
\begin{aligned}
\mathcal{SP}^+(n) \coloneqq \bigl\{
    p ∈ ℝ^{n×n}\ \big|\ &p_{i,j} > 0 \text{ for all } i=1,…,n, j=1,…,m,\\
& p^\mathrm{T} = p,\\
& p\mathbf{1}_n = \mathbf{1}_n\\
a^\mathrm{T}pa > 0 \text{ for all } a ∈ ℝ^{n}\backslash\{\mathbf{0}_n\}
\bigr\},
\end{aligned}
````

where ``\mathbf{1}_n`` and ``\mathbr{0}_n`` are the vectors of length ``n``
containing ones and zeros, respectively. More details about this manifold can be found in
[DouikHassibi:2019](@cite).

# Constructor

    MultinomialSymmetricPositiveDefinite(n)

Generate the manifold of matrices ``\mathbb R^{n×n}`` that are symmetric, positive definite, and doubly stochastic.
"""
struct MultinomialSymmetricPositiveDefinite{T} <: AbstractMultinomialDoublyStochastic
    size::T
end

function MultinomialSymmetricPositiveDefinite(n::Int; parameter::Symbol = :type)
    size = wrap_type_parameter(parameter, (n,))
    return MultinomialSymmetricPositiveDefinite{typeof(size)}(size)
end

function check_point(M::MultinomialSymmetricPositiveDefinite, p; kwargs...)
    # Multinomial checked first via embedding
    n = get_parameter(M.size)[1]
    s = check_point(SymmetricPositiveDefinite(n), p; kwargs...)
    !isnothing(s) &&
        return ManifoldDomainError("The point $(p) does not lie on the $(M).", s)
    return nothing
end

function check_vector(M::MultinomialSymmetricPositiveDefinite, p, X; kwargs...)
    # Multinomial checked first via embedding
    n = get_parameter(M.size)[1]
    s = check_vector(SymmetricPositiveDefinite(n), p, X; kwargs...)
    !isnothing(s) && return ManifoldDomainError(
        "The vector $(X) is not a tangent vector to $(p) on $(M)",
        s,
    )
    return nothing
end

function get_embedding(
        ::MultinomialSymmetricPositiveDefinite{TypeParameter{Tuple{n}}},
    ) where {n}
    return MultinomialSymmetric(n)
end
function get_embedding(M::MultinomialSymmetricPositiveDefinite{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return MultinomialSymmetric(n; parameter = :field)
end
function ManifoldsBase.get_embedding_type(::MultinomialSymmetricPositiveDefinite)
    return ManifoldsBase.IsometricallyEmbeddedManifoldType()
end

"""
    is_flat(M::MultinomialSymmetricPositiveDefinite)

Return whether the [`MultinomialSymmetricPositiveDefinite`](@ref) `M` is flat,
which is the case if and only if its dimension is one.
"""
is_flat(M::MultinomialSymmetricPositiveDefinite) = manifold_dimension(M) == 1

@doc raw"""
    manifold_dimension(M::MultinomialSymmetricPositiveDefinite)

Return the dimension of the [`MultinomialSymmetricPositiveDefinite`](@ref) manifold,
````math
\operatorname{dim}_{\mathcal{SP}^+(n)} = \frac{n(n-1)}{2},
````
the dimension of the polytope of symmetric doubly stochastic matrices, see Section 1 of [Davis:2015](@cite),
of which the manifold is an open subset, see pp. 36 and 41 of [Douik:2020](@cite).
"""
function manifold_dimension(M::MultinomialSymmetricPositiveDefinite)
    n = get_parameter(M.size)[1]
    return div(n * (n - 1), 2)
end

@doc raw"""
    project(M::MultinomialSymmetricPositiveDefinite, p, X)

Project `X` onto the tangent space at `p` on the [`MultinomialSymmetricPositiveDefinite`](@ref) `M`.
The manifold is an open subset of the [`MultinomialSymmetric`](@ref) matrices and has their
geometry, see Section 3.3, p. 39 of [Douik:2020](@cite), so with the symmetric part
``X_{\mathrm{s}} = \frac{1}{2}(X+X^{\mathrm{T}})`` the projection reads

````math
    \operatorname{proj}_p(X) = X_{\mathrm{s}} - (α\mathbf{1}_n^{\mathrm{T}} + \mathbf{1}_n α^{\mathrm{T}}) ⊙ p,
    \qquad (I_n+p)α = X_{\mathrm{s}}\mathbf{1}_n,
````

where ``⊙`` denotes the elementwise product and ``\mathbf{1}_n`` the vector of ``n`` ones.
"""
project(::MultinomialSymmetricPositiveDefinite, ::Any, ::Any)

function project!(M::MultinomialSymmetricPositiveDefinite, Y, p, X)
    return project!(get_embedding(M), Y, p, X)
end

"""
    Random.rand!(
        rng::AbstractRNG,
        M::MultinomialSymmetricPositiveDefinite,
        p::AbstractMatrix,
    )

Generate a random point on [`MultinomialSymmetricPositiveDefinite`](@ref) manifold.
The steps are as follows:
1. Generate a random [totally positive matrix](https://en.wikipedia.org/wiki/Totally_positive_matrix)
    a. Construct a vector `L` of `n` random positive increasing real numbers.
    b. Construct the [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix)
       `V` based on the sequence `L`.
    c. Perform LU factorization of `V` in such way that both L and U components have
       positive elements.
    d. Convert the LU factorization into LDU factorization by taking the diagonal of U
       and dividing U by it, `V=LDU`.
    e. Construct a new matrix `R = UDL` which is totally positive.
2. Project the totally positive matrix `R` onto the manifold of [`MultinomialDoubleStochastic`](@ref)
   matrices.
3. Symmetrize the projected matrix and return the result. If the projection did not
   converge or the result is not positive definite, start over from step 1.

This method roughly follows the procedure described in https://math.stackexchange.com/questions/2773460/how-to-generate-a-totally-positive-matrix-randomly-using-software-like-maple
"""
function Random.rand!(
        rng::AbstractRNG,
        M::MultinomialSymmetricPositiveDefinite,
        p::AbstractMatrix,
    )
    n = get_parameter(M.size)[1]
    is_spd = false
    while !is_spd
        L = sort(exp.(randn(rng, n)))
        V = reduce(hcat, map(xi -> [xi^k for k in 0:(n - 1)], L))'
        Vlu = lu(V, LinearAlgebra.RowNonZero())
        dm = Diagonal(Vlu.U)
        uutd = dm \ Vlu.U
        random_totally_positive = uutd * dm * Vlu.L
        MMDS = MultinomialDoubleStochastic(n)
        ds = project(
            MMDS, random_totally_positive; maxiter = 1000, warn_nonconvergence = false
        )
        p .= (ds .+ ds') ./ 2
        # Sinkhorn's algorithm converges very slowly for nearly decomposable matrices,
        # so reject samples for which it did not converge
        sinkhorn_converged =
            maximum(abs.(sum(p; dims = 2) .- 1)) <= 2 * n * eps(eltype(p))
        if sinkhorn_converged && eigmin(p) > 0
            is_spd = true
        end
    end
    return p
end

@doc raw"""
    retract(M::MultinomialSymmetricPositiveDefinite, p, X, ::ProjectionRetraction)

Compute the projection retraction of the [`MultinomialSymmetric`](@ref) matrices, which projects
``p⊙\exp(X⨸p)`` onto the doubly stochastic matrices, where ``⊙,⨸`` are elementwise
multiplication and division and ``\exp`` is the elementwise exponential.
The result is positive definite only for tangent vectors `X` small enough,
see Section 3.3, p. 39 of [Douik:2020](@cite).
"""
retract(::MultinomialSymmetricPositiveDefinite, ::Any, ::Any, ::ProjectionRetraction)

function ManifoldsBase.retract_project!(M::MultinomialSymmetricPositiveDefinite, q, p, X)
    N = get_embedding(M)
    return ManifoldsBase.retract_project!(N, q, p, X)
end

function Base.show(
        io::IO,
        ::MultinomialSymmetricPositiveDefinite{TypeParameter{Tuple{n}}},
    ) where {n}
    return print(io, "MultinomialSymmetricPositiveDefinite($(n))")
end
function Base.show(io::IO, M::MultinomialSymmetricPositiveDefinite{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "MultinomialSymmetricPositiveDefinite($(n); parameter=:field)")
end
