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

function MultinomialSymmetricPositiveDefinite(n::Int; parameter::Symbol=:type)
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
    return MultinomialMatrices(n, n)
end
function get_embedding(M::MultinomialSymmetricPositiveDefinite{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return MultinomialMatrices(n, n; parameter=:field)
end
function ManifoldsBase.get_embedding_type(::MultinomialSymmetricPositiveDefinite)
    return ManifoldsBase.EmbeddedManifoldType()
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
3. Symmetrize the projected matrix and return the result.

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
        ds = project(MMDS, random_totally_positive; maxiter=1000)
        p .= (ds .+ ds') ./ 2
        if eigmin(p) > 0
            is_spd = true
        end
    end
    return p
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
