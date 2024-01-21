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
    n = get_parameter(M.size)[1]
    s = check_point(SymmetricPositiveDefinite(n), p; kwargs...)
    isnothing(s) && return s
    s2 = check_point(MultinomialMatrices(n, n), p; kwargs...)
    return s2
end

function check_vector(M::MultinomialSymmetricPositiveDefinite, p, X; kwargs...)
    n = get_parameter(M.size)[1]
    s = check_vector(SymmetricPositiveDefinite(n), p, X; kwargs...)
    isnothing(s) && return s
    s2 = check_vector(MultinomialMatrices(n, n), p, X)
    return s2
end
