@doc raw"""
    SymmetricPositiveDefinite{T} <: AbstractDecoratorManifold{ℝ}

The manifold of symmetric positive definite matrices, i.e.

````math
\mathcal P(n) = \bigl\{
  p ∈ ℝ^{n×n}\ \big|\ p = p^{\mathrm{T}} \text{ and } a^\mathrm{T}pa > 0 \text{ for all } a ∈ ℝ^{n}\backslash\{0\}
\bigr\}
````

The tangent space at ``p ∈ \mathcal P(n)`` reads

```math
    T_p\mathcal P(n) =
    \bigl\{ X \in \mathbb R^{n×n} \big|\ X=X^\mathrm{T} \bigr\},
```
i.e. the set of symmetric matrices.

This manifold is the real case ``𝔽 = ℝ`` of the [`HermitianPositiveDefinite`](@ref) matrices,
``\mathcal P(n) = \mathcal H(n)``.

# Constructor

    SymmetricPositiveDefinite(n; parameter::Symbol=:type)

generates the manifold ``\mathcal P(n) \subset ℝ^{n×n}``
"""
const SymmetricPositiveDefinite{T} = HermitianPositiveDefinite{ℝ, T}

function SymmetricPositiveDefinite(n::Int; parameter::Symbol = :type)
    return HermitianPositiveDefinite(n, ℝ; parameter = parameter)
end

@doc raw"""
    injectivity_radius(M::SymmetricPositiveDefinite[, p])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,AffineInvariantMetric}[, p])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}[, p])

Return the injectivity radius of the [`SymmetricPositiveDefinite`](@ref).
Since `M` is a Hadamard manifold with respect to the [`AffineInvariantMetric`](@ref) and the
[`LogCholeskyMetric`](@ref), the injectivity radius is globally ``∞``.
"""
injectivity_radius(::SymmetricPositiveDefinite, ::Any...)

@doc raw"""
    manifold_dimension(M::SymmetricPositiveDefinite)

returns the dimension of
[`SymmetricPositiveDefinite`](@ref) `M` ``=\mathcal P(n), n ∈ ℕ``, i.e.
````math
\dim \mathcal P(n) = \frac{n(n+1)}{2}.
````
"""
function manifold_dimension(M::SymmetricPositiveDefinite)
    n = get_parameter(M.size)[1]
    return div(n * (n + 1), 2)
end

@doc raw"""
    project(M::SymmetricPositiveDefinite, p, X)

project a matrix from the embedding onto the tangent space ``T_p\mathcal P(n)`` of the
[`SymmetricPositiveDefinite`](@ref) matrices, i.e. the set of symmetric matrices.
"""
project(::SymmetricPositiveDefinite, p, X)

@doc raw"""
    rand(M::SymmetricPositiveDefinite; vector_at=nothing, tangent_distr=:Gaussian, σ::Real=1/sqrt(n))

Generate a random symmetric positive definite matrix on the
`SymmetricPositiveDefinite` manifold `M`, or, if `vector_at` is a point ``p``, a random
tangent vector at ``p``. The parameter `σ` has no effect on a point.

For `tangent_distr = :Gaussian` the coordinates of the tangent vector in an orthonormal basis
of ``T_p\mathcal P(n)`` are independent and normally distributed with standard deviation `σ`.
For `tangent_distr = :Rician` the tangent vector is ``RR^{\mathrm{T}}`` with
``R = L + \sqrt{σ}U``, where ``L`` is the lower triangular Cholesky factor of ``p`` and ``U`` is
upper triangular with independent standard normal entries; here `σ` defaults to
``1/\lVert p\rVert_{\mathrm{F}}`` instead.
"""
rand(M::SymmetricPositiveDefinite; σ::Real = 1)

function Random.rand!(
        rng::AbstractRNG,
        M::SymmetricPositiveDefinite,
        pX;
        vector_at = nothing,
        tangent_distr = :Gaussian,
        σ::Real = (tangent_distr === :Rician && vector_at !== nothing) ?
            one(eltype(pX)) / norm(convert(AbstractMatrix, vector_at)) :
            one(eltype(pX)) / sqrt(get_parameter(M.size)[1]),
    )
    N = get_parameter(M.size)[1]
    if vector_at === nothing
        D = Diagonal(1 .+ rand(rng, N)) # random diagonal matrix
        s = qr(σ * randn(rng, N, N)) # random q
        if pX isa MatrixSqrtManifoldPoint
            pX.eigen.values .= D.diag
            pX.eigen.vectors .= s.Q
            sq = sqrt.(D.diag)
            !ismissing(pX.p) && (pX.p .= Symmetric(s.Q * D * transpose(s.Q)))
            !ismissing(pX.sqrt) && (pX.sqrt .= s.Q * Diagonal(sq) * transpose(s.Q))
            !ismissing(pX.sqrt_inv) && (pX.sqrt_inv .= s.Q * Diagonal(1 ./ sq) * transpose(s.Q))
        else
            pX .= Symmetric(s.Q * D * transpose(s.Q))
        end
    elseif tangent_distr === :Gaussian
        # an orthonormal basis of the tangent space at vector_at
        E = one(convert(AbstractMatrix, vector_at))
        Ξ = get_vectors(M, vector_at, get_basis(M, vector_at, DiagonalizingOrthonormalBasis(E)))
        pX .= sum(σ * randn(rng, length(Ξ)) .* Ξ)
    elseif tangent_distr === :Rician
        C = cholesky(Hermitian(convert(AbstractMatrix, vector_at)))
        R = C.L + sqrt(σ) * triu(randn(rng, N, N), 0)
        pX .= R * R'
    end
    return pX
end

function Base.show(io::IO, M::SymmetricPositiveDefinite)
    n = get_parameter(M.size)[1]
    kw = get_parameter_type(M) === :type ? "" : "; parameter=:$(get_parameter_type(M))"
    return print(io, "SymmetricPositiveDefinite($n$kw)")
end

@doc raw"""
    zero_vector(M::SymmetricPositiveDefinite, p)
    zero_vector!(M::SymmetricPositiveDefinite, X, p)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `p` on the [`SymmetricPositiveDefinite`](@ref) manifold `M`.
It can be returned in place of `X`.
"""
zero_vector(::SymmetricPositiveDefinite, ::Any)
