@doc raw"""
    SymmetricPositiveDefinite{N} <: AbstractDecoratorManifold{𝔽}

The manifold of symmetric positive definite matrices, i.e.

````math
\mathcal P(n) =
\bigl\{
p ∈ ℝ^{n × n}\ \big|\ a^\mathrm{T}pa > 0 \text{ for all } a ∈ ℝ^{n}\backslash\{0\}
\bigr\}
````

The tangent space at ``T_p\mathcal P(n)`` reads

```math
    T_p\mathcal P(n) =
    \bigl\{
        X \in \mathbb R^{n×n} \big|\ X=X^\mathrm{T}
    \bigr\},
```
i.e. the set of symmetric matrices,

# Constructor

    SymmetricPositiveDefinite(n)

generates the manifold $\mathcal P(n) \subset ℝ^{n × n}$
"""
struct SymmetricPositiveDefinite{N} <: AbstractDecoratorManifold{ℝ} end

SymmetricPositiveDefinite(n::Int) = SymmetricPositiveDefinite{n}()

@doc raw"""
    SPDPoint <: AbstractManifoldsPoint

Store the result of `eigen(p)` of an SPD matrix and (optionally) ``p^{\frac{1}[2}}`` and ``p^{-\frac{1}[2}}``
to avoid their repeated computations.

This result only has the result of `eigen` as a mandatory storage, the other three
can be stored. If they are not stored they are computed and returned (but then still not stored)
when required.

# Constructor

    SPDPoint(p::AbstractMatrix; store_p=true, store_sqrt=true, store_sqrt_inv=true)

Create an SPD point using an symmetric positive defincite matrix `p`, where you can optionally store `p`, `p_sqrt` and `p_sqrt_inv`
"""
mutable struct SPDPoint{
    P<:Union{<:AbstractMatrix,Missing},
    Q<:Union{<:AbstractMatrix,Missing},
    R<:Union{<:AbstractMatrix,Missing},
    E,
} <: AbstractManifoldPoint
    p::P
    eigen::E
    sqrt::Q
    sqrt_inv::R
end
SPDPoint(p::SPDPoint) = p
function SPDPoint(p::AbstractMatrix; store_p=true, store_sqrt=true, store_sqrt_inv=true)
    e = eigen(Symmetric(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    if store_sqrt
        s_sqrt = Diagonal(sqrt.(S))
        p_sqrt = U * s_sqrt * transpose(U)
    else
        p_sqrt = missing
    end
    if store_sqrt_inv
        s_sqrt_inv = Diagonal(1 ./ sqrt.(S))
        p_sqrt_inv = U * s_sqrt_inv * transpose(U)
    else
        p_sqrt_inv = missing
    end
    if store_p
        q = p
    else
        q = missing
    end
    return SPDPoint{typeof(q),typeof(p_sqrt),typeof(p_sqrt_inv),typeof(e)}(
        q,
        e,
        p_sqrt,
        p_sqrt_inv,
    )
end
convert(::Type{SPDPoint}, p::AbstractMatrix) = SPDPoint(p)

function active_traits(f, ::SymmetricPositiveDefinite, args...)
    return merge_traits(IsEmbeddedManifold(), IsDefaultMetric(LinearAffineMetric()))
end

function allocate(p::SPDPoint)
    pV = Matrix(p)
    e2 = eigen(pV)
    return SPDPoint(
        ismissing(p.p) ? missing : allocate(pV),
        e2,
        ismissing(p.sqrt) ? missing : allocate(p.sqrt),
        ismissing(p.sqrt_inv) ? missing : allocate(p.sqrt_inv),
    )
end
function allocate(p::SPDPoint, ::Type{T}) where {T}
    pV = Matrix(p)
    e2 = eigen(pV)
    return SPDPoint(
        ismissing(p.p) ? missing : allocate(p.p, T),
        e2,
        ismissing(p.sqrt) ? missing : allocate(p.sqrt, T),
        ismissing(p.sqrt_inv) ? missing : allocate(p.sqrt_inv, T),
    )
end

function allocate_result(M::SymmetricPositiveDefinite, zero_vector, p::SPDPoint)
    return allocate_result(M, zero_vector, Matrix(p))
end
function allocate_coordinates(M::SymmetricPositiveDefinite, p::SPDPoint, T, n::Int)
    return allocate_coordinates(M, Matrix(p), T, n)
end
function allocate_result(M::SymmetricPositiveDefinite, get_vector, p::SPDPoint, c)
    return allocate_result(M, get_vector, Matrix(p), c)
end

@doc raw"""
    check_point(M::SymmetricPositiveDefinite, p; kwargs...)

checks, whether `p` is a valid point on the [`SymmetricPositiveDefinite`](@ref) `M`, i.e. is a matrix
of size `(N,N)`, symmetric and positive definite.
The tolerance for the second to last test can be set using the `kwargs...`.
"""
function check_point(M::SymmetricPositiveDefinite{N}, p; kwargs...) where {N}
    if !isapprox(norm(p - transpose(p)), 0.0; kwargs...)
        return DomainError(
            norm(p - transpose(p)),
            "The point $(p) does not lie on $(M) since its not a symmetric matrix:",
        )
    end
    if !all(eigvals(p) .> 0)
        return DomainError(
            eigvals(p),
            "The point $p does not lie on $(M) since its not a positive definite matrix.",
        )
    end
    return nothing
end
function check_point(M::SymmetricPositiveDefinite, p::SPDPoint; kwargs...)
    return check_point(M, Matrix(p); kwargs...)
end

"""
    check_vector(M::SymmetricPositiveDefinite, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`SymmetricPositiveDefinite`](@ref) `M`,
i.e. atfer [`check_point`](@ref)`(M,p)`, `X` has to be of same dimension as `p`
and a symmetric matrix, i.e. this stores tangent vetors as elements of the corresponding
Lie group.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(M::SymmetricPositiveDefinite{N}, p, X; kwargs...) where {N}
    if !isapprox(norm(X - transpose(X)), 0.0; kwargs...)
        return DomainError(
            X,
            "The vector $(X) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not symmetric.",
        )
    end
    return nothing
end
function check_vector(M::SymmetricPositiveDefinite, p::SPDPoint, X; kwargs...)
    return check_vector(M, Matrix(p), X; kwargs...)
end

function check_size(M::SymmetricPositiveDefinite, p::SPDPoint; kwargs...)
    return check_size(M, Matrix(p); kwargs...)
end
function check_size(M::SymmetricPositiveDefinite, p::SPDPoint, X; kwargs...)
    return check_size(M, Matrix(p), X; kwargs...)
end

function Base.copy(p::SPDPoint)
    return SPDPoint(
        is_missing(p.p) ? missing : copy(p.p),
        eigen(Matrix(p)),
        is_missing(p.sqrt) ? missing : copy(p.sqrt),
        is_missing(p.sqrt_inv) ? missing : copy(p.sqrt_inv),
    )
end

function copy(M::SymmetricPositiveDefinite, p::SPDPoint)
    return SPDPoint(
        Matrix(p);
        store_p=!ismissing(p.p),
        store_sqrt=!ismissing(p.sqrt),
        store_sqrt_inv=!ismissing(p.sqrt_inv),
    )
end
#
# Lazy copyto, only copy if both are not missing,
# create from `p` if it is a nonmissing field in q.
#
function copyto!(q::SPDPoint, p::SPDPoint)
    if !ismissing(q.p) # we have to fill the Fields
        if !ismissing(p.p)
            !ismissing(q.p) && copyto!(q.p, p.p)
        else # otherwise compute and copy
            copyto(q.p, Matrix(p))
        end
    end
    copyto!(q.eigen.values, p.eigen.values)
    copyto!(q.eigen.vectors, p.eigen.vectors)
    if !ismissing(q.sqrt)
        if !ismissing(p.sqrt)
            copyto!(q.sqrt, p.sqrt)
        else # otherwise compute and copy
            copyto(q.sqrt, eigvals_sqrt(p))
        end
    end
    if !ismissing(q.sqrt_inv)
        if !ismissing(p.sqrt_inv)
            copyto!(q.sqrt_inv, p.sqrt_inv)
        else # otherwise compute and copy
            copyto(q.sqrt_inv, sqrt_inv(p))
        end
    end
    return q
end

embed(::SymmetricPositiveDefinite, p) = p
embed(::SymmetricPositiveDefinite, p::SPDPoint) = Matrix(p)
embed(::SymmetricPositiveDefinite, p, X) = X

function get_embedding(M::SymmetricPositiveDefinite)
    return Euclidean(representation_size(M)...; field=ℝ)
end

@doc raw"""
    injectivity_radius(M::SymmetricPositiveDefinite[, p])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}[, p])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}[, p])

Return the injectivity radius of the [`SymmetricPositiveDefinite`](@ref).
Since `M` is a Hadamard manifold with respect to the [`LinearAffineMetric`](@ref) and the
[`LogCholeskyMetric`](@ref), the injectivity radius is globally $∞$.
"""
injectivity_radius(::SymmetricPositiveDefinite) = Inf
injectivity_radius(::SymmetricPositiveDefinite, p) = Inf
injectivity_radius(::SymmetricPositiveDefinite, ::AbstractRetractionMethod) = Inf
injectivity_radius(::SymmetricPositiveDefinite, p, ::AbstractRetractionMethod) = Inf

function isapprox(p::SPDPoint, q::SPDPoint; kwargs...)
    return isapprox(Matrix(p), Matrix(q); kwargs...)
end

@doc raw"""
    manifold_dimension(M::SymmetricPositiveDefinite)

returns the dimension of
[`SymmetricPositiveDefinite`](@ref) `M`$=\mathcal P(n), n ∈ ℕ$, i.e.
````math
\dim \mathcal P(n) = \frac{n(n+1)}{2}.
````
"""
@generated function manifold_dimension(::SymmetricPositiveDefinite{N}) where {N}
    return div(N * (N + 1), 2)
end

"""
    mean(
        M::SymmetricPositiveDefinite,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolation();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` using
[`GeodesicInterpolation`](@ref).
"""
mean(::SymmetricPositiveDefinite, ::Any)

function default_estimation_method(::SymmetricPositiveDefinite, ::typeof(mean))
    return GeodesicInterpolation()
end

@doc raw"""
    Matrix(p::SPDPoint)

return the point `p` as a matrix.
The matrix is either stored within the [`SPDPoint`](@ref) or reconstructed from `p.eigen`.
"""
Matrix(p::SPDPoint)
Matrix(p::SPDPoint) = p.p
Matrix(p::SPDPoint{Missing}) = p.eigen.vectors * p.eigen.values * p.eigen.vectors'

@doc raw"""
    eigvals_sqrt(p::AbstractMatrix)
    eigvals_sqrt(p::SPDPoint)

return ``p^{\frac{1}{2}}`` by either computing it (if it is missing or for the `AbstractMatrix`)
or returning the stored value from within the [`SPDPoint`](@ref).

This method assumes that `p` represents an spd matrix.
"""
function eigvals_sqrt(p::AbstractMatrix)
    e = eigen(Symmetric(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    Ssqrt = Diagonal(sqrt.(S))
    return Symmetric(U * Ssqrt * transpose(U))
end
eigvals_sqrt(p::SPDPoint) = Symmetric(p.sqrt)
function eigvals_sqrt(p::SPDPoint{P,Missing}) where {P<:AbstractMatrix}
    U = p.eigen.vectors
    S = max.(p.eigen.values, floatmin(eltype(p.eigen.values)))
    Ssqrt = Diagonal(sqrt.(S))
    return Symmetric(U * Ssqrt * transpose(U))
end
function eigvals_sqrt(p::SPDPoint{Missing,Missing})
    U = p.eigen.vectors
    S = max.(p.eigen.values, floatmin(eltype(p.eigen.values)))
    Ssqrt = Diagonal(sqrt.(S))
    return Symmetric(U * Ssqrt * transpose(U))
end

@doc raw"""
    eigvals_sqrt_inv(p::AbstractMatrix)
    eigvals_sqrt_inv(p::SPDPoint)

return ``p^{-\frac{1}{2}}`` by either computing it (if it is missing or for the `AbstractMatrix`)
or returning the stored value from within the [`SPDPoint`](@ref).

This method assumes that `p` represents an spd matrix.
"""
eigvals_sqrt_inv(p::SPDPoint) = Symmetric(p.sqrt_inv)
function eigvals_sqrt_inv(p::SPDPoint{P,Q,Missing}) where {P,Q}
    U = p.eigen.vectors
    S = max.(p.eigen.values, floatmin(eltype(p.eigen.values)))
    SsqrtInv = Diagonal(1 ./ sqrt.(S))
    return Symmetric(U * SsqrtInv * transpose(U))
end

@doc raw"""
    eigvals_sqrt_and_sqrt_inv(p::AbstractMatrix)
    eigvals_sqrt_and_sqrt_inv(p::SPDPoint)

return ``p^{\frac{1}{2}}`` and ``p^{-\frac{1}{2}}`` by either computing them (if they are missing or for the `AbstractMatrix`)
or returning their stored value from within the [`SPDPoint`](@ref).

Compared to calling single methods [`sqrt`](@ref) and [`sqrt_inv`](@ref) this method
only computes the eigenvectors once for the case of the `AbstractMatrix` or if both are missing.

This method assumes that `p` represents an spd matrix.
"""
_sqrt_and_sqrt_inv(p)

function eigvals_sqrt_and_sqrt_inv(p::AbstractMatrix)
    e = eigen(Symmetric(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    Ssqrt = Diagonal(sqrt.(S))
    SsqrtInv = Diagonal(1 ./ sqrt.(S))
    return (Symmetric(U * Ssqrt * transpose(U)), Symmetric(U * SsqrtInv * transpose(U)))
end
eigvals_sqrt_and_sqrt_inv(p::SPDPoint) = (Symmetric(p.sqrt), Symmetric(p.sqrt_inv))
function eigvals_sqrt_and_sqrt_inv(p::SPDPoint{P,Q,Missing}) where {P,Q}
    return (Symmetric(p.sqrt), eigvals_sqrt_inv(p))
end
function eigvals_sqrt_and_sqrt_inv(p::SPDPoint{P,Missing,R}) where {P,R}
    return (eigvals_sqrt(p), Symmetric(p.sqrt_inv))
end
function eigvals_sqrt_and_sqrt_inv(p::SPDPoint{P,Missing,Missing}) where {P}
    S = max.(p.eigen.values, floatmin(eltype(p.eigen.values)))
    U = p.eigen.vectors
    Ssqrt = Diagonal(sqrt.(S))
    SsqrtInv = Diagonal(1 ./ sqrt.(S))
    return (Symmetric(U * Ssqrt * transpose(U)), Symmetric(U * SsqrtInv * transpose(U)))
end

number_eltype(p::SPDPoint) = number_eltype(Matrix(p))

@doc raw"""
    project(M::SymmetricPositiveDefinite, p, X)

project a matrix from the embedding onto the tangent space $T_p\mathcal P(n)$ of the
[`SymmetricPositiveDefinite`](@ref) matrices, i.e. the set of symmetric matrices.
"""
project(::SymmetricPositiveDefinite, p, X)

project!(::SymmetricPositiveDefinite, Y, p, X) = (Y .= Symmetric((X + X') / 2))

@doc raw"""
    rand(M::SymmetricPositiveDefinite; σ::Real=1)

Generate a random symmetric positive definite matrix on the
`SymmetricPositiveDefinite` manifold `M`.
"""
rand(M::SymmetricPositiveDefinite; σ::Real=1)

function Random.rand!(M::SymmetricPositiveDefinite, pX::SPDPoint; kwargs...)
    p = rand(M; kwargs...)
    pP = SPDPoint(p; store_p=false, store_sqrt=false, store_sqrt_inv=false)
    !ismissing(pX.p) && pX.p .= p
    pX.eigen = pP.eigen
    !ismissing(pX.sqrt) && pX.sqrt .= eigvals_sqrt(pP)
    !ismissing(pX.sqrt_inv) && pX.sqrt_inv .= eigvals_sqrt_inv(pP)
    return pX
end

function Random.rand!(
    M::SymmetricPositiveDefinite{N},
    pX;
    vector_at=nothing,
    σ::Real=one(eltype(pX)) / (vector_at === nothing ? 1 : norm(vector_at)),
    tangent_distr=:Gaussian,
) where {N}
    if vector_at === nothing
        D = Diagonal(1 .+ rand(N)) # random diagonal matrix
        s = qr(σ * randn(N, N)) # random q
        pX .= Symmetric(s.Q * D * transpose(s.Q))
    elseif tangent_distr === :Gaussian
        # generate ONB in TxM
        I = one(vector_at)
        B = get_basis(M, vector_at, DiagonalizingOrthonormalBasis(I))
        Ξ = get_vectors(M, vector_at, B)
        Ξx =
            vector_transport_to.(
                Ref(M),
                Ref(I),
                Ξ,
                Ref(vector_at),
                Ref(ParallelTransport()),
            )
        pX .= sum(σ * randn(length(Ξx)) .* Ξx)
    elseif tangent_distr === :Rician
        C = cholesky(Hermitian(vector_at))
        R = C.L + sqrt(σ) * triu(randn(size(vector_at, 1), size(vector_at, 2)), 0)
        pX .= R * R'
    end
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::SymmetricPositiveDefinite{N},
    pX;
    vector_at=nothing,
    σ::Real=one(eltype(pX)) / (vector_at === nothing ? 1 : norm(vector_at)),
    tangent_distr=:Gaussian,
) where {N}
    if vector_at === nothing
        D = Diagonal(1 .+ rand(rng, N)) # random diagonal matrix
        s = qr(σ * randn(rng, N, N)) # random q
        pX .= Symmetric(s.Q * D * transpose(s.Q))
    elseif tangent_distr === :Gaussian
        # generate ONB in TxM
        I = one(vector_at)
        B = get_basis(M, vector_at, DiagonalizingOrthonormalBasis(I))
        Ξ = get_vectors(M, vector_at, B)
        Ξx =
            vector_transport_to.(
                Ref(M),
                Ref(I),
                Ξ,
                Ref(vector_at),
                Ref(ParallelTransport()),
            )
        pX .= sum(σ * randn(rng, length(Ξx)) .* Ξx)
    elseif tangent_distr === :Rician
        C = cholesky(Hermitian(vector_at))
        R = C.L + sqrt(σ) * triu(randn(rng, size(vector_at, 1), size(vector_at, 2)), 0)
        pX .= R * R'
    end
    return pX
end

@doc raw"""
    representation_size(M::SymmetricPositiveDefinite)

Return the size of an array representing an element on the
[`SymmetricPositiveDefinite`](@ref) manifold `M`, i.e. $n × n$, the size of such a
symmetric positive definite matrix on $\mathcal M = \mathcal P(n)$.
"""
@generated representation_size(::SymmetricPositiveDefinite{N}) where {N} = (N, N)

function Base.show(io::IO, ::SymmetricPositiveDefinite{N}) where {N}
    return print(io, "SymmetricPositiveDefinite($(N))")
end

function Base.show(io::IO, ::MIME"text/plain", p::SPDPoint)
    pre = " "
    summary(io, p)
    println(io, "\np:")
    sp = sprint(show, "text/plain", p.p; context=io, sizehint=0)
    sp = replace(sp, '\n' => "\n$(pre)")
    println(io, pre, sp)
    println(io, "p^{1/2}:")
    sps = sprint(show, "text/plain", p.sqrt; context=io, sizehint=0)
    sps = replace(sps, '\n' => "\n$(pre)")
    println(io, pre, sps)
    println(io, "p^{-1/2}:")
    spi = sprint(show, "text/plain", p.sqrt_inv; context=io, sizehint=0)
    spi = replace(spi, '\n' => "\n$(pre)")
    return print(io, pre, spi)
end

@doc raw"""
    zero_vector(M::SymmetricPositiveDefinite, p)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `p` on the [`SymmetricPositiveDefinite`](@ref) manifold `M`.
"""
zero_vector(::SymmetricPositiveDefinite, ::Any)

zero_vector(M::SymmetricPositiveDefinite, p::SPDPoint) = zero_vector(M, Matrix(p))

zero_vector!(::SymmetricPositiveDefinite{N}, v, ::Any) where {N} = fill!(v, 0)
