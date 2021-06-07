
@doc raw"""
    Tucker{N, R, D, 𝔽} <: AbstractManifold{𝔽}

The manifold of $N_1 \times \dots \times N_D$ real-valued or complex-valued tensors of
fixed multilinear rank $(R_1, \dots, R_D)$ . If $R_1 = \dots = R_D = 1$, this is the
manifold of rank-1 tensors.

# Representation in HOSVD format

Any tensor $\mathcal{A}$ on the Tucker manifold can be represented in HOSVD
[^DeLathauwer2000] form
```math
\mathcal{A} = (U_1,\dots,U_D) \cdot \mathcal{C}
```
where $\mathcal C \in \mathbb{F}^{R_1 \times \dots \times R_D}$ and, for $d=1,\dots,D$,
the matrix $U_d \in \mathbb{F}^{N_d \times R_d}$ contains the singular vectors of the
$d$th unfolding of $\mathcal{A}$

# Tangent space

The tangent space to the Tucker manifold at
$\mathcal{A} = (U_1,\dots,U_D) \cdot \mathcal{C}$ is [^Koch2010]
```math
T_{\mathcal{A}} \mathcal{M} =
\bigl\{
    (U_1,\dots,U_D) \cdot \dot{\mathcal{C}}
    + \sum_{d=1}^D \bigl(
        (U_1, \dots, U_{d-1}, \dot{U}_d, U_{d+1}, \dots, U_D)
        \cdot \mathcal{C}
    \bigr)
\bigr\}
```
where $\dot{\mathcal{C}}$ is arbitrary and $\dot{U}_d^{\mathrm{H}} U_d = 0$ for all $d$.

# Constructor
    Tucker(n⃗ :: NTuple{D, Int}, r⃗ :: NTuple{D, Int}[, field = ℝ])

Generate the manifold of $N_1 \times \dots \times N_D$ tensors of fixed multilinear rank
$(R_1, \dots, R_D)$

[^DeLathauwer2000]:
    > Lieven De Lathauwer, Bart De Moor, Joos Vandewalle: "A multilinear singular value decomposition"
    > SIAM Journal on Matrix Analysis and Applications, 21(4), pp. 1253-1278, 2000
    > doi: [10.1137/S0895479896305696](https://doi.org/10.1137/S0895479896305696)

[^Koch2010]:
    > Othmar Koch, Christian Lubic, "Dynamical Tensor approximation"
    > SIAM Journal on Matrix Analysis and Applications, 31(5), pp. 2360-2375, 2010
    > doi: [10.1137/09076578X](https://doi.org/10.1137/09076578X)
"""
struct Tucker{N,R,D,𝔽} <: AbstractManifold{𝔽} end
function Tucker(n⃗::NTuple{D,Int}, r⃗::NTuple{D,Int}, field::AbstractNumbers=ℝ) where {D}
    @assert isValidTuckerRank(n⃗, r⃗)
    return Tucker{n⃗,r⃗,D,field}()
end

#=
    HOSVD{T, D}

Higher-order singular value decomposition of an order D tensor with eltype T
fields:
* U: singular vectors of the unfoldings
* core: core tensor
* σ : singular values of the unfoldings
=#
struct HOSVD{T,D}
    U::NTuple{D,Matrix{T}}
    core::Array{T,D}
    σ::NTuple{D,Vector{T}}
end

@doc raw"""
    HOSVDRetraction <: AbstractRetractionMethod

Retraction based on the higher-order singular value decomposition of a tensor
"""
struct HOSVDRetraction <: AbstractRetractionMethod end

"""
    TuckerPoint{T, D}

An order D tensor of fixed multilinear rank and entries of type T. The tensor is
represented in HOSVD form. See also [`Tucker`](@ref).

# Constructors:
    TuckerPoint(core :: AbstractArray{T, D}, factors :: Vararg{MtxT, D}) where {T, D, MtxT <: AbstractMatrix{T}}

A tensor of the form (factors[1], …, factors[D]) ⋅ core
where it is assumed that the dimensions of the core are the multilinear rank of the tensor.

    TuckerPoint(A :: AbstractArray{T, D}, mlrank :: NTuple{D, Int}) where {T, D}

The low-multilinear rank tensor arising from the sequentially truncated the higher-order
singular value decomposition of A [^Vannieuwenhoven2012].

[^Vannieuwenhoven2012]:
    > Nick Vannieuwenhoven, Raf Vandebril, Karl Meerbergen: "A new truncation strategy for the higher-order singular value decomposition"
    > SIAM Journal on Scientific Computing, 34(2), pp. 1027-1052, 2012
    > doi: [10.1137/110836067](https://doi.org/10.1137/110836067)

"""
struct TuckerPoint{T,D} <: AbstractManifoldPoint
    hosvd::HOSVD{T,D}
end
function TuckerPoint(
    core::AbstractArray{T,D}, factors::Vararg{MtxT,D}
) where {T,D,MtxT<:AbstractMatrix{T}}
    # Take the QR decompositions of the factors and multiply the R factors into the core
    qrfacs = qr.(factors)
    Q = map(qrfac -> qrfac.Q, qrfacs)
    R = map(qrfac -> qrfac.R, qrfacs)
    core′ = reshape(Kronecker.:⊗(reverse(R)...) * vec(core), size(core))

    # Convert to HOSVD format by taking the HOSVD of the core
    decomp = st_hosvd(core′)
    factors′ = Q .* decomp.U
    return TuckerPoint(HOSVD{T,D}(factors′, decomp.core, decomp.σ))
end
function TuckerPoint(A::AbstractArray{T,D}, mlrank::NTuple{D,Int}) where {T,D}
    return TuckerPoint(st_hosvd(A, mlrank))
end

@doc raw"""
    TuckerTVector{T, D} <: TVector

Tangent space to the Tucker manifold at $x = (U_1,\dots,U_D) ⋅ \mathcal{C}$. This vector is
represented as
```math
(U_1,\dots,U_D) \cdot \dot{\mathcal{C}} +
\sum_{d=1}^D (U_1,\dots,U_{d-1},\dot{U}_d,U_{d+1},\dots,U_D) \cdot \mathcal{C}
```
where $\dot{U}_d^\mathrm{H} U_d = 0$. See also [`Tucker`](@ref)
"""
struct TuckerTVector{T,D} <: TVector
    Ċ::Array{T,D}
    U̇::NTuple{D,Matrix{T}}
end

#=
    HOSVDBasis{T, D}

A implicitly stored basis of the tangent space to the Tucker manifold.
If 𝔄 = (U¹ ⊗ ... ⊗ Uᴰ) C is a HOSVD, then this basis is defined as follows:

ℬ = {(U¹ ⊗ ... ⊗ Uᴰ) eᵢ} ∪ {(U¹ ⊗ ... ⊗ 1/σ[d][j] Uᵈ⊥ eᵢ eⱼᵀ ⊗ ... ⊗ Uᴰ) C}

See also:
[^Dewaele2021]
> Nick Dewaele, Paul Breiding, Nick Vannieuwenhoven, "The condition number of many tensor decompositions is invariant under Tucker compression"
#TODO arXiv
=#
struct HOSVDBasis{T,D}
    point::TuckerPoint{T,D}
    U⊥::NTuple{D,Matrix{T}}
end
CachedHOSVDBasis{𝔽,T,D} = CachedBasis{
    𝔽,DefaultOrthonormalBasis{𝔽,TangentSpaceType},HOSVDBasis{T,D}
}

⊗ᴿ(a...) = Kronecker.:⊗(reverse(a)...)

Base.:*(s::Number, x::TuckerTVector) = TuckerTVector(s * x.Ċ, s .* x.U̇)
Base.:*(x::TuckerTVector, s::Number) = TuckerTVector(x.Ċ * s, x.U̇ .* s)
Base.:/(x::TuckerTVector, s::Number) = TuckerTVector(x.Ċ / s, x.U̇ ./ s)
Base.:\(s::Number, x::TuckerTVector) = TuckerTVector(s \ x.Ċ, s .\ x.U̇)
Base.:+(x::TuckerTVector, y::TuckerTVector) = TuckerTVector(x.Ċ + y.Ċ, x.U̇ .+ y.U̇)
Base.:-(x::TuckerTVector, y::TuckerTVector) = TuckerTVector(x.Ċ - y.Ċ, x.U̇ .- y.U̇)
Base.:-(x::TuckerTVector) = TuckerTVector(-x.Ċ, map(-, x.U̇))
Base.:+(x::TuckerTVector) = TuckerTVector(x.Ċ, x.U̇)
Base.:(==)(x::TuckerTVector, y::TuckerTVector) = (x.Ċ == y.Ċ) && all(x.U̇ .== y.U̇)

allocate(p::TuckerPoint) = allocate(p, number_eltype(p))
function allocate(p::TuckerPoint, ::Type{T}) where {T}
    # This is not necessarily a valid HOSVD but it's not worth computing the HOSVD
    # just for allocation
    return TuckerPoint(
        HOSVD(allocate(p.hosvd.U, T), allocate(p.hosvd.core, T), allocate(p.hosvd.σ, T))
    )
end
allocate(x::TuckerTVector) = allocate(x, number_eltype(x))
function allocate(x::TuckerTVector, ::Type{T}) where {T}
    return TuckerTVector(allocate(x.Ċ, T), allocate(x.U̇, T))
end

@doc raw"""
    check_point(M::Tucker{N,R,D}, x; kwargs...) where {N,R,D}

Check whether the array or [`TuckerPoint`](@ref) x is a point on the [`Tucker`](@ref)
manifold, i.e. it is an $N_1 \times \dots \times N_D$ tensor of multilinear rank
$(R_1,\dots,R_D)$. The keyword arguments are passed to the matrix rank function applied to
the unfoldings.
For a [`TuckerPoint`](@ref) it is checked that the point is in correct HOSVD form.
"""
function check_point(M::Tucker{N,R,D}, x; kwargs...) where {N,R,D}
    s = "The point $(x) does not lie on $(M), "
    size(x) == N || return DomainError(size(x), s * "since its size is not $(N).")
    for d in 1:ndims(x)
        r = rank(unfold(x, d); kwargs...)
        r == R[d] || return DomainError(size(x), s * "since its rank is not $(R).")
    end
    return nothing
end
function check_point(M::Tucker{N,R,D}, x::TuckerPoint; kwargs...) where {N,R,D}
    s = "The point $(x) does not lie on $(M), "
    U = x.hosvd.U
    ℭ = x.hosvd.core
    ncolsU = map(u -> size(u, 2), U)
    if ncolsU ≠ size(ℭ)
        return DomainError(
            ncolsU, s * "since the dimensions of the Tucker factors do not match"
        )
    end
    if size(ℭ) ≠ R
        return DomainError(
            size(x.hosvd.core), s * "since the size of the core is not $(R)."
        )
    end
    if size(x) ≠ N
        return DomainError(size(x), s * "since its dimensions are not $(N).")
    end
    for u in U
        if u' * u ≉ LinearAlgebra.I
            return DomainError(
                norm(u' * u - LinearAlgebra.I),
                s * "since its factor matrices are not unitary.",
            )
        end
    end
    for d in 1:ndims(x.hosvd.core)
        gram = unfold(ℭ, d) * unfold(ℭ, d)'
        if gram ≉ Diagonal(x.hosvd.σ[d])^2
            return DomainError(
                norm(gram - Diagonal(x.hosvd.σ[d])^2),
                s *
                "since the unfoldings of the core are not diagonalised by" *
                "the singular values.",
            )
        end
        if rank(Diagonal(x.hosvd.σ[d]); kwargs...) ≠ R[d]
            return DomainError(
                minimum(x.hosvd.σ[d]),
                s * "since the core does not have full multilinear rank.",
            )
        end
    end
    return nothing
end

@doc raw"""
    check_vector(M::Tucker{N,R,D}, p::TuckerPoint{T,D}, v::TuckerTVector) where {N,R,T,D}

Check whether a [`TuckerTVector`](@ref) `v` is is in the tangent space to `M` at `p`. This
is the case when the dimensions of the factors in `v` agree with those of `p` and the factor
matrices of `v` are in the orthogonal complement of the HOSVD factors of `p`.
"""
function check_vector(
    M::Tucker{N,R,D}, p::TuckerPoint{T,D}, v::TuckerTVector
) where {N,R,T,D}
    s = "The tangent vector $(v) is not a tangent vector to $(p) on $(M), "
    if size(p.hosvd.core) ≠ size(v.Ċ) || any(size.(v.U̇) .≠ size.(p.hosvd.U))
        return DomainError(
            size(v.Ċ), s * "since the array dimensons of $(p) and $(v)" * "do not agree."
        )
    end
    for (U, U̇) in zip(p.hosvd.U, v.U̇)
        if norm(U' * U̇) ≥ √eps(eltype(U)) * √length(U)
            return DomainError(
                norm(U' * U̇),
                s *
                "since the columns of x.hosvd.U are not" *
                "orthogonal to those of v.U̇.",
            )
        end
    end
    return nothing
end

"""
    Base.convert(::Type{Matrix{T}}, basis :: CachedBasis{𝔽,DefaultOrthonormalBasis{𝔽, TangentSpaceType},HOSVDBasis{T, D}}) where {𝔽, T, D}
    Base.convert(::Type{Matrix}, basis :: CachedBasis{𝔽,DefaultOrthonormalBasis{𝔽, TangentSpaceType},HOSVDBasis{T, D}}) where {𝔽, T, D}

Convert a HOSVD basis to a matrix whose columns are the vectorisations of the basis vectors.
"""
function Base.convert(::Type{Matrix{T}}, basis::CachedHOSVDBasis{𝔽,T,D}) where {𝔽,T,D}
    𝔄 = basis.data.point
    U⊥ = basis.data.U⊥
    U = 𝔄.hosvd.U
    σ = 𝔄.hosvd.σ
    ℭ = 𝔄.hosvd.core
    r⃗ = size(ℭ)
    n⃗ = size(𝔄)

    J = Matrix{T}(undef, prod(n⃗), manifold_dimension(Tucker(n⃗, r⃗)))
    # compute all possible ∂𝔄╱∂ℭ
    J[:, 1:prod(r⃗)] = ⊗ᴿ(U...)
    # compute all possible ∂𝔄╱∂U[d] for d = 1,...,D
    nextcolumn = prod(r⃗)
    for d in 1:D
        Udᵀ𝔄⁽ᵈ⁾::Matrix{T} = unfold(ℭ, d) * ⊗ᴿ(U[1:(d - 1)]..., U[(d + 1):end]...)'
        for i in 1:size(U⊥[d], 2), j in 1:r⃗[d]
            ∂𝔄ᵢⱼ⁽ᵈ⁾ = 1 / σ[d][j] * U⊥[d][:, i] * Udᵀ𝔄⁽ᵈ⁾[j, :]'
            ∂𝔄ᵢⱼ = fold(∂𝔄ᵢⱼ⁽ᵈ⁾, d, n⃗)
            J[:, nextcolumn += 1] = vec(∂𝔄ᵢⱼ)
        end
    end
    return J
end
function Base.convert(::Type{Matrix}, basis::CachedHOSVDBasis{𝔽,T,D}) where {𝔽,T,D}
    return convert(Matrix{T}, basis)
end

"""
    Base.convert(::Type{Array}, A :: TuckerPoint) where {T, TA <: T, D}
    Base.convert(::Type{Array{T,D}}, A :: TuckerPoint{TA, D}) where {T, TA <: T, D}

Convert a point on the Tucker manifold to a full tensor.

    Base.convert(::Type{Array}, A :: TuckerPoint, X)
    Base.convert(::Type{Array{T,D}}, A :: TuckerPoint{TA, D}, X :: TuckerTVector) where {T, TA <: T, D}

Convert a tangent vector X to the Tucker manifold at a point A to full tensor.
"""
function Base.convert(::Type{Array{T,D}}, 𝔄::TuckerPoint{TA,D}) where {T,TA<:T,D}
    return reshape(⊗ᴿ(𝔄.hosvd.U...) * vec(𝔄.hosvd.core), size(𝔄))
end
Base.convert(::Type{Array}, 𝔄::TuckerPoint{T,D}) where {T,D} = convert(Array{T,D}, 𝔄)
function Base.convert(
    ::Type{Array{T,D}}, 𝔄::TuckerPoint{TA,D}, X::TuckerTVector
) where {T,TA<:T,D}
    X_ambient = ⊗ᴿ(𝔄.hosvd.U...) * vec(X.Ċ)
    for d in 1:D
        # TODO: a lot of products between factor matrices and unfoldings of the core
        # will be recomputed
        X_ambient +=
            ⊗ᴿ(ntuple(d_ -> d_ == d ? X.U̇[d_] : 𝔄.hosvd.U[d_], D)...) * vec(𝔄.hosvd.core)
    end
    return reshape(X_ambient, size(𝔄))
end
function Base.convert(::Type{Array}, 𝔄::TuckerPoint{T,D}, X::TuckerTVector) where {T,D}
    return convert(Array{T,D}, 𝔄, X)
end

Base.copy(x::TuckerTVector) = TuckerTVector(copy(x.Ċ), map(copy, x.U̇))

function Base.copyto!(q::TuckerPoint, p::TuckerPoint)
    for d in 1:ndims(q)
        copyto!(q.hosvd.U[d], p.hosvd.U[d])
        copyto!(q.hosvd.σ[d], p.hosvd.σ[d])
    end
    copyto!(q.hosvd.core, p.hosvd.core)
    return q
end
function Base.copyto!(y::TuckerTVector, x::TuckerTVector)
    for d in 1:ndims(y.Ċ)
        copyto!(y.U̇[d], x.U̇[d])
    end
    copyto!(y.Ċ, x.Ċ)
    return y
end

#=
Inverse of the k'th unfolding of a size n₁ × ... × n_D tensor
=#
function fold(𝔄♭::AbstractMatrix{T}, k, n⃗::NTuple{D,Int})::Array{T,D} where {T,D,Int}
    @assert 1 ≤ k ≤ D
    @assert size(𝔄♭, 1) == n⃗[k]

    # (compiler doesn't know we are reshaping back into order D array without type assertion)
    size_pre_permute::NTuple{D,Int} = (n⃗[k], n⃗[1:(k - 1)]..., n⃗[(k + 1):D]...)
    perm::NTuple{D,Int} = ((2:k)..., 1, ((k + 1):D)...)
    return permutedims(reshape(𝔄♭, size_pre_permute), perm)
end

@doc raw"""
    get_basis(:: Tucker, A :: TuckerPoint, basisType::DefaultOrthonormalBasis{𝔽, TangentSpaceType}) where 𝔽

A implicitly stored basis of the tangent space to the Tucker manifold.
Assume $\mathcal{A} = (U_1,\dots,U_D) \cdot \mathcal{C}$ is in HOSVD format and that, for
$d=1,\dots,D$, the singular values of the
$d$'th unfolding are $\sigma_{dj}$, with $j = 1,\dots,R_d$.
The basis of the tangent space is as follows: [^Dewaele2021]

```math
\bigl\{
    (U_1,\dots,U_D) e_i
\bigr\} \cup \bigl\{
    (U_1,\dots, \sigma_{dj}^{-1} U_d^{\perp} e_i e_j^T,\dots,U_D) \cdot \mathcal{C}
\bigr\}
```

in which $U_d^\perp$ is such that $[U_d \quad U_d^{\perp}]$ forms an orthonormal basis
of $\mathbb{R}^{N_d}$, for each $d = 1,\dots,D$.

[^Dewaele2021]:
    > Nick Dewaele, Paul Breiding, Nick Vannieuwenhoven, "The condition number of many tensor decompositions is invariant under Tucker compression"
    > arxiv: #TODO
"""
function get_basis(
    ::Tucker,
    𝔄::TuckerPoint,
    basisType::DefaultOrthonormalBasis{𝔽,TangentSpaceType}=DefaultOrthonormalBasis(),
) where {𝔽}
    D = ndims(𝔄)
    n⃗ = size(𝔄)
    r⃗ = size(𝔄.hosvd.core)

    U = 𝔄.hosvd.U
    U⊥ = ntuple(d -> Matrix(qr(I - U[d] * U[d]', Val(true)).Q)[:, 1:(n⃗[d] - r⃗[d])], D)

    basis = HOSVDBasis(𝔄, U⊥)
    return CachedBasis(basisType, basis)
end

"""
    get_coordinates(::Tucker, A, X, b)

The coordinates of a tangent vector X at point A on the Tucker manifold with respect to the
basis b.
"""
function get_coordinates(::Tucker, 𝔄, X, ℬ::CachedHOSVDBasis)
    coords = vec(X.Ċ)
    for d in 1:length(X.U̇)
        coord_mtx = (ℬ.data.U⊥[d] \ X.U̇[d]) * Diagonal(𝔄.hosvd.σ[d])
        coords = vcat(coords, vec(coord_mtx'))
    end
    return coords
end

"""
    get_vector(::Tucker, A, x, b)

The tangent vector at a point A whose coordinates with respect to the basis b are x.
"""
function get_vector(
    ::Tucker, 𝔄::TuckerPoint, ξ::AbstractVector{T}, ℬ::CachedHOSVDBasis
) where {T}
    U = 𝔄.hosvd.U
    ℭ = 𝔄.hosvd.core
    σ = 𝔄.hosvd.σ
    U⊥ = ℬ.data.U⊥
    D = ndims(ℭ)
    r⃗ = size(ℭ)
    n⃗ = size(𝔄)

    # split ξ into ξ_core and ξU so that vcat(ξ_core, ξU...) == ξ
    ξ_core = ξ[1:length(ℭ)]
    ξU = Vector{T}[]
    nextcolumn = length(ℭ) + 1
    for d in 1:D
        numcols = r⃗[d] * (n⃗[d] - r⃗[d])
        push!(ξU, ξ[nextcolumn:(nextcolumn + numcols - 1)])
        nextcolumn += numcols
    end

    # Construct ∂U[d] by plugging in the definition of
    #    our orthonormal basis:
    # V[d] = ∂U[d] = ∑ᵢⱼ { ξ[d]ᵢⱼ (σ[d]ⱼ)⁻¹ U⊥[d] 𝐞ᵢ 𝐞ⱼᵀ }
    #      = ∑ⱼ (σ[d]ⱼ)⁻¹ U⊥[d] ( ∑ᵢ ξ[d]ᵢⱼ  𝐞ᵢ) 𝐞ⱼᵀ
    ∂U = similar.(U)
    for d in 1:D
        # Assuming ξ = [ξ₁₁, ..., ξ₁ⱼ, ..., ξᵢ₁, ..., ξᵢⱼ, ..., ], we can
        # reshape ξU[d] into a matrix with row indices i and column indices j
        grid = transpose(reshape(ξU[d], r⃗[d], n⃗[d] - r⃗[d]))
        # Notice that ∑ᵢ ξᵈᵢⱼ𝐞ᵢ = grid[:,j].
        # This means V[d] = U⊥[d] * grid * Diagonal(σ[d])⁻¹
        ∂U[d][:, :] = U⊥[d] * grid * Diagonal(1 ./ σ[d])
    end

    ∂C = reshape(ξ_core, size(ℭ))
    return TuckerTVector(∂C, ∂U)
end

function get_vectors(ℳ::Tucker, 𝔄::TuckerPoint, ℬ::CachedHOSVDBasis)
    return collect(iterate_vectors(ℳ, 𝔄, ℬ))
end

"""
    inner(::Tucker, A::TuckerPoint, x::TuckerTVector, y::TuckerTVector)

The Euclidean inner product between tangent vectors `x` and `y` at the point `A` on
the Tucker manifold.function

    inner(::Tucker, A::TuckerPoint, x::TuckerTVector, y)
    inner(::Tucker, A::TuckerPoint, x, y::TuckerTVector)

The Euclidean inner product between `x` and `y` where `x` is a vector tangent to the Tucker
manifold at `A` and `y` is a vector or the ambient space or vice versa. The vector in the
ambient space is represented as a full tensor.
"""
function inner(::Tucker, 𝔄::TuckerPoint, x::TuckerTVector, y::TuckerTVector)
    ℭ = 𝔄.hosvd.core
    dotprod = dot(x.Ċ, y.Ċ)
    for d in 1:ndims(𝔄)
        dotprod += dot(x.U̇[d] * unfold(ℭ, d), y.U̇[d] * unfold(ℭ, d))
    end
    return dotprod
end
inner(::Tucker, 𝔄::TuckerPoint, x::TuckerTVector, y) = dot(convert(Array, 𝔄, x), y)
inner(::Tucker, 𝔄::TuckerPoint, x, y::TuckerTVector) = dot(x, convert(Array, 𝔄, y))

"""
    inverse_retract(ℳ::Tucker, A::TuckerPoint, B::TuckerPoint, r::ProjectionInverseRetraction)

The projection inverse retraction on the Tucker manifold interprets `B` as a point in the
ambient Euclidean space and projects it onto the tangent space at to `ℳ` at `A`.
"""
function inverse_retract(
    ℳ::Tucker, 𝔄::TuckerPoint, 𝔅::TuckerPoint, r::ProjectionInverseRetraction
)
    # default allocate_result implementation gives the wrong answer
    return inverse_retract!(ℳ, zero_vector(ℳ, 𝔄), 𝔄, 𝔅, r)
end
function inverse_retract!(
    ℳ::Tucker, X, 𝔄::TuckerPoint, 𝔅::TuckerPoint, ::ProjectionInverseRetraction
)
    diffVector = convert(Array, 𝔅) - convert(Array, 𝔄)
    return project!(ℳ, X, 𝔄, diffVector)
end

function isapprox(p::TuckerPoint, q::TuckerPoint; kwargs...)
    return isapprox(convert(Array, p), convert(Array, q); kwargs...)
end
isapprox(::Tucker, p::TuckerPoint, q::TuckerPoint; kwargs...) = isapprox(p, q; kwargs...)
function isapprox(::Tucker, p::TuckerPoint, x::TuckerTVector, y::TuckerTVector; kwargs...)
    return isapprox(convert(Array, p, x), convert(Array, p, y); kwargs...)
end

"""
    isValidTuckerRank(n⃗, r⃗)

Determines whether there are tensors of dimensions n⃗ with multilinear rank r⃗
"""
function isValidTuckerRank(n⃗, r⃗)
    return all(r⃗ .≤ n⃗) && all(ntuple(i -> r⃗[i] ≤ prod(r⃗) ÷ r⃗[i], length(r⃗)))
end

function iterate_vectors(ℳ::Tucker, 𝔄::TuckerPoint{T,D}, ℬ::CachedHOSVDBasis) where {T,D}
    # TODO: This is a lazy implementation. This is very similar to convert(Matrix, ℬ).
    dimℳ = manifold_dimension(ℳ)
    coords = zeros(T, dimℳ)

    function iᵗʰvector(i)
        coords[i] = 1
        vector = get_vector(ℳ, 𝔄, coords, ℬ)
        coords[i] = 0
        return vector
    end
    iterator = (iᵗʰvector(i) for i in 1:dimℳ)
    return iterator
end

@doc raw"""
    manifold_dimension(::Tucker)

The dimension of the manifold of $N_1 \times \dots \times N_D$ tensors of multilinear
rank $R_1 \times \dots \times R_D$, i.e.
```math
    \mathrm{dim}(\mathcal{M}) = \prod_{d=1}^D R_d + \sum_{d=1}^D R_d (N_d - R_d).
```
"""
manifold_dimension(::Tucker{n⃗,r⃗}) where {n⃗,r⃗} = prod(r⃗) + sum(r⃗ .* (n⃗ .- r⃗))

@doc raw"""
    Base.ndims(:: TuckerPoint{T, D})

The order of a tensor of low multilinear rank
"""
Base.ndims(::TuckerPoint{T,D}) where {T,D} = D

number_eltype(::TuckerPoint{T,D}) where {T,D} = T
number_eltype(::TuckerTVector{T,D}) where {T,D} = T

"""
    project(ℳ::Tucker, 𝔄::TuckerPoint, X)

The least-squares projection of a tensor `X` to the tangent space to `ℳ` at `A`.
"""
project(ℳ::Tucker, 𝔄::TuckerPoint, X) = project!(ℳ, zero_vector(ℳ, 𝔄), 𝔄, X)
#Default implementation of project does allocate_result with the wrong type
function project!(ℳ::Tucker, Y, 𝔄::TuckerPoint, X)
    ℬ = get_basis(ℳ, 𝔄, DefaultOrthonormalBasis())
    coords = [inner(ℳ, 𝔄, ℬᵢ, X) for ℬᵢ in iterate_vectors(ℳ, 𝔄, ℬ)]
    copyto!(Y, get_vector(ℳ, 𝔄, coords, ℬ))
    return Y
end

representation_size(::Tucker{N}) where {N} = N

@doc raw"""
    retract(::Tucker, A, x, ::HOSVDRetraction)

The truncated HOSVD-based retraction [^Kressner2014] to the Tucker manifold, i.e.
$R_{\mathcal{A}}(x)$ is the sequentially tuncated HOSVD of $\mathcal{A} + x$

[^Kressner2014]:
    > Daniel Kressner, Michael Steinlechner, Bart Vandereycken: "Low-rank tensor completion by Riemannian optimization"
    > BIT Numerical Mathematics, 54(2), pp. 447-468, 2014
    > doi: [10.1007/s10543-013-0455-z](https://doi.org/10.1007/s10543-013-0455-z)

"""
retract(::Tucker, ::Any, ::Any, ::HOSVDRetraction)

function retract!(
    ::Tucker, q::TuckerPoint, p::TuckerPoint{T,D}, x::TuckerTVector, ::HOSVDRetraction
) where {T,D}
    U = p.hosvd.U
    V = x.U̇
    ℭ = p.hosvd.core
    𝔊 = x.Ċ
    r⃗ = size(ℭ)

    # Build the core tensor S and the factors [Uᵈ  Vᵈ]
    S = zeros(T, 2 .* size(ℭ))
    S[CartesianIndices(ℭ)] = ℭ + 𝔊
    UQ = Matrix{T}[]
    for d in 1:D
        # We make the following adaptation to Kressner2014:
        # Fix the i'th term of the sum and replace Vᵢ by Qᵢ Rᵢ.
        # We can absorb the R factor into the core by replacing Vᵢ by Qᵢ
        # and C (in the i'th term of the sum) by C ×ᵢ Rᵢ
        Q, R = qr(V[d])
        idxOffset = CartesianIndex(ntuple(i -> i == d ? r⃗[d] : 0, D))
        ℭ_transf = fold(R * unfold(ℭ, d), d, size(ℭ))
        S[CartesianIndices(ℭ) .+ idxOffset] = ℭ_transf
        push!(UQ, hcat(U[d], Matrix(Q)))
    end

    #Convert to truncated HOSVD of p + x
    hosvd_S = st_hosvd(S, r⃗)
    factors = UQ .* hosvd_S.U
    for i in 1:D
        q.hosvd.U[i] .= factors[i]
        q.hosvd.σ[i] .= hosvd_S.σ[i]
    end
    q.hosvd.core .= hosvd_S.core
    return q
end

function Base.show(io::IO, ::MIME"text/plain", 𝒯::Tucker{N,R,D,𝔽}) where {N,R,D,𝔽}
    return print(io, "Tucker(", N, ", ", R, ", ", 𝔽, ")")
end
function Base.show(io::IO, ::MIME"text/plain", 𝔄::TuckerPoint)
    pre = " "
    summary(io, 𝔄)
    for d in eachindex(𝔄.hosvd.U)
        println(io, string("\nU factor ", d, ":"))
        su = sprint(show, "text/plain", 𝔄.hosvd.U[d]; context=io, sizehint=0)
        su = replace(su, '\n' => "\n$(pre)")
        println(io, pre, su)
    end
    println(io, "\nCore :")
    su = sprint(show, "text/plain", 𝔄.hosvd.core; context=io, sizehint=0)
    su = replace(su, '\n' => "\n$(pre)")
    return print(io, pre, su)
end
function Base.show(io::IO, ::MIME"text/plain", x::TuckerTVector)
    pre = " "
    summary(io, x)
    for d in eachindex(x.U̇)
        println(io, string("\nU̇ factor ", d, ":"))
        su = sprint(show, "text/plain", x.U̇[d]; context=io, sizehint=0)
        su = replace(su, '\n' => "\n$(pre)")
        println(io, pre, su)
    end
    println(io, "\nĊ factor :")
    su = sprint(show, "text/plain", x.Ċ; context=io, sizehint=0)
    su = replace(su, '\n' => "\n$(pre)")
    return print(io, pre, su)
end
function Base.show(io::IO, mime::MIME"text/plain", ℬ::CachedHOSVDBasis{𝔽,T,D}) where {𝔽,T,D}
    summary(io, ℬ)
    print(" ≅")
    su = sprint(show, "text/plain", convert(Matrix{T}, ℬ); context=io, sizehint=0)
    su = replace(su, '\n' => "\n ")
    return println(io, " ", su)
end

"""
    Base.size(::TuckerPoint)

The dimensions of a tensor of low multilinear rank
"""
Base.size(𝔄::TuckerPoint) = map(u -> size(u, 1), 𝔄.hosvd.U)

#=
    st_hosvd(𝔄, mlrank=size(𝔄))

This is the HOSVD of an approximation of 𝔄, i.e. the core of this decomposition
is also in HOSVD format.
=#
function st_hosvd(𝔄, mlrank=size(𝔄))
    T = eltype(𝔄)
    D = ndims(𝔄)
    n⃗ = size(𝔄)
    # Add type assertions to U and σ for type stability
    U::NTuple{D,Matrix{T}} = ntuple(d -> Matrix{T}(undef, n⃗[d], mlrank[d]), D)
    σ::NTuple{D,Vector{T}} = ntuple(d -> Vector{T}(undef, mlrank[d]), D)

    for d in 1:D
        # unfold
        r_d = mlrank[d]
        𝔄⁽ᵈ⁾ = unfold(𝔄, d)
        # truncated SVD + incremental construction of the core
        UΣVᵀ = svd(𝔄⁽ᵈ⁾)
        U[d] .= UΣVᵀ.U[:, 1:r_d]
        σ[d] .= UΣVᵀ.S[1:r_d]
        𝔄⁽ᵈ⁾ = Diagonal(σ[d]) * UΣVᵀ.Vt[1:r_d, :]
        # reshape back into a tensor (compiler doesn't know we are reshaping back into
        # an order D array without type assertion)
        m⃗::NTuple{D,Int} = tuple(mlrank[1:d]..., n⃗[(d + 1):D]...)
        𝔄 = fold(𝔄⁽ᵈ⁾, d, m⃗)
    end

    # Make sure the truncated core is in "all-orthogonal" HOSVD format
    if mlrank ≠ n⃗
        hosvd_core = st_hosvd(𝔄, mlrank)
        U = U .* hosvd_core.U
        𝔄 = hosvd_core.core
        σ = hosvd_core.σ
    end

    return HOSVD{T,D}(U, 𝔄, σ)
end

#=
	unfold(𝔄, k)

Mode-k unfolding of the array 𝔄 of order d ≥ k
=#
function unfold(𝔄, k)
    d = ndims(𝔄)
    𝔄_ = permutedims(𝔄, vcat(k, 1:(k - 1), (k + 1):d))
    return reshape(𝔄_, size(𝔄, k), div(length(𝔄), size(𝔄, k)))
end

@doc raw"""
    zero_vector(::Tucker, A::TuckerPoint)

The zero element in the tangent space to A on the Tucker manifold
"""
zero_vector(::Tucker, 𝔄::TuckerPoint) = TuckerTVector(zero(𝔄.hosvd.core), zero.(𝔄.hosvd.U))

function zero_vector!(::Tucker, X::TuckerTVector, ::TuckerPoint)
    for U̇ in X.U̇
        fill!(U̇, zero(eltype(U̇)))
    end
    fill!(X.Ċ, zero(eltype(Ċ)))
    return X
end
