
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
    @assert is_valid_mlrank(n⃗, r⃗)
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
    core::AbstractArray{T,D},
    factors::Vararg{MtxT,D},
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
    @assert is_valid_mlrank(size(A), mlrank)
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

# An implicitly stored basis of the tangent space to the Tucker manifold. This is the basis
# from [Dewaele2021] and acts as the default orthonormal basis.
struct HOSVDBasis{T,D}
    point::TuckerPoint{T,D}
    U⊥::NTuple{D,Matrix{T}}
end
CachedHOSVDBasis{𝔽,T,D} =
    CachedBasis{𝔽,DefaultOrthonormalBasis{𝔽,TangentSpaceType},HOSVDBasis{T,D}}

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
function allocate(p::TuckerPoint{Tp,D}, ::Type{T}) where {T,Tp,D}
    @assert promote_type(Tp, T) == T
    return TuckerPoint(
        HOSVD(allocate(p.hosvd.U, T), allocate(p.hosvd.core, T), allocate(p.hosvd.σ, T)),
    )
end
allocate(x::TuckerTVector) = allocate(x, number_eltype(x))
function allocate(x::TuckerTVector, ::Type{T}) where {T}
    return TuckerTVector(allocate(x.Ċ, T), allocate(x.U̇, T))
end

# Tuple-like broadcasting of TuckerTVector
Base.axes(::TuckerTVector) = ()

function Broadcast.BroadcastStyle(::Type{TuckerTVector{T,D}}) where {T,D}
    return Broadcast.Style{TuckerTVector{Any,D}}()
end
function Broadcast.BroadcastStyle(
    ::Broadcast.AbstractArrayStyle{0},
    b::Broadcast.Style{<:TuckerTVector},
)
    return b
end

function Broadcast.instantiate(
    bc::Broadcast.Broadcasted{Broadcast.Style{TuckerTVector{Any,D}},Nothing},
) where {D}
    return bc
end
function Broadcast.instantiate(
    bc::Broadcast.Broadcasted{Broadcast.Style{TuckerTVector{Any,D}}},
) where {D}
    Broadcast.check_broadcast_axes(bc.axes, bc.args...)
    return bc
end

Broadcast.broadcastable(v::TuckerTVector) = v

Base.@propagate_inbounds function Broadcast._broadcast_getindex(
    v::TuckerTVector,
    ::Val{I},
) where {I}
    if I isa Symbol
        return getfield(v, I)
    else
        return getfield(v, I[1])[I[2]]
    end
end

####

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
    x_buffer = similar(x)
    for d in 1:ndims(x)
        r = rank(tensor_unfold!(x_buffer, x, d); kwargs...)
        r == R[d] || return DomainError(size(x), s * "since its rank is not $(R).")
    end
    return nothing
end
function check_point(M::Tucker{N,R,D}, x::TuckerPoint; kwargs...) where {N,R,D}
    s = "The point $(x) does not lie on $(M), "
    U = x.hosvd.U
    ℭ = x.hosvd.core
    if size(ℭ) ≠ R
        return DomainError(
            size(x.hosvd.core),
            s * "since the size of the core is not $(R).",
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
    ℭ_buffer = similar(ℭ)
    for d in 1:ndims(x.hosvd.core)
        ℭ⁽ᵈ⁾ = tensor_unfold!(ℭ_buffer, ℭ, d)
        gram = ℭ⁽ᵈ⁾ * ℭ⁽ᵈ⁾'
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
    M::Tucker{N,R,D},
    p::TuckerPoint{T,D},
    v::TuckerTVector,
) where {N,R,T,D}
    s = "The tangent vector $(v) is not a tangent vector to $(p) on $(M), "
    if size(p.hosvd.core) ≠ size(v.Ċ) || any(size.(v.U̇) .≠ size.(p.hosvd.U))
        return DomainError(
            size(v.Ċ),
            s * "since the array dimensons of $(p) and $(v)" * "do not agree.",
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
function Base.convert(::Type{Matrix{T}}, ℬ::CachedHOSVDBasis{𝔽,T,D}) where {𝔽,T,D}
    𝔄 = ℬ.data.point
    r⃗ = size(𝔄.hosvd.core)
    n⃗ = size(𝔄)
    ℳ = Tucker(n⃗, r⃗)

    J = Matrix{T}(undef, prod(n⃗), manifold_dimension(ℳ))
    # compute all possible ∂𝔄╱∂ℭ (in one go is quicker than one vector at a time)
    J[:, 1:prod(r⃗)] = ⊗ᴿ(𝔄.hosvd.U...)
    # compute all possible ∂𝔄╱∂U[d] for d = 1,...,D
    function fill_column!(i, vᵢ)
        Jᵢ_tensor = reshape(view(J, :, i), n⃗) # changes to this apply to J as well
        return embed!(ℳ, Jᵢ_tensor, 𝔄, vᵢ)
    end
    foreach(fill_column!, ℳ, 𝔄, ℬ, (prod(r⃗) + 1):manifold_dimension(ℳ))
    return J
end
function Base.convert(::Type{Matrix}, basis::CachedHOSVDBasis{𝔽,T,D}) where {𝔽,T,D}
    return convert(Matrix{T}, basis)
end

@inline function Base.copy(
    bc::Broadcast.Broadcasted{Broadcast.Style{TuckerTVector{Any,D}}},
) where {D}
    return TuckerTVector(
        @inbounds(Broadcast._broadcast_getindex(bc, Val(:Ċ))),
        ntuple(i -> @inbounds(Broadcast._broadcast_getindex(bc, Val((:U̇, i)))), Val(D)),
    )
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
@inline function Base.copyto!(
    dest::TuckerTVector,
    bc::Broadcast.Broadcasted{Broadcast.Style{TuckerTVector{Any,D}}},
) where {D}
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{TuckerTVector} # only a single input argument to broadcast!
        A = bc.args[1]
        return copyto!(dest, A)
    end
    bc′ = Broadcast.preprocess(dest, bc)
    copyto!(dest.Ċ, Broadcast._broadcast_getindex(bc′, Val(:Ċ)))
    for i in 1:D
        copyto!(dest.U̇[i], Broadcast._broadcast_getindex(bc, Val((:U̇, i))))
    end
    return dest
end

@doc raw"""
embed(::Tucker, A :: TuckerPoint)

Convert a point `A` on the Tucker manifold to a full tensor, represented as an
$N_1 \times \dots \times N_D$-array.

embed(::Tucker, A::TuckerPoint, X::TuckerTVector)

Convert a tangent vector `X` with base point `A` on the Tucker manifold to a full tensor,
represented as an $N_1 \times \dots \times N_D$-array.
"""
embed(::Tucker, ::Any, ::TuckerPoint)

function embed!(::Tucker, q, p::TuckerPoint)
    return copyto!(q, reshape(⊗ᴿ(p.hosvd.U...) * vec(p.hosvd.core), size(p)))
end
function embed!(ℳ::Tucker, Y, 𝔄::TuckerPoint{T,D}, X::TuckerTVector) where {T,D}
    mul!(vec(Y), ⊗ᴿ(𝔄.hosvd.U...), vec(X.Ċ))
    𝔄_embedded = embed(ℳ, 𝔄)
    buffer = similar(𝔄_embedded)
    for k in 1:D
        U̇ₖUₖᵀ𝔄₍ₖ₎ = X.U̇[k] * (𝔄.hosvd.U[k]' * tensor_unfold!(buffer, 𝔄_embedded, k))
        Y .= Y + tensor_fold!(buffer, U̇ₖUₖᵀ𝔄₍ₖ₎, k)
    end
    return Y
end

@doc raw"""
Base.foreach(f, M::Tucker, p::TuckerPoint, basis::AbstractBasis, indices=1:manifold_dimension(M))

Let `basis` be and [`AbstractBasis`](@ref) at a point `p` on `M`. Suppose `f` is a function
that takes an index and a vector as an argument.
This function applies `f` to `i` and the `i`th basis vector sequentially for each `i` in
`indices`.
Using a [`CachedBasis`](@ref) may speed up the computation.

**NOTE**: The i'th basis vector is overwritten in each iteration. If any information about
the vector is to be stored, `f` must make a copy.
"""
function Base.foreach(
    f,
    M::Tucker,
    p::TuckerPoint,
    basis::AbstractBasis,
    indices=1:manifold_dimension(M),
)
    # Use mutating variants to avoid superfluous allocation
    bᵢ = zero_vector(M, p)
    eᵢ = zeros(number_eltype(p), manifold_dimension(M))
    for i in indices
        eᵢ[i] = one(eltype(eᵢ))
        get_vector!(M, bᵢ, p, eᵢ, basis)
        eᵢ[i] = zero(eltype(eᵢ))
        f(i, bᵢ)
    end
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
> arxiv: [2106.13034](https://arxiv.org/abs/2106.13034)
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

#=
get_coordinates(::Tucker, A, X :: TuckerTVector, b)

The coordinates of a tangent vector X at point A on the Tucker manifold with respect to the
basis b.
=#
function get_coordinates(::Tucker, 𝔄, X::TuckerTVector, ℬ::CachedHOSVDBasis)
    coords = vec(X.Ċ)
    for d in 1:length(X.U̇)
        coord_mtx = (ℬ.data.U⊥[d] \ X.U̇[d]) * Diagonal(𝔄.hosvd.σ[d])
        coords = vcat(coords, vec(coord_mtx'))
    end
    return coords
end
function get_coordinates(
    M::Tucker,
    𝔄,
    X,
    ℬ::DefaultOrthonormalBasis{𝔽,TangentSpaceType},
) where {𝔽}
    return get_coordinates(M, 𝔄, X, get_basis(M, 𝔄, ℬ))
end

#=
get_vector(::Tucker, A, x, b)

The tangent vector at a point A whose coordinates with respect to the basis b are x.
=#
function get_vector!(
    ::Tucker,
    y,
    𝔄::TuckerPoint,
    x::AbstractVector{T},
    ℬ::CachedHOSVDBasis,
) where {T}
    ξ = convert(Vector{promote_type(number_eltype(𝔄), eltype(x))}, x)
    ℭ = 𝔄.hosvd.core
    σ = 𝔄.hosvd.σ
    U⊥ = ℬ.data.U⊥
    D = ndims(ℭ)
    r⃗ = size(ℭ)
    n⃗ = size(𝔄)

    # split ξ into ξ_core and ξU so that vcat(ξ_core, ξU...) == ξ, but avoid copying
    ξ_core = view(ξ, 1:length(ℭ))
    ξU = Vector{typeof(ξ_core)}(undef, D)
    nextcolumn = length(ℭ) + 1
    for d in 1:D
        numcols = r⃗[d] * (n⃗[d] - r⃗[d])
        ξU[d] = view(ξ, nextcolumn:(nextcolumn + numcols - 1))
        nextcolumn += numcols
    end

    # Construct ∂U[d] by plugging in the definition of the orthonormal basis [Dewaele2021]
    # ∂U[d] = ∑ᵢⱼ { ξU[d]ᵢⱼ (σ[d]ⱼ)⁻¹ U⊥[d] 𝐞ᵢ 𝐞ⱼᵀ }
    #       = U⊥[d] * ∑ⱼ (σ[d]ⱼ)⁻¹ (∑ᵢ ξU[d]ᵢⱼ  𝐞ᵢ) 𝐞ⱼᵀ
    # ξU[d] = [ξ₁₁, ..., ξ₁ⱼ, ..., ξᵢ₁, ..., ξᵢⱼ, ..., ]
    # => turn these i and j into matrix indices and do matrix operations
    for d in 1:D
        grid = transpose(reshape(ξU[d], r⃗[d], n⃗[d] - r⃗[d]))
        mul!(y.U̇[d], U⊥[d], grid * Diagonal(1 ./ σ[d]))
    end

    y.Ċ .= reshape(ξ_core, size(y.Ċ))
    return y
end
function get_vector!(
    ℳ::Tucker,
    y,
    𝔄::TuckerPoint,
    x,
    ℬ::DefaultOrthonormalBasis{𝔽,TangentSpaceType},
) where {𝔽}
    return get_vector!(ℳ, y, 𝔄, x, get_basis(ℳ, 𝔄, ℬ))
end

function get_vectors(ℳ::Tucker, 𝔄::TuckerPoint{T,D}, ℬ::CachedHOSVDBasis) where {T,D}
    vectors = Vector{TuckerTVector{T,D}}(undef, manifold_dimension(ℳ))
    foreach((i, vᵢ) -> setindex!(vectors, copy(vᵢ), i), ℳ, 𝔄, ℬ)
    return vectors
end
function get_vectors(ℳ::Tucker, 𝔄::TuckerPoint, ℬ::DefaultOrthonormalBasis)
    return get_vectors(ℳ, 𝔄, get_basis(ℳ, 𝔄, ℬ))
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
    ℭ_buffer = similar(ℭ)
    for k in 1:ndims(𝔄)
        ℭ₍ₖ₎ = tensor_unfold!(ℭ_buffer, ℭ, k)
        dotprod += dot(x.U̇[k] * ℭ₍ₖ₎, y.U̇[k] * ℭ₍ₖ₎)
    end
    return dotprod
end
inner(M::Tucker, 𝔄::TuckerPoint, x::TuckerTVector, y) = dot(embed(M, 𝔄, x), y)
inner(M::Tucker, 𝔄::TuckerPoint, x, y::TuckerTVector) = dot(x, embed(M, 𝔄, y))

"""
inverse_retract(ℳ::Tucker, A::TuckerPoint, B::TuckerPoint, r::ProjectionInverseRetraction)

The projection inverse retraction on the Tucker manifold interprets `B` as a point in the
ambient Euclidean space and projects it onto the tangent space at to `ℳ` at `A`.
"""
inverse_retract(
    ::Tucker,
    ::Any,
    ::TuckerPoint,
    ::TuckerPoint,
    ::ProjectionInverseRetraction,
)

function inverse_retract!(
    ℳ::Tucker,
    X,
    𝔄::TuckerPoint,
    𝔅::TuckerPoint,
    ::ProjectionInverseRetraction,
)
    diffVector = embed(ℳ, 𝔅) - embed(ℳ, 𝔄)
    return project!(ℳ, X, 𝔄, diffVector)
end

function isapprox(p::TuckerPoint, q::TuckerPoint; kwargs...)
    ℳ = Tucker(size(p), size(p.hosvd.core))
    return isapprox(embed(ℳ, p), embed(ℳ, q); kwargs...)
end
isapprox(::Tucker, p::TuckerPoint, q::TuckerPoint; kwargs...) = isapprox(p, q; kwargs...)
function isapprox(M::Tucker, p::TuckerPoint, x::TuckerTVector, y::TuckerTVector; kwargs...)
    return isapprox(embed(M, p, x), embed(M, p, y); kwargs...)
end

#=
Determines whether there are tensors of dimensions n⃗ with multilinear rank r⃗
=#
function is_valid_mlrank(n⃗, r⃗)
    return all(r⃗ .≥ 1) &&
           all(r⃗ .≤ n⃗) &&
           all(ntuple(i -> r⃗[i] ≤ prod(r⃗) ÷ r⃗[i], length(r⃗)))
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
Base.ndims(A :: TuckerPoint{T, D})

The order of the tensor corresponding to the [`TuckerPoint`](@ref) `A`
"""
Base.ndims(::TuckerPoint{T,D}) where {T,D} = D

number_eltype(::TuckerPoint{T,D}) where {T,D} = T
number_eltype(::TuckerTVector{T,D}) where {T,D} = T

"""
project(ℳ::Tucker, 𝔄::TuckerPoint, X)

The least-squares projection of a tensor `X` to the tangent space to `ℳ` at `A`.
"""
project(::Tucker, ::Any, ::TuckerPoint, ::Any)

function project!(ℳ::Tucker, Y, 𝔄::TuckerPoint, X)
    ℬ = get_basis(ℳ, 𝔄, DefaultOrthonormalBasis())
    coords = Vector{number_eltype(𝔄)}(undef, manifold_dimension(ℳ))
    f!(i, ℬᵢ) = setindex!(coords, inner(ℳ, 𝔄, ℬᵢ, X), i)
    foreach(f!, ℳ, 𝔄, ℬ)
    return get_vector!(ℳ, Y, 𝔄, coords, ℬ)
end

@doc raw"""
retract(::Tucker, A, x, ::PolarRetraction)

The truncated HOSVD-based retraction [^Kressner2014] to the Tucker manifold, i.e.
$R_{\mathcal{A}}(x)$ is the sequentially tuncated HOSVD of $\mathcal{A} + x$

In the exceptional case that the multilinear rank of $\mathcal{A} + x$ is lower than A, this
retraction produces a boundary point.

[^Kressner2014]:
> Daniel Kressner, Michael Steinlechner, Bart Vandereycken: "Low-rank tensor completion by Riemannian optimization"
> BIT Numerical Mathematics, 54(2), pp. 447-468, 2014
> doi: [10.1007/s10543-013-0455-z](https://doi.org/10.1007/s10543-013-0455-z)

"""
retract(::Tucker, ::Any, ::Any, ::PolarRetraction)

function retract!(
    ::Tucker,
    q::TuckerPoint,
    p::TuckerPoint{T,D},
    x::TuckerTVector,
    ::PolarRetraction,
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
    buffer = similar(ℭ)
    for k in 1:D
        # We make the following adaptation to Kressner2014:
        # Fix the i'th term of the sum and replace Vᵢ by Qᵢ Rᵢ.
        # We can absorb the R factor into the core by replacing Vᵢ by Qᵢ
        # and C (in the i'th term of the sum) by C ×ᵢ Rᵢ
        Q, R = qr(V[k])
        idxOffset = CartesianIndex(ntuple(i -> i == k ? r⃗[k] : 0, D))
        ℭ⨉ₖR = tensor_fold!(buffer, R * tensor_unfold!(buffer, ℭ, k), k)
        S[CartesianIndices(ℭ) .+ idxOffset] = ℭ⨉ₖR
        push!(UQ, hcat(U[k], Matrix(Q)))
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
    println(io, "\nCore:")
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
    println(io, "\nĊ factor:")
    su = sprint(show, "text/plain", x.Ċ; context=io, sizehint=0)
    su = replace(su, '\n' => "\n$(pre)")
    return print(io, pre, su)
end
function Base.show(io::IO, ::MIME"text/plain", ℬ::CachedHOSVDBasis{𝔽,T,D}) where {𝔽,T,D}
    summary(io, ℬ)
    print(io, " ≅")
    su = sprint(show, "text/plain", convert(Matrix{T}, ℬ); context=io, sizehint=0)
    su = replace(su, '\n' => "\n ")
    return println(io, " ", su)
end

"""
Base.size(A::TuckerPoint)

The dimensions of a [`TuckerPoint`](@ref) `A`, when regarded as a full tensor
"""
Base.size(𝔄::TuckerPoint) = map(u -> size(u, 1), 𝔄.hosvd.U)

#=
Modification of the ST-HOSVD from [Vannieuwenhoven2012]
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
    # Initialise arrays to store successive truncations (𝔄′) and unfoldings (buffer)
    # so that the type remains constant at every truncation
    𝔄′ = reshape(view(𝔄, 1:length(𝔄)), n⃗)
    fold_buffer = reshape(view(similar(𝔄), 1:length(𝔄)), n⃗)
    unfold_buffer = view(similar(𝔄), 1:length(𝔄))

    for k in 1:D
        rₖ = mlrank[k]
        𝔄′₍ₖ₎ = tensor_unfold!(unfold_buffer, 𝔄′, k)
        # truncated SVD + incremental construction of the core
        UΣVᵀ = svd(𝔄′₍ₖ₎)
        U[k] .= UΣVᵀ.U[:, 1:rₖ]
        σ[k] .= UΣVᵀ.S[1:rₖ]
        𝔄′₍ₖ₎_trunc = Diagonal(σ[k]) * UΣVᵀ.Vt[1:rₖ, :]
        size𝔄′ = ntuple(i -> i ≤ k ? mlrank[i] : n⃗[i], D)
        fold_buffer = reshape(view(fold_buffer, 1:prod(size𝔄′)), size𝔄′)
        unfold_buffer = view(unfold_buffer, 1:prod(size𝔄′))
        𝔄′ = tensor_fold!(fold_buffer, 𝔄′₍ₖ₎_trunc, k)
    end
    core = Array(𝔄′)

    # Make sure the truncated core is in "all-orthogonal" HOSVD format
    if mlrank ≠ n⃗
        hosvd_core = st_hosvd(core, mlrank)
        U = U .* hosvd_core.U
        core = hosvd_core.core
        σ = hosvd_core.σ
    end

    return HOSVD{T,D}(U, core, σ)
end

# In-place inverse of the k'th unfolding of a size n₁ × ... × n_D tensor.
# The size of the reshaped tensor is determined by the size of 𝔄.
# The result is stored in 𝔄. The returned value uses the same address space as 𝔄.
function tensor_fold!(𝔄::AbstractArray{T,D}, 𝔄₍ₖ₎::AbstractMatrix{T}, k) where {T,D}
    @assert length(𝔄₍ₖ₎) == length(𝔄) && size(𝔄₍ₖ₎, 1) == size(𝔄, k)
    @assert pointer(𝔄) !== pointer(𝔄₍ₖ₎)
    # Caution: tuple operations can be type unstable if used incorrectly
    σ(i) = i == 1 ? k : i ≤ k ? i - 1 : i
    σ⁻¹(i) = i < k ? i + 1 : i == k ? 1 : i
    permuted_size = ntuple(i -> size(𝔄, σ(i)), D)
    return permutedims!(𝔄, reshape(𝔄₍ₖ₎, permuted_size), ntuple(σ⁻¹, D))
end

# In-place mode-k unfolding of the array 𝔄 of order D ≥ k.
# The argument buffer is an array of arbitrary dimensions of the same length as 𝔄.
# The returned value uses the same address space as the buffer.
function tensor_unfold!(buffer, 𝔄::AbstractArray{T,D}, k) where {T,D}
    @assert length(buffer) == length(𝔄)
    @assert pointer(𝔄) !== pointer(buffer)
    𝔄₍ₖ₎ = reshape(buffer, size(𝔄, k), :)
    # Caution: tuple operations can be type unstable if used incorrectly
    σ(i) = i == 1 ? k : i ≤ k ? i - 1 : i
    permuted_size = ntuple(i -> size(𝔄, σ(i)), D)
    permutedims!(reshape(𝔄₍ₖ₎, permuted_size), 𝔄, ntuple(σ, D))
    return 𝔄₍ₖ₎
end

@doc raw"""
zero_vector(::Tucker, A::TuckerPoint)

The zero element in the tangent space to A on the Tucker manifold, represented as a
[`TuckerTVector`](@ref)
"""
function zero_vector!(::Tucker, X::TuckerTVector, ::TuckerPoint)
    for U̇ in X.U̇
        fill!(U̇, zero(eltype(U̇)))
    end
    fill!(X.Ċ, zero(eltype(X.Ċ)))
    return X
end

# The standard implementation of allocate_result on vector-valued functions gives an element
# of the same type as the manifold point. We want a vector instead.
for fun in [:get_vector, :inverse_retract, :project, :zero_vector]
    @eval function ManifoldsBase.allocate_result(
        ::Tucker,
        ::typeof($(fun)),
        p::TuckerPoint,
        args...,
    )
        return TuckerTVector(allocate(p.hosvd.core), allocate(p.hosvd.U))
    end
end

function ManifoldsBase.allocate_result(::Tucker{N}, f::typeof(embed), p, args...) where {N}
    dims = N
    return Array{number_eltype(p),length(dims)}(undef, dims)
end
