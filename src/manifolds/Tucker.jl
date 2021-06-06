
@doc raw"""
Tucker{N, R, D, 𝔽} <: AbstractManifold{𝔽}

The manifold of $N_1 \times \dots \times N_D$ real-valued or complex-valued matrices of fixed multilinear rank
$(R_1, \dots, R_D).


[^Kressner2014]:
> Daniel Kressner, Michael Steinlechner, Bart Vandereycken: "Low-rank tensor completion by Riemannian optimization"
> BIT Numerical Mathematics, 54(2), pp. 447-468, 2014
> doi: [10.1007/s10543-013-0455-z](https://doi.org/10.1007/s10543-013-0455-z)
"""
struct Tucker{N, R, D, 𝔽} <: AbstractManifold{𝔽} end
function Tucker(n⃗ :: NTuple{D, Int}, r⃗ :: NTuple{D, Int}, field :: AbstractNumbers = ℝ) where D
    @assert isValidTuckerRank(n⃗, r⃗)
    Tucker{n⃗, r⃗, D, field}()
end

"""
    HOSVD{T, D}

Higher-order singular value decomposition of an order D tensor with eltype T 
fields: 
* U: singular vectors of the unfoldings
* core: core tensor
* σ : singular values of the unfoldings
"""
struct HOSVD{T, D}
    U    :: NTuple{D, Matrix{T}}
    core :: Array{T, D}
    σ    :: NTuple{D, Vector{T}}
end

struct HOSVDRetraction <: AbstractRetractionMethod end

"""
    TuckerPoint{T, D} 

An order D tensor of fixed multilinear rank and entries of type T.
"""
struct TuckerPoint{T, D} <: AbstractManifoldPoint
    hosvd :: HOSVD{T, D}
end
function TuckerPoint(core :: AbstractArray{T, D}, factors :: Vararg{MtxT, D}) where {T, D, MtxT <: AbstractMatrix{T}}
    # Take the QR decompositions of the factors and multiply the R factors into the core
    qrfacs  = qr.(factors)
    Q       = map(qrfac -> qrfac.Q, qrfacs)
    R       = map(qrfac -> qrfac.R, qrfacs)
    core′   = reshape(Kronecker.:⊗(reverse(R)...) * vec(core), size(core))
    
    # Convert to HOSVD format by taking the HOSVD of the core
    decomp   = st_hosvd(core′)
    factors′ = Q .* decomp.U
    TuckerPoint(HOSVD{T, D}(factors′, decomp.core, decomp.σ))
end
function TuckerPoint(A :: AbstractArray, mlrank :: NTuple{D, Int}) where {D}
    #TODO  
    error("Not implemented")
end

@doc raw"""
    TuckerTVectort{T, D} <: TVector

Tangent space to the Tucker manifold at `x = (U_1,\dots,U_D) ⋅ \mathcal{C}`. This vector is represented as
```math 
(U_1,\dots,U_D) \cdot \dot{\mathcal{C}} + \sum_{d=1}^D (U_1,\dots,U_{d-1},\dot{U}_d,U_{d+1},\dots,U_D) \cdot \mathcal{C}
````
where $\dot_{U}_d^\mathrm{H} U_d = 0$
"""
struct TuckerTVector{T, D} <: TVector
    Ċ :: Array{T, D}
    U̇ :: NTuple{D, Matrix{T}}
end

"""
    HOSVDBasis{T, D}

A implicitly stored basis of the tangent space to the Tucker manifold.
If 𝔄 = (U¹ ⊗ ... ⊗ Uᴰ) C is a HOSVD, then this basis is defined as follows:

ℬ = {(U¹ ⊗ ... ⊗ Uᴰ) eᵢ} ∪ {(U¹ ⊗ ... ⊗ 1/σ[d][j] Uᵈ⊥ eᵢ eⱼᵀ ⊗ ... ⊗ Uᴰ) C}

See also:
[^Dewaele2021]
> Nick Dewaele, Paul Breiding, Nick Vannieuwenhoven, "The condition number of many tensor decompositions is invariant under Tucker compression"
#TODO arXiv
"""
struct HOSVDBasis{T, D}
	point :: TuckerPoint{T, D}
    U⊥    :: NTuple{D, Matrix{T}}
end
CachedHOSVDBasis{𝔽, T, D} = CachedBasis{𝔽,DefaultOrthonormalBasis{𝔽, TangentSpaceType},HOSVDBasis{T, D}}

⊗ᴿ(a...) = Kronecker.:⊗(reverse(a)...)

Base.:*(s::Number, x::TuckerTVector) = TuckerTVector(s * x.Ċ, s .* x.U̇)
Base.:*(x::TuckerTVector, s::Number) = TuckerTVector(x.Ċ * s, x.U̇ .* s)
Base.:/(x::TuckerTVector, s::Number) = TuckerTVector(x.Ċ / s, x.U̇ ./ s)
Base.:\(s::Number, x::TuckerTVector) = TuckerTVector(s \ x.Ċ, s .\ x.U̇)
Base.:+(x::TuckerTVector, y::TuckerTVector) = TuckerTVector(x.Ċ + y.Ċ, x.U̇ .+ y.U̇)
Base.:-(x::TuckerTVector, y::TuckerTVector) = TuckerTVector(x.Ċ - y.Ċ, x.U̇ .- y.U̇)
Base.:-(x::TuckerTVector) = TuckerTVector(-x.Ċ, map(-, x.U̇))
Base.:+(x::TuckerTVector) = TuckerTVector(x.Ċ, x.U̇)
Base.:(==)(x :: TuckerTVector, y :: TuckerTVector) = (x.Ċ == y.Ċ) && (x.U̇ .== y.U̇)

allocate(p :: TuckerPoint) = allocate(p, number_eltype(p))
function allocate(p::TuckerPoint, ::Type{T}) where T
    # This is not necessarily a valid HOSVD it's not worth computing the HOSVD
    # just for allocation
    TuckerPoint(HOSVD(allocate(p.hosvd.U, T), allocate(p.hosvd.core, T), allocate(p.hosvd.σ, T)))
end
allocate(x :: TuckerTVector) = allocate(x, number_eltype(x))
function allocate(x::TuckerTVector, ::Type{T}) where T
    TuckerTVector(allocate(x.Ċ, T), allocate(x.U̇, T))
end

"""
An orthonormal basis for the tangent space to the Tucker manifold at a point 𝔄, represented as a matrix
"""
function Base.convert(::Type{Matrix{T}}, basis :: CachedHOSVDBasis{𝔽, T, D}) where {𝔽, T, D}
    𝔄    = basis.data.point
    U⊥   = basis.data.U⊥
    U    = 𝔄.hosvd.U
    σ    = 𝔄.hosvd.σ
    ℭ    = 𝔄.hosvd.core
    r⃗    = size(ℭ)
    n⃗    = size(𝔄)

    J = Matrix{T}(undef, prod(n⃗), manifold_dimension(Tucker(n⃗, r⃗)))
    # compute all possible ∂𝔄╱∂ℭ
    J[:, 1:prod(r⃗)] = ⊗ᴿ(U...)
    # compute all possible ∂𝔄╱∂U[d] for d = 1,...,D
    nextcolumn = prod(r⃗)
    for d = 1:D
        Udᵀ𝔄⁽ᵈ⁾ :: Matrix{T} = unfold(ℭ, d) * ⊗ᴿ(U[1:d-1]..., U[d+1:end]...)'
        for i = 1:size(U⊥[d], 2), j = 1:r⃗[d]
            ∂𝔄ᵢⱼ⁽ᵈ⁾ = 1/σ[d][j] * U⊥[d][:,i] * Udᵀ𝔄⁽ᵈ⁾[j,:]'
            ∂𝔄ᵢⱼ    = fold(∂𝔄ᵢⱼ⁽ᵈ⁾, d, n⃗)
            J[:,nextcolumn += 1] = vec(∂𝔄ᵢⱼ)
        end
    end
    J
end
Base.convert(::Type{Matrix}, basis :: CachedHOSVDBasis{𝔽, T, D}) where {𝔽, T, D} = convert(Matrix{T}, basis)

function Base.convert(::Type{Array{T,D}}, 𝔄 :: TuckerPoint{TA, D}) where {T, TA <: T, D}
    reshape(⊗ᴿ(𝔄.hosvd.U...) * vec(𝔄.hosvd.core), size(𝔄))
end
Base.convert(::Type{Array}, 𝔄 :: TuckerPoint{T, D}) where {T,D} = convert(Array{T,D}, 𝔄)
function Base.convert(::Type{Array{T,D}}, 𝔄 :: TuckerPoint{TA, D}, X :: TuckerTVector) where {T, TA <: T, D}
    X_ambient = ⊗ᴿ(𝔄.hosvd.U...) * vec(X.Ċ)
    for d = 1:D
        X_ambient += ⊗ᴿ(ntuple(d_ -> d_ == d ? X.U̇[d_] : 𝔄.hosvd.U[d_], D)...) * vec(𝔄.hosvd.core)
    end
    reshape(X_ambient, size(𝔄))
end
function Base.convert(::Type{Array}, 𝔄 :: TuckerPoint{T, D}, X :: TuckerTVector) where {T,D}
    convert(Array{T,D}, 𝔄, X)
end

Base.copy(x :: TuckerTVector) = TuckerTVector(copy(x.Ċ), map(copy, x.U̇))

"""
Inverse of the k'th unfolding of a size n₁ × ... × n_D tensor
"""
function fold(𝔄♭ :: AbstractMatrix{T}, k, n⃗ :: NTuple{D, Int}) :: Array{T, D} where {T, D, Int}
    @assert 1 ≤ k ≤ D
    @assert size(𝔄♭, 1) == n⃗[k]

    # (compiler doesn't know we are reshaping back into order D array without type assertion)
    size_pre_permute :: NTuple{D, Int} = (n⃗[k], n⃗[1:k-1]..., n⃗[k+1:D]...)
    perm :: NTuple{D, Int} = ((2:k)..., 1, (k+1:D)...)
    permutedims(reshape(𝔄♭, size_pre_permute), perm)
end

function get_basis(:: Tucker, 𝔄 :: TuckerPoint, basisType::DefaultOrthonormalBasis{𝔽, TangentSpaceType}) where 𝔽
    D = ndims(𝔄)
    n⃗ = size(𝔄) 
    r⃗ = size(𝔄.hosvd.core) 

    U = 𝔄.hosvd.U
    U⊥ = ntuple(d -> Matrix(qr(I - U[d]*U[d]', Val(true)).Q)[:,1:n⃗[d]-r⃗[d]], D)

    basis = HOSVDBasis(𝔄, U⊥)
	CachedBasis(basisType, basis)
end

function get_coordinates(::Tucker, 𝔄, X, ℬ::CachedBasis)
    coords = vec(X.Ċ)
    for d = 1:length(X.U̇)
        coord_mtx = (ℬ.data.U⊥[d] \ X.U̇[d]) * Diagonal(𝔄.hosvd.σ[d])
        coords = vcat(coords, vec(coord_mtx'))
    end
    coords
end

function get_vector(::Tucker, 𝔄 :: TuckerPoint, ξ :: AbstractVector{T}, ℬ :: CachedHOSVDBasis) where T
    U = 𝔄.hosvd.U
    ℭ = 𝔄.hosvd.core
    σ = 𝔄.hosvd.σ
    U⊥ = ℬ.data.U⊥
    D = ndims(ℭ)
    r⃗ = size(ℭ)
    n⃗ = size(𝔄)

    # split ξ into ξ_core and ξU so that vcat(ξ_core, ξU...) == ξ
    ξ_core     = ξ[1:length(ℭ)]
    ξU         = Vector{T}[]
    nextcolumn = length(ℭ) + 1
    for d = 1:D
        numcols = r⃗[d]*(n⃗[d] - r⃗[d])
        push!(ξU, ξ[nextcolumn:nextcolumn + numcols - 1])
        nextcolumn += numcols
    end

    # Construct ∂U[d] by plugging in the definition of
    #    our orthonormal basis:
    # V[d] = ∂U[d] = ∑ᵢⱼ { ξ[d]ᵢⱼ (σ[d]ⱼ)⁻¹ U⊥[d] 𝐞ᵢ 𝐞ⱼᵀ }
    #      = ∑ⱼ (σ[d]ⱼ)⁻¹ U⊥[d] ( ∑ᵢ ξ[d]ᵢⱼ  𝐞ᵢ) 𝐞ⱼᵀ
    ∂U = similar.(U)
    for d = 1:D
        # Assuming ξ = [ξ₁₁, ..., ξ₁ⱼ, ..., ξᵢ₁, ..., ξᵢⱼ, ..., ], we can
        # reshape ξU[d] into a matrix with row indices i and column indices j
        grid = transpose(reshape(ξU[d], r⃗[d], n⃗[d] - r⃗[d]))
        # Notice that ∑ᵢ ξᵈᵢⱼ𝐞ᵢ = grid[:,j].
        # This means V[d] = U⊥[d] * grid * Diagonal(σ[d])⁻¹
        ∂U[d][:,:] = U⊥[d] * grid * Diagonal(1 ./ σ[d])
    end

    ∂C = reshape(ξ_core, size(ℭ))
    TuckerTVector(∂C, ∂U)
end

"""
    isValidTuckerRank(n⃗, r⃗)

Determines whether there are tensors of dimensions n⃗ with multilinear rank r⃗
"""
isValidTuckerRank(n⃗, r⃗) = all(r⃗ .≤ n⃗) && all(ntuple(i -> r⃗[i] ≤ prod(r⃗) ÷ r⃗[i], length(r⃗)))

manifold_dimension(:: Tucker{n⃗, r⃗}) where {n⃗, r⃗} = prod(r⃗) + sum(r⃗ .* (n⃗ .- r⃗))

Base.ndims(:: TuckerPoint{T, D}) where {T,D} = D

number_eltype(p::TuckerPoint{T,D}) where {T, D} = T
number_eltype(x::TuckerTVector{T,D}) where {T, D} = T

function retract!(::Tucker, q::TuckerPoint, p::TuckerPoint{T, D}, x::TuckerTVector, ::HOSVDRetraction) where {T, D}
    U = p.hosvd.U 
    V = x.U̇
    ℭ = p.hosvd.core
    𝔊 = x.Ċ
    r⃗ = size(ℭ)

    # Build the core tensor S and the factors [Uᵈ  Vᵈ]
    S = zeros(T, 2 .* size(ℭ))
    S[CartesianIndices(ℭ)] = ℭ + 𝔊
    UQ = Matrix{T}[]
    for d = 1:D
        # We make the following adaptation to Kressner2014:
        # Fix the i'th term of the sum and replace Vᵢ by Qᵢ Rᵢ.
        # We can absorb the R factor into the core by replacing Vᵢ by Qᵢ
        # and C (in the i'th term of the sum) by C ×ᵢ Rᵢ
        Q, R = qr(V[d])
        idxOffset = CartesianIndex(ntuple(i -> i == d ? r⃗[d] : 0, D))
        ℭ_transf  = fold(R * unfold(ℭ, d), d, size(ℭ))
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
    q
end

function Base.show(io::IO, ::MIME"text/plain", 𝒯::Tucker{N,R,D,𝔽}) where {N,R,D,𝔽}
    print(io, "Tucker(", N, ", ", R, ", ", 𝔽, ")")
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
    print(io, pre, su)
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
    print(io, pre, su)
end
function Base.show(io :: IO, mime::MIME"text/plain", ℬ :: CachedHOSVDBasis{𝔽, T, D}) where {𝔽, T, D} 
    summary(io, ℬ)
    print(" ≅")
    su = sprint(show, "text/plain", convert(Matrix{T}, ℬ); context=io, sizehint=0)
    su = replace(su, '\n' => "\n ")
    println(io, " ", su)
end


Base.size(𝔄 :: TuckerPoint) = map(u -> size(u,1), 𝔄.hosvd.U)

"""
    st_hosvd(𝔄, mlrank=size(𝔄)) 

The sequentially truncated HOSVD, as in 
[^Vannieuwenhoven2012]
> Nick Vannieuwenhoven, Raf Vandebril, Karl Meerbergen: "A new truncation strategy for the higher-order singular value decomposition"
> SIAM Journal on Scientific Computing, 34(2), pp. 1027-1052, 2012
> doi: [10.1137/110836067](https://doi.org/10.1137/110836067)
"""
function st_hosvd(𝔄, mlrank=size(𝔄)) 
    T = eltype(𝔄)
    D = ndims(𝔄)
    n⃗ = size(𝔄)
    U :: Vector{Matrix{T}} = collect(ntuple(d -> Matrix{T}(undef, n⃗[d], mlrank[d]), D))
    σ :: Vector{Vector{T}} = [Vector{T}(undef, mlrank[d]) for d = 1:D]

    for d = 1:D
        # unfold
        r_d  = mlrank[d]
        𝔄⁽ᵈ⁾ = unfold(𝔄, d)
        # truncated SVD + incremental construction of the core
        UΣVᵀ = svd(𝔄⁽ᵈ⁾)
        U[d] = UΣVᵀ.U[:,1:r_d]
        σ[d] = UΣVᵀ.S[1:r_d]
        𝔄⁽ᵈ⁾ = Diagonal(σ[d]) * UΣVᵀ.Vt[1:r_d,:]
        # reshape back into a tensor
        # (compiler doesn't know we are reshaping back into order D array without type assertion)
        m⃗ :: NTuple{D, Int} = tuple(mlrank[1:d]..., n⃗[d+1:D]...)
        𝔄 = fold(𝔄⁽ᵈ⁾, d, m⃗)
    end

    HOSVD{T, D}(tuple(U...), 𝔄, tuple(σ...))
end

"""
	unfold(𝔄, k)

Mode-k unfolding of the array 𝔄 of order d ≥ k
"""
function unfold(𝔄, k)
	d  = ndims(𝔄)
	𝔄_ = permutedims(𝔄, vcat(k, 1:k-1, k+1:d))
	reshape(𝔄_, size(𝔄, k), div(length(𝔄), size(𝔄, k)))
end


