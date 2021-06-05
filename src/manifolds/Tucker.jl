
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
    decomp   = hosvd(core′)
    factors′ = Q .* decomp.U
    TuckerPoint(HOSVD{T, D}(factors′, decomp.core, decomp.σ))
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
    U̇ :: Vector{Matrix{T}}
end

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

"""
    isValidTuckerRank(n⃗, r⃗)

Determines whether there are tensors of dimensions n⃗ with multilinear rank r⃗
"""
isValidTuckerRank(n⃗, r⃗) = all(r⃗ .≤ n⃗) && all(ntuple(i -> r⃗[i] ≤ prod(r⃗) ÷ r⃗[i], length(r⃗)))

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
    hosvd_S = st_hosvd(S, mlrank=r⃗)
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

"""
    st_hosvd(𝔄; mlrank=size(𝔄)) 

The sequentially truncated HOSVD, as in 
[^Vannieuwenhoven2012]
> Nick Vannieuwenhoven, Raf Vandebril, Karl Meerbergen: "A new truncation strategy for the higher-order singular value decomposition"
> SIAM Journal on Scientific Computing, 34(2), pp. 1027-1052, 2012
> doi: [10.1137/110836067](https://doi.org/10.1137/110836067)
"""
function st_hosvd(𝔄; mlrank=size(𝔄)) 
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


