
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

"""
    HOSVD{D, T}

Higher-order singular value decomposition of an order D tensor with eltype T 
fields: 
* U: singular vectors of the unfoldings
* core: core tensor
* σ : singular values of the unfoldings
"""
struct HOSVD{D, T}
    U    :: NTuple{D, Matrix{T}}
    core :: Array{T, D}
    σ    :: NTuple{D, Vector{T}}
end

"""
    TuckerPoint{D, T} 

An order D tensor of fixed multilinear rank and entries of type T.
"""
struct TuckerPoint{D, T} <: AbstractManifoldPoint
    hosvd :: HOSVD{D, T}
end
function TuckerPoint(core :: AbstractArray{T, D}, factors :: Vararg{MtxT, D}) where {T, D, MtxT <: AbstractMatrix{T}}
    # Take the QR decompositions of the factors and multiply the R factors into the core
    qrfacs  = qr.(factors)
    Q       = map(qrfac -> qrfac.Q, qrfacs)
    R       = map(qrfac -> qrfac.R, qrfacs)
    core′   = reshape(Kronecker.:⊗(reverse(R)...) * vec(core), size(core))

    # Convert to HOSVD format by taking the HOSVD of the core
    _TuckerPoint_orthogonal(core′, Q...)
end

@doc raw"""
_TuckerPoint_orthogonal(core :: AbstractArray{T, D}, Q :: Vararg{MtxT, D}) where {T, D, MtxT <: AbstractMatrix{T}}

Create a Tucker tensor $(Q_1,\dots,Q_D) \cdot \mathcal{C} $ where the matrices Q are already orthogonal
"""
function _TuckerPoint_orthogonal(core :: AbstractArray{T, D}, Q :: Vararg{MtxT, D}) where {T, D, MtxT <: AbstractMatrix{T}}
    # All we need to do is ensure that the core is in HOSVD form
    decomp   = st_hosvd(core)
    factors  = Q .* decomp.U
    TuckerPoint(HOSVD{D, T}(factors, decomp.core, decomp.σ))
end

@doc raw"""
    TuckerTVectort{D, T} <: TVector

Tangent space to the Tucker manifold at `x = (U_1,\dots,U_D) ⋅ \mathcal{C}`. This vector is represented as
```math 
(U_1,\dots,U_D) \cdot \dot{\mathcal{C}} + \sum_{d=1}^D (U_1,\dots,U_{d-1},\dot{U}_d,U_{d+1},\dots,U_D) \cdot \mathcal{C}
````
where $\dot_{U}_d^\mathrm{H} U_d = 0$
"""
struct TuckerTVector{D, T} <: TVector
    Ċ :: Array{D, T}
    U̇ :: Vector{Matrix{T}}
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

    HOSVD{D, T}(tuple(U...), 𝔄, tuple(σ...))
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


