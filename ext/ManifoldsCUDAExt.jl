"""
    ManifoldsCUDAExt

CUDA extension for Manifolds.jl providing GPU-native overrides for:
- `exp!` on `GeneralUnitaryMatrices` via Hermitian eigendecomposition
- `log_safe!` via eigendecomposition on GPU (cuSOLVER `geev!`/`heevd!`)
- QR retraction on `GeneralUnitaryMatrices` and `Grassmann`
- Random point generation on `UnitaryMatrices`
"""
module ManifoldsCUDAExt

using Manifolds
using ManifoldsBase
using ManifoldsBase: AbstractManifold
import ManifoldsBase: retract_qr_fused!

using Manifolds:
    GeneralUnitaryMatrices,
    UnitaryMatrices,
    Grassmann

using CUDA
using LinearAlgebra
using Random

# GPU-native matrix exponential for skew-Hermitian tangent vectors.
#
# For skew-Hermitian X (X' = -X), the matrix iX is Hermitian.
# Using eigen(Hermitian(iX)) = (V, λ) where λ are real:
#   exp(X) = V * Diagonal(exp(-i*λ)) * V'
#
# For real skew-symmetric X, we promote to complex, compute, and take real part.
function Manifolds.exp!(
    M::GeneralUnitaryMatrices, q::CuArray{T}, p::CuArray{T}, X::CuArray{T}
) where {T<:Complex}
    iX = Hermitian(im .* X)
    F = eigen(iX)
    expX = F.vectors * Diagonal(exp.(-im .* F.values)) * F.vectors'
    q .= p * expX
    return q
end

function Manifolds.exp!(
    M::GeneralUnitaryMatrices, q::CuArray{T}, p::CuArray{T}, X::CuArray{T}
) where {T<:Real}
    Xc = CuArray{complex(T)}(X)
    iX = Hermitian(im .* Xc)
    F = eigen(iX)
    expX = F.vectors * Diagonal(exp.(-im .* F.values)) * F.vectors'
    q .= real.(CuArray{complex(T)}(p) * expX)
    return q
end

# GPU-native matrix logarithm via eigendecomposition.
#
# cuSOLVER provides geev! for general eigendecomposition on GPU.
# For a matrix A with eigendecomposition A = V * Diagonal(λ) * V⁻¹:
#   log(A) = V * Diagonal(log(λ)) * V⁻¹
function Manifolds.log_safe!(Y::CuArray{T}, A::CuArray{T}) where {T<:Complex}
    F = eigen(A)
    Y .= F.vectors * Diagonal(log.(F.values)) * inv(F.vectors)
    return Y
end

function Manifolds.log_safe!(Y::CuArray{T}, A::CuArray{T}) where {T<:Real}
    Ac = CuArray{complex(T)}(A)
    F = eigen(Ac)
    logA = F.vectors * Diagonal(log.(F.values)) * inv(F.vectors)
    Y .= real.(logA)
    return Y
end

# GPU-native QR retraction for GeneralUnitaryMatrices.
# Base uses Diagonal(sign.(diag(R))) which triggers scalar indexing.
function ManifoldsBase.retract_qr_fused!(
    ::GeneralUnitaryMatrices,
    q::CuArray{T},
    p::CuArray{T},
    X::CuArray{T},
    t::Number,
) where {T}
    A = p + p * (T(t) * X)
    qr_decomp = qr(A)
    Q_mat = CuArray(qr_decomp.Q)
    R_mat = CuArray(qr_decomp.R)
    d = diag(R_mat)
    signs = sign.(d .+ T(0.5))
    q .= Q_mat .* reshape(signs, 1, :)
    return q
end

# GPU-native QR retraction for Grassmann.
# Base uses Array(qr.Q) creating a CPU matrix in GPU broadcast.
function ManifoldsBase.retract_qr_fused!(
    ::Grassmann,
    q::CuArray{T},
    p::CuArray{T},
    X::CuArray{T},
    t::Number,
) where {T}
    q .= p .+ T(t) .* X
    qr_decomp = qr(q)
    Q_mat = CuArray(qr_decomp.Q)
    R_mat = CuArray(qr_decomp.R)
    d = diag(R_mat)
    signs = sign.(d .+ T(0.5))
    q .= Q_mat .* reshape(signs, 1, :)
    return q
end

# GPU-native random point generation for UnitaryMatrices.
# Base rand! uses randn(rng, T, n, n) which creates CPU arrays.
function Random.rand!(
    rng::AbstractRNG,
    M::UnitaryMatrices,
    pX::CuArray;
    vector_at = nothing,
    σ::Real = one(real(eltype(pX))),
)
    n = ManifoldsBase.get_parameter(M.size)[1]
    if vector_at === nothing
        A = CUDA.randn(eltype(pX), n, n) .* σ
        qr_decomp = qr(A)
        Q_mat = CuArray(qr_decomp.Q)
        R_mat = CuArray(qr_decomp.R)
        d = diag(R_mat)
        signs = sign.(d)
        pX .= Q_mat .* reshape(signs, 1, :)
    else
        Z = CUDA.randn(eltype(pX), size(pX)...) .* σ
        Manifolds.project!(M, pX, vector_at, Z)
    end
    return pX
end

end # module
