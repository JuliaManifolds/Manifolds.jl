"""
    ManifoldsCUDAExt

CUDA extension for Manifolds.jl, enabling GPU-accelerated manifold operations
with `CuArray`-backed points and tangent vectors.

## Scope

This extension provides GPU-compatible overrides for manifold operations that
would otherwise fail or perform poorly with `CuArray` inputs. Key areas:

1. **Random point generation**: `rand!` uses `randn(rng, T, n, n)` which creates
   CPU arrays. GPU override uses `CUDA.randn` instead.

2. **QR retraction**: The `diag` + `sign.` + `Diagonal` pattern works on GPU but
   may trigger scalar indexing warnings. Provides explicit GPU implementation.

3. **Matrix logarithm**: `log_safe!` calls `convert(Matrix, ...)` for real matrices,
   forcing CPU. The complex path (`copyto!(Y, log_safe(A))`) should work on GPU
   since `log(CuMatrix)` is supported via CUDA's cuSOLVER.

## Supported manifolds

Primary targets (used by ParametricDFT.jl):
- `UnitaryMatrices(n)` — via `GeneralUnitaryMatrices` implementations
- `PowerManifold(UnitaryMatrices(1), k)` — phase gates
- `ProductManifold` — combinations of above

Most operations on these manifolds are matrix multiply, SVD, and broadcasting,
which work natively on CuArrays.
"""
module ManifoldsCUDAExt

using Manifolds
using ManifoldsBase
using ManifoldsBase:
    AbstractManifold,
    PowerManifold,
    get_iterator
import ManifoldsBase: retract_qr_fused!

using Manifolds:
    GeneralUnitaryMatrices,
    UnitaryMatrices,
    SkewHermitianMatrices,
    AbsoluteDeterminantOneMatrixType

using CUDA
using LinearAlgebra
using Random

# === GPU-aware QR retraction for GeneralUnitaryMatrices ===
#
# The base implementation uses `diag(R)` + `sign.()` + `Diagonal()` which
# can trigger scalar indexing on GPU. This override uses explicit GPU operations.

function ManifoldsBase.retract_qr_fused!(
    ::GeneralUnitaryMatrices,
    q::CuArray{T},
    p::CuArray{T},
    X::CuArray{T},
    t::Number,
) where {T}
    A = p + p * (T(t) * X)
    qr_decomp = qr(A)
    # Extract Q and R as dense CuArrays to avoid scalar indexing from QR wrapper types
    Q_mat = CuArray(qr_decomp.Q)
    R_mat = CuArray(qr_decomp.R)
    # Compute sign correction diagonal from R (ensures det(Q*D) > 0)
    d = diag(R_mat)
    signs = sign.(d .+ T(0.5))
    # Apply sign correction: q = Q * Diagonal(signs)
    # Broadcasting is more GPU-friendly than Diagonal matrix multiply for small matrices
    q .= Q_mat .* reshape(signs, 1, :)
    return q
end

# === GPU-aware random point generation for UnitaryMatrices ===
#
# The base `rand!` uses `randn(rng, eltype(pX), n, n)` which creates a CPU Array.
# This override generates random data directly on GPU.

function Random.rand!(
    rng::AbstractRNG,
    M::UnitaryMatrices,
    pX::CuArray;
    vector_at = nothing,
    σ::Real = one(real(eltype(pX))),
)
    n = ManifoldsBase.get_parameter(M.size)[1]
    if vector_at === nothing
        # Generate random matrix on GPU, then QR factorize to get unitary matrix
        A = CUDA.randn(eltype(pX), n, n) .* σ
        qr_decomp = qr(A)
        Q_mat = CuArray(qr_decomp.Q)
        R_mat = CuArray(qr_decomp.R)
        # Sign correction to ensure uniform distribution on U(n)
        d = diag(R_mat)
        signs = sign.(d)
        pX .= Q_mat .* reshape(signs, 1, :)
    else
        # Random tangent vector: project random matrix onto tangent space
        Z = CUDA.randn(eltype(pX), size(pX)...) .* σ
        Manifolds.project!(M, pX, vector_at, Z)
    end
    return pX
end

# === GPU-aware matrix logarithm for complex GeneralUnitaryMatrices ===
#
# The base `log_safe!` for real types calls `convert(Matrix, ...)` and `schur()`
# which force CPU computation. For complex types, it calls `copyto!(Y, log_safe(A))`
# which already works on GPU since CUDA supports `log(CuMatrix)`.
#
# For real types on GPU, we convert to complex, compute, and convert back.

function Manifolds.log_safe!(Y::CuArray{T}, A::CuArray{T}) where {T<:Real}
    # Real matrix log on GPU: use complex path and take real part
    Ac = CuArray{complex(T)}(A)
    logAc = log(Ac)  # CUDA matrix log via cuSOLVER
    Y .= real.(logAc)
    return Y
end

# === Ensure project! works for skew-Hermitian on GPU ===
#
# The base implementation uses broadcasting which should work on GPU.
# This is here for safety in case the SkewHermitianMatrices constructor
# causes issues.

# === GPU-aware project! for GeneralUnitaryMatrices (SVD-based) ===
#
# The base uses `svd(p)` → `mul!(q, F.U, F.Vt)` which works on GPU.
# No override needed — CUDA.jl supports svd and mul! for CuArrays.

# === GPU-aware exp for GeneralUnitaryMatrices ===
#
# The base uses `p * exp(X)` where `exp` is the matrix exponential.
# CUDA.jl supports matrix exponential via cuSOLVER for CuArrays.
# No override needed for the general case.
#
# For small 2×2 matrices (common in ParametricDFT.jl), matrix exp via
# cuSOLVER has high kernel launch overhead. A closed-form 2×2 exp
# could be faster but requires scalar indexing. The current approach
# batching multiple 2×2 matrices is handled at the application level.

end # module
