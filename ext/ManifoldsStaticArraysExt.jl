module ManifoldsStaticArraysExt

if isdefined(Base, :get_extension)
    using Manifolds
    using StaticArrays

    import Manifolds:
        _stiefel_inv_retr_qr_mul_by_r!,
        _qrfac_to_q,
        log_lie!,
        allocate,
        diagonalizing_projectors,
        mid_point,
        get_basis_diagonalizing,
        get_coordinates_induced_basis!,
        get_coordinates_orthonormal,
        get_vector_orthonormal,
        local_metric,
        log,
        log!,
        normal_vector
else
    using ..Manifolds
    using ..StaticArrays

    import ..Manifolds:
        _stiefel_inv_retr_qr_mul_by_r!,
        _qrfac_to_q,
        log_lie!,
        allocate,
        diagonalizing_projectors,
        mid_point,
        get_basis_diagonalizing,
        get_coordinates_induced_basis!,
        get_coordinates_orthonormal,
        get_vector_orthonormal,
        local_metric,
        log,
        log!,
        normal_vector
end

allocate(p::StaticArray, s::Size{S}) where {S} = similar(p, maybesize(s))
allocate(p::StaticArray, ::Type{T}, s::Size{S}) where {S,T} = similar(p, T, maybesize(s))

#
# Circle – these probably should have fallbacks (TODO) wihtout?
function diagonalizing_projectors(M::Circle{ℝ}, p, X)
    sbv = sign(X[])
    proj = ProjectorOntoVector(M, p, @SVector [sbv == 0 ? one(sbv) : sbv])
    return ((zero(number_eltype(p)), proj),)
end

function get_basis_diagonalizing(::Circle{ℝ}, p, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.frame_direction[])
    vs = @SVector [@SVector [sbv == 0 ? one(sbv) : sbv]]
    return CachedBasis(B, (@SVector [0]), vs)
end

get_coordinates_orthonormal(::Circle{ℝ}, p, X, ::RealNumbers) = @SVector [X[]]

function get_coordinates_orthonormal(::Circle{ℂ}, p, X, ::Union{RealNumbers,ComplexNumbers})
    X, p = X[1], p[1]
    Xⁱ = imag(X) * real(p) - real(X) * imag(p)
    return @SVector [Xⁱ]
end

function get_vector_orthonormal(::Circle{ℂ}, p, c, ::Union{RealNumbers,ComplexNumbers})
    @SArray fill(1im * c[1] * p[1])
end

mid_point(M::Circle{ℂ}, p1::StaticArray, p2::StaticArray) = Scalar(mid_point(M, p1[], p2[]))

#
# EmbeddedTorus
function normal_vector(M::EmbeddedTorus, p)
    θ, φ = _torus_theta_phi(M, p)
    t = @SVector [-sin(φ), cos(φ), 0]
    s = @SVector [cos(φ) * (-sin(θ)), sin(φ) * (-sin(θ)), cos(θ)]
    return normalize(cross(t, s))
end
function get_coordinates_induced_basis!(
    M::EmbeddedTorus,
    Y,
    p,
    X,
    B::InducedBasis{ℝ,TangentSpaceType,DefaultTorusAtlas},
)
    θ, φ = get_parameters(M, B.A, B.i, p)

    sinθ, cosθ = sincos(θ + B.i[1])
    sinφ, cosφ = sincos(φ + B.i[2])

    A = @SMatrix [
        (-M.r*sinθ*cosφ) (-M.R * sinφ-M.r * cosθ * sinφ)
        (-M.r*sinθ*sinφ) (M.R * cosφ+M.r * cosθ * cosφ)
        (M.r*cosθ) 0
    ]
    Y .= A \ SVector{3}(X)
    return Y
end

#
# Euclidean
function get_vector_orthonormal(
    ::Euclidean{<:TypeParameter},
    ::SArray{S},
    c,
    ::RealNumbers,
) where {S}
    return SArray{S}(c)
end
function get_vector_orthonormal(
    ::Euclidean{TypeParameter{Tuple{N}},ℝ},
    ::SArray{S},
    c,
    ::RealNumbers,
) where {N,S}
    # probably doesn't need rewrapping in SArray
    return c
end
function local_metric(
    ::MetricManifold{𝔽,<:AbstractManifold,EuclideanMetric},
    p,
    B::InducedBasis{𝔽,TangentSpaceType,<:RetractionAtlas},
) where {𝔽}
    return Diagonal(ones(SVector{size(p, 1),eltype(p)}))
end
function local_metric(
    ::Euclidean,
    p,
    B::InducedBasis{𝔽,TangentSpaceType,<:RetractionAtlas},
) where {𝔽}
    return Diagonal(ones(SVector{size(p, 1),eltype(p)}))
end

#
# General Unitary Group
function log_lie!(
    G::GeneralUnitaryMultiplicationGroup{TypeParameter{Tuple{3}},ℝ},
    X::AbstractMatrix,
    q::AbstractMatrix,
)
    e = Identity(G)
    cosθ = (tr(q) - 1) / 2
    if cosθ ≈ -1
        eig = eigen_safe(q)
        ival = findfirst(λ -> isapprox(λ, 1), eig.values)
        inds = SVector{3}(1:3)
        ax = eig.vectors[inds, ival]
        return get_vector!(G, X, e, π * ax, DefaultOrthogonalBasis())
    end
    X .= q ./ usinc_from_cos(cosθ)
    return project!(G, X, e, X)
end

#
# General Unitary Matrices
function log(M::GeneralUnitaryMatrices{TypeParameter{Tuple{3}},ℝ}, p::SMatrix, q::SMatrix)
    U = transpose(p) * q
    cosθ = (tr(U) - 1) / 2
    if cosθ ≈ -1
        eig = eigen_safe(U)
        ival = findfirst(λ -> isapprox(λ, 1), eig.values)
        inds = SVector{3}(1:3)
        #TODO this is to stop convert error of ax as a complex number
        ax::Vector{Float64} = eig.vectors[inds, ival]
        return get_vector(M, p, π * ax, DefaultOrthogonalBasis())
    end
    X = U ./ usinc_from_cos(cosθ)
    return (X .- X') ./ 2
end
function log!(M::GeneralUnitaryMatrices{TypeParameter{Tuple{3}},ℝ}, X, p, q)
    U = transpose(p) * q
    cosθ = (tr(U) - 1) / 2
    if cosθ ≈ -1
        eig = eigen_safe(U)
        ival = findfirst(λ -> isapprox(λ, 1), eig.values)
        inds = SVector{3}(1:3)
        ax = eig.vectors[inds, ival]
        return get_vector!(M, X, p, π * ax, DefaultOrthogonalBasis())
    end
    X .= U ./ usinc_from_cos(cosθ)
    return project!(SkewSymmetricMatrices(3), X, p, X)
end

#
# Power Manifold
function allocate(::PowerManifoldNestedReplacing, x::AbstractArray{<:SArray})
    return similar(x)
end

#
# Stiefel manifold
_qrfac_to_q(qrfac::StaticArrays.QR) = qrfac.Q

function _stiefel_inv_retr_qr_mul_by_r!(
    M::Stiefel{TypeParameter{Tuple{n,1}}},
    X,
    q,
    A::StaticArray,
    ::Type{ElT},
) where {n,ElT}
    return invoke(
        _stiefel_inv_retr_qr_mul_by_r!,
        Tuple{
            Stiefel{TypeParameter{Tuple{n,1}}},
            typeof(X),
            typeof(q),
            AbstractArray,
            typeof(ElT),
        },
        M,
        X,
        q,
        A,
        ElT,
    )
end

function _stiefel_inv_retr_qr_mul_by_r!(
    M::Stiefel{TypeParameter{Tuple{n,2}}},
    X,
    q,
    A::StaticArray,
    ::Type{ElT},
) where {n,ElT}
    return invoke(
        _stiefel_inv_retr_qr_mul_by_r!,
        Tuple{
            Stiefel{TypeParameter{Tuple{n,2}}},
            typeof(X),
            typeof(q),
            AbstractArray,
            typeof(ElT),
        },
        M,
        X,
        q,
        A,
        ElT,
    )
end
function _stiefel_inv_retr_qr_mul_by_r!(
    M::Stiefel{TypeParameter{Tuple{n,k}}},
    X,
    q,
    A::StaticArray,
    ::Type{ElT},
) where {n,k,ElT}
    R = zeros(MMatrix{k,k,ElT})
    return _stiefel_inv_retr_qr_mul_by_r_generic!(M, X, q, R, A)
end
function _stiefel_inv_retr_qr_mul_by_r!(M::Stiefel, X, q, A, ::Type{ElT}) where {ElT}
    n, k = get_parameter(M.size)
    R = zeros(ElT, k, k)
    return _stiefel_inv_retr_qr_mul_by_r_generic!(M, X, q, R, A)
end

#
# Sphere
# Todo – document why this is better than the default and where the fidderence is?
function local_metric(
    ::Sphere{TypeParameter{Tuple{n}},ℝ},
    p,
    B::DefaultOrthonormalBasis,
) where {n}
    return Diagonal(ones(SVector{n,eltype(p)}))
end

#
# Unitary matrices
function get_coordinates_orthonormal(
    ::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ},
    p,
    X::Quaternions.Quaternion,
    ::QuaternionNumbers,
)
    return @SVector [X.v1, X.v2, X.v3]
end
end
