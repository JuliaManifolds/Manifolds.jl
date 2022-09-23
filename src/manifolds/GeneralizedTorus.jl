
@doc raw"""
    TorusInR3{TR<:Real} <: AbstractManifold{ℝ}

Surface in ℝ³ described by parametric equations:
```math
x(θ, φ) = (R + r\cos θ)\cos φ
y(θ, φ) = (R + r\cos θ)\sin φ
z(θ, φ) = r\sin φ
```
for θ, φ in $[-π, π)$. It is assumed that $R > r > 0$.

Alternative names include anchor ring, donut and doughnut.

# Constructor

    TorusInR3(R, r)
"""
struct TorusInR3{TR<:Real} <: AbstractManifold{ℝ}
    R::TR
    r::TR
end

aspect_ratio(M::TorusInR3) = M.R / M.r

"""
    check_point(M::TorusInR3, p; kwargs...)

Check whether `p` is a valid point on the [`TorusInR3`](@ref) `M`.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(M::TorusInR3, p; kwargs...)
    A = (dot(p, p) + M.R^2 - M.r^2)^2
    B = 4 * M.R^2 * (p[1]^2 + p[2]^2)
    if !isapprox(A, B; kwargs...)
        return DomainError(A - B, "The point $(p) does not lie on the $(M).")
    end
    return nothing
end

function check_vector(M::TorusInR3, p, X; atol=eps(eltype(p)), kwargs...)
    dot_nX = dot(_torus_normal(M, p), X)
    if !isapprox(dot_nX, 0; atol, kwargs...)
        return DomainError(dot_nX, "The vector $(X) is not tangent to $(p) from $(M).")
    end
    return nothing
end

function get_embedding(::TorusInR3)
    return Euclidean(3)
end

@doc raw"""
    manifold_dimension(M::TorusInR3)

Return the dimension of the [`AbstractSphere`](@ref) `M`, respectively i.e. the
dimension of the embedding -1.
"""
manifold_dimension(::TorusInR3) = 2

representation_size(::TorusInR3) = (3,)

@doc raw"""
    DefaultTorusAtlas()

Atlas for torus with charts indexed by two angles numbers $θ₀, φ₀ ∈ [-π, π)$. Inverse of
a chart $(θ₀, φ₀)$ is given by

```math
x(θ, φ) = (R + r\cos(θ + θ₀))\cos(φ + φ₀)
y(θ, φ) = (R + r\cos(θ + θ₀))\sin(φ + φ₀)
z(θ, φ) = r\sin(φ + φ₀)
```
"""
struct DefaultTorusAtlas <: AbstractAtlas{ℝ} end

function affine_connection!(
    M::TorusInR3,
    Zc,
    p,
    Xc,
    Yc,
    B::InducedBasis{ℝ,TangentSpaceType,DefaultTorusAtlas},
)
    # as in https://www.cefns.nau.edu/~schulz/torus.pdf
    θ = p[1]
    sinθ, cosθ = sincos(θ)
    Γ²₁₁ = (M.R + M.r * cosθ) * sinθ / M.r
    Γ¹₁₂ = -M.r * sinθ / (M.R + M.r * cosθ)

    Zc[1] = Xc' * [0 Γ¹₁₂; Γ²₁₁ 0] * Yc
    Zc[2] = Xc' * [Γ¹₁₂ 0; 0 0] * Yc
    return Zc
end

"""
    inverse_chart_injectivity_radius(M::AbstractManifold, A::AbstractAtlas, i)

Injectivity radius of `get_point` for chart `i` from atlas `A` of manifold `M`.
"""
function inverse_chart_injectivity_radius(::TorusInR3, ::DefaultTorusAtlas, i)
    return π
end

function _torus_theta_phi(M::TorusInR3, p)
    φ = atan(p[2], p[1])
    sinφ, cosφ = sincos(φ)
    rsinθ = p[3]
    if p[1]^2 > p[2]^2
        rcosθ = p[1] / cosφ - M.R
    else
        rcosθ = p[2] / sinφ - M.R
    end
    return (atan(rsinθ, rcosθ), φ)
end

function _torus_param(M::TorusInR3, θ, φ)
    sinθ, cosθ = sincos(θ)
    sinφ, cosφ = sincos(φ)
    return ((M.R + M.r * cosθ) * cosφ, (M.R + M.r * cosθ) * sinφ, M.r * sinθ)
end

"""
    _torus_normal(M::TorusInR3, p)

Outward-pointing normal vector to torus at `p`.
"""
function _torus_normal(M::TorusInR3, p)
    θ, φ = _torus_theta_phi(M, p)
    t = @SVector [-sin(φ), cos(φ), 0]
    s = @SVector [cos(φ) * (-sin(θ)), sin(φ) * (-sin(θ)), cos(θ)]
    return normalize(cross(t, s))
end

function get_chart_index(M::TorusInR3, ::DefaultTorusAtlas, p)
    return _torus_theta_phi(M, p)
end
function get_chart_index(::TorusInR3, ::DefaultTorusAtlas, i, a)
    return (a[1], a[2])
end

function get_parameters!(M::TorusInR3, x, ::DefaultTorusAtlas, i::NTuple{2}, p)
    x .= _torus_theta_phi(M, p) .- i
    return x
end

function get_point!(M::TorusInR3, p, ::DefaultTorusAtlas, i, x)
    p .= _torus_param(M, (x .+ i)...)
    return p
end

function get_coordinates_induced_basis!(
    M::TorusInR3,
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

function get_vector_induced_basis!(
    M::TorusInR3,
    Y,
    p,
    X,
    B::InducedBasis{ℝ,TangentSpaceType,DefaultTorusAtlas},
)
    θ, φ = get_parameters(M, B.A, B.i, p)
    dθ, dφ = X
    sinθ, cosθ = sincos(θ + B.i[1])
    sinφ, cosφ = sincos(φ + B.i[2])
    Y[1] = -M.R * sinφ * dφ + M.r * (-cosθ * sinφ * dφ - sinθ * cosφ * dθ)
    Y[2] = M.R * cosφ * dφ + M.r * (cosθ * cosφ * dφ - sinθ * sinφ * dθ)
    Y[3] = M.r * cosθ * dθ
    return Y
end

function local_metric(M::TorusInR3, p, ::InducedBasis{ℝ,TangentSpaceType,DefaultTorusAtlas})
    diag = ((M.R + M.r * cos(p[1]))^2, M.r^2)
    return Diagonal(SVector(diag))
end
