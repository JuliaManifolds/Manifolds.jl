
@doc raw"""
    EmbeddedTorus{TR<:Real} <: AbstractDecoratorManifold{ℝ}

Surface in ℝ³ described by parametric equations:
```math
x(θ, φ) = (R + r\cos θ)\cos φ
y(θ, φ) = (R + r\cos θ)\sin φ
z(θ, φ) = r\sin θ
```
for θ, φ in $[-π, π)$. It is assumed that $R > r > 0$.

Alternative names include anchor ring, donut and doughnut.

# Constructor

    EmbeddedTorus(R, r)
"""
struct EmbeddedTorus{TR<:Real} <: AbstractDecoratorManifold{ℝ}
    R::TR
    r::TR
end

function active_traits(f, ::EmbeddedTorus, args...)
    return merge_traits(IsMetricManifold())
end

aspect_ratio(M::EmbeddedTorus) = M.R / M.r

@doc raw"""
    check_point(M::EmbeddedTorus, p; kwargs...)

Check whether `p` is a valid point on the [`EmbeddedTorus`](@ref) `M`.
The tolerance for the last test can be set using the `kwargs...`.

The method checks if ```(p_1^2 + p_2^2 + p_3^2 + R^2 - r^2)^2```
is apprximately equal to ```4R^2(p_1^2 + p_2^2)```.
"""
function check_point(M::EmbeddedTorus, p; kwargs...)
    A = (dot(p, p) + M.R^2 - M.r^2)^2
    B = 4 * M.R^2 * (p[1]^2 + p[2]^2)
    if !isapprox(A, B; kwargs...)
        return DomainError(A - B, "The point $(p) does not lie on the $(M).")
    end
    return nothing
end

@doc raw"""
    check_vector(M::EmbeddedTorus, p, X; atol=eps(eltype(p)), kwargs...)

Check whether `X` is a valid vector tangent to `p` on the [`EmbeddedTorus`](@ref) `M`.
The method checks if the vector `X` is orthogonal to the vector normal to the torus,
see [`normal_vector`](@ref). Absolute tolerance can be set using `atol`.
"""
function check_vector(M::EmbeddedTorus, p, X; atol=eps(eltype(p)), kwargs...)
    dot_nX = dot(normal_vector(M, p), X)
    if !isapprox(dot_nX, 0; atol, kwargs...)
        return DomainError(dot_nX, "The vector $(X) is not tangent to $(p) from $(M).")
    end
    return nothing
end

function get_embedding(::EmbeddedTorus)
    return Euclidean(3)
end

@doc raw"""
    manifold_dimension(M::EmbeddedTorus)

Return the dimension of the [`EmbeddedTorus`](@ref) `M` that is 2.
"""
manifold_dimension(::EmbeddedTorus) = 2

representation_size(::EmbeddedTorus) = (3,)

@doc raw"""
    DefaultTorusAtlas()

Atlas for torus with charts indexed by two angles numbers $θ₀, φ₀ ∈ [-π, π)$. Inverse of
a chart $(θ₀, φ₀)$ is given by

```math
x(θ, φ) = (R + r\cos(θ + θ₀))\cos(φ + φ₀)
y(θ, φ) = (R + r\cos(θ + θ₀))\sin(φ + φ₀)
z(θ, φ) = r\sin(θ + θ₀)
```
"""
struct DefaultTorusAtlas <: AbstractAtlas{ℝ} end

"""
    affine_connection(M::EmbeddedTorus, A::DefaultTorusAtlas, i, a, Xc, Yc)

Affine connection on [`EmbeddedTorus`](@ref) `M`.
"""
affine_connection(M::EmbeddedTorus, A::DefaultTorusAtlas, i, a, Xc, Yc)

function affine_connection!(M::EmbeddedTorus, Zc, ::DefaultTorusAtlas, i, a, Xc, Yc)
    # as in https://www.cefns.nau.edu/~schulz/torus.pdf
    θ = a[1] .+ i[1]
    sinθ, cosθ = sincos(θ)
    Γ¹₂₂ = (M.R + M.r * cosθ) * sinθ / M.r
    Γ²₁₂ = -M.r * sinθ / (M.R + M.r * cosθ)

    Zc[1] = Xc[2] * Γ¹₂₂ * Yc[2]
    Zc[2] = Γ²₁₂ * (Xc[1] * Yc[2] + Xc[2] * Yc[1])
    return Zc
end

"""
    check_chart_switch(::EmbeddedTorus, A::DefaultTorusAtlas, i, a; ϵ = pi/3)

Return true if parameters `a` lie closer than `ϵ` to chart boundary.
"""
function check_chart_switch(::EmbeddedTorus, A::DefaultTorusAtlas, i, a; ϵ=pi / 3)
    return abs(i[1] - a[1]) > (pi - ϵ) || abs(i[2] - a[2]) > (pi - ϵ)
end

"""
    gaussian_curvature(M::EmbeddedTorus, p)

Gaussian curvature at point `p` from [`EmbeddedTorus`](@ref) `M`.
"""
function gaussian_curvature(M::EmbeddedTorus, p)
    θ, φ = _torus_theta_phi(M, p)
    return cos(θ) / (M.r * (M.R + M.r * cos(θ)))
end

"""
    inner(M::EmbeddedTorus, ::DefaultTorusAtlas, i, a, Xc, Yc)

Inner product on [`EmbeddedTorus`](@ref) in chart `i` in the [`DefaultTorusAtlas`](@ref).
between vectors with coordinates `Xc` and `Yc` tangent at point with parameters `a`.
Vector coordinates must be given in the induced basis.
"""
function inner(M::EmbeddedTorus, ::DefaultTorusAtlas, i, a, Xc, Yc)
    diag_1 = M.r^2
    diag_2 = (M.R + M.r * cos(a[1] + i[1]))^2
    return Xc[1] * diag_1 * Yc[1] + Xc[2] * diag_2 * Yc[2]
end

"""
    inverse_chart_injectivity_radius(M::AbstractManifold, A::AbstractAtlas, i)

Injectivity radius of `get_point` for chart `i` from the [`DefaultTorusAtlas`](@ref) `A` of the [`EmbeddedTorus`](@ref).
"""
function inverse_chart_injectivity_radius(::EmbeddedTorus, ::DefaultTorusAtlas, i)
    return π
end

function _torus_theta_phi(M::EmbeddedTorus, p)
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

function _torus_param(M::EmbeddedTorus, θ, φ)
    sinθ, cosθ = sincos(θ)
    sinφ, cosφ = sincos(φ)
    return ((M.R + M.r * cosθ) * cosφ, (M.R + M.r * cosθ) * sinφ, M.r * sinθ)
end

"""
    normal_vector(M::EmbeddedTorus, p)

Outward-pointing normal vector on the [`EmbeddedTorus`](@ref) at the point `p`.
"""
function normal_vector(M::EmbeddedTorus, p)
    θ, φ = _torus_theta_phi(M, p)
    t = @SVector [-sin(φ), cos(φ), 0]
    s = @SVector [cos(φ) * (-sin(θ)), sin(φ) * (-sin(θ)), cos(θ)]
    return normalize(cross(t, s))
end

function get_chart_index(M::EmbeddedTorus, ::DefaultTorusAtlas, p)
    return _torus_theta_phi(M, p)
end
function get_chart_index(::EmbeddedTorus, ::DefaultTorusAtlas, i, a)
    return (a[1], a[2])
end

function get_parameters!(M::EmbeddedTorus, x, ::DefaultTorusAtlas, i::NTuple{2}, p)
    x .= _torus_theta_phi(M, p) .- i
    return x
end

function get_point!(M::EmbeddedTorus, p, ::DefaultTorusAtlas, i, x)
    p .= _torus_param(M, (x .+ i)...)
    return p
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

function get_vector_induced_basis!(
    M::EmbeddedTorus,
    Y,
    p,
    X,
    B::InducedBasis{ℝ,TangentSpaceType,DefaultTorusAtlas},
)
    θ, φ = get_parameters(M, B.A, B.i, p)
    dθ, dφ = X
    sinθ, cosθ = sincos(θ + B.i[1])
    sinφ, cosφ = sincos(φ + B.i[2])
    Y[1] = -M.r * sinθ * cosφ * dθ - (M.R + M.r * cosθ) * sinφ * dφ
    Y[2] = -M.r * sinθ * sinφ * dθ + (M.R + M.r * cosθ) * cosφ * dφ
    Y[3] = M.r * cosθ * dθ
    return Y
end
