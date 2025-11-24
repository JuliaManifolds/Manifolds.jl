using Manifolds, RecursiveArrayTools, OrdinaryDiffEq, DiffEqCallbacks, BoundaryValueDiffEq

using DifferentiationInterface, ForwardDiff

using Manifolds: TangentSpaceType

using LinearAlgebra

# space outside of a black hole with Schwarzschild radius rₛ
struct BlackHoleOutside <: AbstractManifold{ℝ}
    rₛ::Float64
end

struct SchwarzschildAtlas <: AbstractAtlas{ℝ} end

manifold_dimension(::BlackHoleOutside) = 4

function Manifolds.affine_connection!(M::BlackHoleOutside, Zc, ::SchwarzschildAtlas, i, a, Xc, Yc)
    return levi_civita_affine_connection!(M, Zc, i, a, Xc, Yc)
end

function Manifolds.check_chart_switch(::BlackHoleOutside, A::SchwarzschildAtlas, i, a)
    return false
end

function Manifolds.inner(M::BlackHoleOutside, ::SchwarzschildAtlas, i, a, Xc, Yc)
    t, r, θ, ϕ = a
    r_block = (1 - M.rₛ / r)
    # assuming c = 1
    return Xc[1] * r_block * Yc[1] - Xc[2] * Yc[2] / r_block - r^2 * (Xc[3] * Yc[3] - (sin(θ)^2) * Xc[4] * Yc[4])
end


function Manifolds.get_chart_index(M::BlackHoleOutside, ::SchwarzschildAtlas, p)
    return nothing
end

function Manifolds.get_parameters!(M::BlackHoleOutside, x, ::SchwarzschildAtlas, i, p)
    x[1] = p[1] # t
    r = norm(p[2:4])
    x[2] = r
    x[3] = acos(p[4] / r) # θ
    X[4] = atan(p[3], p[2]) # ϕ
    return x
end

function Manifolds.get_point!(M::BlackHoleOutside, p, ::SchwarzschildAtlas, i, x)
    p[1] = x[1]
    p[2] = x[2] * sin(x[3]) * cos(x[4])
    p[3] = x[2] * sin(x[3]) * sin(x[4])
    p[4] = x[2] * cos(x[3])
    return p
end


# generic stuff
