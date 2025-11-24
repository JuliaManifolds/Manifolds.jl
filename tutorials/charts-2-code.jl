using Manifolds, RecursiveArrayTools, OrdinaryDiffEq, DiffEqCallbacks, BoundaryValueDiffEq

using DifferentiationInterface, ForwardDiff

using Manifolds: TangentSpaceType

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

function levi_civita_affine_connection!(M::AbstractManifold, Zc, A::AbstractAtlas, i, a, Xc, Yc; 
    backend::AbstractADType = AutoForwardDiff())
    # number of coordinates
    n = length(a)

    # metric g_{ij} in this chart (coordinate basis)
    g = zeros(Float64, n, n)
    for p in 1:n, q in 1:n
        e_p = zeros(Float64, n); e_p[p] = 1.0
        e_q = zeros(Float64, n); e_q[q] = 1.0
        g[p, q] = inner(M, A, i, a, e_p, e_q)
    end

    # inverse metric
    ginv = inv(Symmetric(g))

    # helper: directional derivative at a in direction dir of the scalar function
    # f_dir(V1, V2) = d/dt|0 inner(M, A, i, a + t*dir, V1, V2)
    function directional_derivative_scalar(dir, V1, V2)
        out = zeros(1)
        f(t) = inner(M, A, i, a .+ (t .* dir), V1, V2)
        return DifferentiationInterface.derivative(f, out, backend, 0.0)
    end

    # compute S_k = 1/2 ( X[g(Y, e_k)] + Y[g(X, e_k)] - e_k[g(X,Y)] )
    S = zeros(Float64, n)
    for k in 1:n
        e_k = zeros(Float64, n); e_k[k] = 1.0
        term1 = directional_derivative_scalar(Xc, Yc, e_k)
        term2 = directional_derivative_scalar(Yc, Xc, e_k)
        term3 = directional_derivative_scalar(e_k, Xc, Yc)
        S[k] = 0.5 * (term1 + term2 - term3)
    end

    # raise index: (∇_X Y)^l = g^{l k} S_k
    Zc .= ginv * S

    return Zc
end


function get_coordinates_induced_basis_generic!(
        M::AbstractManifold,
        c,
        p,
        X,
        B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas};
        backend::AbstractADType = AutoForwardDiff(),
    )
    DifferentiationInterface.derivative!(t -> get_parameters(M, B.A, B.i, p + t * X), c, backend, 0.0)
    return c
end

function get_vector_induced_basis_generic!(
        M::AbstractManifold,
        Y,
        p,
        Xc,
        B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas};
        backend::AbstractADType = AutoForwardDiff(),
    )
    p_i = get_parameters(M, B.A, B.i, p)
    DifferentiationInterface.derivative!(t -> get_point(M, B.A, B.i, p_i + t * Xc), Y, backend, 0.0)
    
    return Y
end
