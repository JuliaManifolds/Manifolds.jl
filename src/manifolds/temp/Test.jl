include("../src/segre/Segre.jl")
include("../src/segre/SegreAlphaWarpedMetric.jl")
using StatsBase: sample

# Tests are written for manifolds with this type
T = Union{
    Segre{V, ℝ},
    MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}}
    } where {V, A}

# Verbose isapprox
import Base.isapprox
function isapprox(a, b, verbose; kwargs...)#={{{=#
    if verbose; println(a, " ?≈ ", b); end

    return isapprox(a, b; kwargs...)
end#=}}}=#


""" Approximate derivative of f at x """
function finite_difference(#={{{=#
    f::Function, # :: ℝ -> some vector space
    x::Float64,
    h::Float64;
    order=1::Int64
    )

    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    if order == 1
        return (
            (1 / 12) *  f(x - 2 * h) +
            (-2 / 3) *  f(x - 1 * h) +
            (2 / 3) *   f(x + 1 * h) +
            (-1 / 12) * f(x + 2 * h)
            ) / h
    elseif order == 2
    return (
        (-1 / 12) * f(x - 2 * h) +
        (4 / 3) *   f(x - 1 * h) +
        (-5 / 2) *  f(x) +
        (4 / 3) *   f(x + 1 * h) +
        (-1 / 12) * f(x + 2 * h)
        ) / h^2
    elseif order == 3
    return (
        (1 / 8) *   f(x - 3 * h) +
        (-1) *      f(x - 2 * h) +
        (13 / 8) *  f(x - 1 * h) +
        (-13 / 8) * f(x + 1 * h) +
        (1) *       f(x + 2 * h) +
        (-1 / 8) *  f(x + 3 * h)
        ) / h^3
    end
end#=}}}=#

""" Test that exp maps into the manifold. """
function test_exp(#={{{=#
    M::T;
    verbose=false
    )
    if verbose; println("M = ", M); end

    p = rand(M)
    if verbose; println("p = ", p); end

    v = rand(M; vector_at=p)
    if verbose; println("v = ", v); println(); end
    
    @assert(is_point(M, p))
    @assert(is_vector(M, p, v))
    @assert(is_point(M, exp(M, p, v)))
end#=}}}=#

""" Test that geodesics are unit speed. """
function test_geodesic_speed(#={{{=#
    M::T;
    verbose=false
    )
    if verbose; println("M = ", M); end
    
    p = rand(M)
    p[1][1] = p[1][1] + 0.5 # Keep away from 0
    if verbose; println("p = ", p); end
        
    v = rand(M, vector_at=p); v = v / norm(M, p, v)
    if verbose; println("v = ", v); end

    geodesic_speed = finite_difference(
        t -> distance(M, p, exp(M, p, t * v)),
        0.5 * rand(),
        1e-5
        )
    @assert(isapprox(geodesic_speed, 1.0, verbose; rtol=1e-6))
    if verbose; println(); end
end#=}}}=#

""" Test that geodesics only have normal curvature. """
function test_geodesic_curvature(#={{{=#
    M::T;
    verbose=false
    )
    # Make sure embed() is a _Riemannian_ embedding and not just a smooth embedding

    if verbose; println("M = ", M); end
    
    p = rand(M)
    if verbose; println("p = ", p); end
        
    v = rand(M, vector_at=p); v = v / norm(M, p, v)
    if verbose; println("v = ", v); end
            
    gamma(t) = embed(M, exp(M, p, t * v))
    n = finite_difference(gamma, 0.0, 1e-3; order=2) # Acceleration vector at p
    v_ = embed_vector(M, p, rand(M, vector_at=p)) # Random Tangent vector at p

    @assert(isapprox(dot(n, v_), 0.0, verbose; atol=1e-5))
    if verbose; println(); end
end#=}}}=#

""" Test that geodesics are minimizing. """
function test_geodesic_minimizes(#={{{=#
    M::T;
    verbose=false
    )
    if verbose; println("M = ", M); end
    
    p = rand(M)
    if verbose; println("p = ", p); end
        
    v = rand(M, vector_at=p); v = v / norm(M, p, v)
    if verbose; println("v = ", v); end

    n = manifold_dimension(M)

    # We test the following:
    #   Geodesics are (locally) length-minizing. So let B_a be a one-parameter
    #   family of curves such that B_0 is a geodesic. Then the derivative of
    #   length(B_a) at a = 0 should be 0, and the second derivative at should
    #   be nonnegative.

    x = get_coordinates(M, p, v)
    x0 = 0.0 * x
    x1 = 0.2 * x
    x2 = 0.4 * x
    x3 = 0.6 * x
    x4 = 0.8 * x
    x5 = 1.0 * x
            
    function curve_length(y::Vector{Float64})
        @assert(length(y) == 4 * n)

        # Control points
        y1 = y[1:n]
        y2 = y[n + 1:2 * n]
        y3 = y[2 * n + 1:3 * n]
        y4 = y[3 * n + 1:4 * n]

        # Bezier curve from 0 to v
        b(t) = (
            (1 - t)^5 * x0 +
            5 * t * (1 - t)^4 * (x1 + y1) +
            10 * t^2 * (1 - t)^3 * (x2 + y2) +
            10 * t^3 * (1 - t)^2 * (x3 + y3) +
            5 * t^4 * (1 - t) * (x4 + y4) +
            t^5 * x5
            )

        # Length of curve on manifold
        ps = [exp(M, p, get_vector(M, p, b(t))) for t in 0.0:1e-3:1.0]
        ds = [distance(M, p1, p2) for (p1, p2) in zip(ps[1:end - 1], ps[2:end])]
        return sum(ds)
    end

    dy = rand(4 * n); dy = dy / norm(dy)
    f = a -> curve_length(a * dy)
    @assert(isapprox(finite_difference(f, 0.0, 1e-3), 0.0, verbose; atol=1e-5))
    if verbose; println(finite_difference(f, 0.0, 1e-2; order=2), " ?≥ 0"); end
    @assert(finite_difference(f, 0.0, 1e-2; order=2) >= 0.0)
    if verbose; println(); end
end#=}}}=#

""" Test that log is inverse of exp. """
function test_log(#={{{=#
    M::Segre{V, ℝ};
    verbose=false
    ) where {V}
    # TODO: This function can/should be written for a general manifold that implements injectivity_radius

    if verbose; println("M = ", M); end
    
    p = rand(M)
    if verbose; println("p = ", p); end

    # Make sure we choose p and q compatible
    m(a, b) = sqrt(sum([distance(Sphere(n - 1), x, y)^2 for (n, x, y) in zip(V, a[2:end], b[2:end])]))
    q = rand(M)
    while m(p, q) > pi
        q = rand(M)
    end
    if verbose; println("q = ", q); end
        
    v = rand(M, vector_at=p); v = v / norm(M, p, v)
    if verbose; println("v = ", v); end
            
    @assert(isapprox(
        norm(embed(M, q) - embed(M, exp(M, p, log(M, p, q)))),
        0.0,
        verbose; atol=1e-10))
    if verbose; println(); end
end#=}}}=#

""" Test that log is inverse of exp. """
function test_log(#={{{=#
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}};
    verbose=false
    ) where {V, A}
    if verbose; println("M = ", M); end
    
    p = rand(M)
    if verbose; println("p = ", p); end

    # Make sure we choose p and q compatible
    m(a, b) = sqrt(sum([distance(Sphere(n - 1), x, y)^2 for (n, x, y) in zip(V, a[2:end], b[2:end])]))
    q = rand(M)
    while A * m(p, q) > pi
        q = rand(M)
    end
    if verbose; println("q = ", q); end
        
    v = rand(M, vector_at=p); v = v / norm(M, p, v)
    if verbose; println("v = ", v); end
            
    @assert(isapprox(
        norm(embed(M, q) - embed(M, exp(M, p, log(M, p, q)))),
        0.0,
        verbose; atol=1e-10))
    if verbose; println(); end
end#=}}}=#

""" Test that get_coordinates is left and right inverse of get_vector. """
function test_get_coordinates(#={{{=#
    M::T;
    verbose=false
    )
    if verbose; println("M = ", M); end
    
    p = rand(M)
    if verbose; println("p = ", p); end

    v = rand(M, vector_at=p)
    if verbose; println("v = ", v); end

    X = rand(manifold_dimension(M))
    if verbose; println("X = ", X); end

    @assert(isapprox(v, get_vector(M, p, get_coordinates(M, p, v)), verbose))
    @assert(isapprox(X, get_coordinates(M, p, get_vector(M, p, X)), verbose))
    if verbose; println(); end
end#=}}}=#

""" Test symmetries of Riemann tensor. """
function test_riemann_tensor(#={{{=#
    M::T;
    verbose=false
    )
    if verbose; println("M = ", M); end

    p = rand(M)
    if verbose; println("p = ", p); end

    u = rand(M, vector_at=p)
    v = rand(M, vector_at=p)
    w = rand(M, vector_at=p)
    z = rand(M, vector_at=p)

    if verbose; println("u = ", u); end
    if verbose; println("v = ", v); end
    if verbose; println("w = ", w); end
    if verbose; println("z = ", z); end

    @assert(is_vector(M, p, riemann_tensor(M, p, u, v, w)))

    # Symmetries
    @assert(isapprox(
        inner(M, p, riemann_tensor(M, p, u, v, w), z),
        -inner(M, p, riemann_tensor(M, p, v, u, w), z),
        verbose, atol=1e-10))
    @assert(isapprox(
        inner(M, p, riemann_tensor(M, p, u, v, w), z),
        -inner(M, p, riemann_tensor(M, p, u, v, z), w),
        verbose, atol=1e-10))
    @assert(isapprox(
        inner(M, p, riemann_tensor(M, p, u, v, w), z),
        inner(M, p, riemann_tensor(M, p, w, z, u), v),
        verbose, atol=1e-10))

    # Bianchi identity
    @assert(isapprox(
        inner(M, p, riemann_tensor(M, p, u, v, w), z)
        + inner(M, p, riemann_tensor(M, p, u, z, v), w)
        + inner(M, p, riemann_tensor(M, p, u, w, z), v),
        0.0,
        verbose, atol=1e-10))

    # Other properties of the Riemann tensor are checked by checking the sectional curvature

    if verbose; println(); end
end#=}}}=#

""" Test sectional curvature. """
function test_curvature(#={{{=#
    M::T;
    verbose=false
    )
    if verbose; println("M = ", M); end

    p = rand(M)
    p[1][1] = p[1][1] + 0.1 # Keep away from 0.0
    if verbose; println("p = ", p); end

    u = rand(M, vector_at=p); u = u / norm(M, p, u)
    v = rand(M, vector_at=p)
    v = v - inner(M, p, u, v) * u
    v = v / norm(M, p, v)

    if verbose; println("u = ", u); end
    if verbose; println("v = ", v); end

    r = 1e-2
    ps = [exp(M, p, r * (cos(theta) * u + sin(theta) * v)) for theta in 0.0:1e-3:(2 * pi)]
    ds = [distance(M, p1, p2) for (p1, p2) in zip(ps, [ps[2:end]..., ps[1]])]
    C = sum(ds)
    K = 3 * (2 * pi * r - C) / (pi * r^3) # https://en.wikipedia.org/wiki/Bertrand%E2%80%93Diguet%E2%80%93Puiseux_theorem

    @assert(isapprox(K, sectional_curvature(M, p, u, v), verbose; rtol=1e-2, atol=1e-2))
    if verbose; println(); end
end#=}}}=#

function main(;#={{{=#
    kwargs...
    )

    # List of manifolds to test
    Ms = [
        [Segre(Tuple([rand(range(2, 7)) for _ in 1:1])) for _ in 1:5]...,
        [Segre(Tuple([rand(range(2, 7)) for _ in 1:2])) for _ in 1:5]...,
        [Segre(Tuple([rand(range(2, 7)) for _ in 1:3])) for _ in 1:5]...,
        [Segre(Tuple([rand(range(2, 7)) for _ in 1:4])) for _ in 1:5]...,
        [MetricManifold(Segre(Tuple([rand(range(2, 7)) for _ in 1:1])), AlphaWarpedMetric{rand() + 0.5}()) for _ in 1:5]...,
        [MetricManifold(Segre(Tuple([rand(range(2, 7)) for _ in 1:2])), AlphaWarpedMetric{rand() + 0.5}()) for _ in 1:5]...,
        [MetricManifold(Segre(Tuple([rand(range(2, 7)) for _ in 1:3])), AlphaWarpedMetric{rand() + 0.5}()) for _ in 1:5]...,
        [MetricManifold(Segre(Tuple([rand(range(2, 7)) for _ in 1:4])), AlphaWarpedMetric{rand() + 0.5}()) for _ in 1:5]...,
        ]

    println("Testing")
    println.(Ms)

    println("Testing that exp maps to the manifold.")
    for M in Ms
        test_exp(M; kwargs...)
    end
    
    println("Testing that geodesics are unit speed.")
    for M in Ms
        test_geodesic_speed(M; kwargs...)
    end
    
    # # This only makes sense if you have a Riemannian embedding, but should be redundant anyways given test_geodesic_minimizes
    # println("Testing that geodesics only have normal curvature.")
    # for M in Ms
    #     test_geodesic_curvature(M; kwargs...)
    # end
    
    println("Testing that geodesics are minimizing.")
    for M in Ms
        test_geodesic_minimizes(M; kwargs...)
    end
    
    println("Testing that log is inverse of exp.")
    for M in Ms
        test_log(M; kwargs...)
    end

    println("Testing that get_coordinates is inverse of get_vector.")
    for M in Ms
        test_get_coordinates(M; kwargs...)
    end

    println("Testing that the Riemann tensor has the correct symmetries.")
    for M in Ms
        test_riemann_tensor(M; kwargs...)
    end

    println("Testing that sectional curvature corresponds to angular excess.")
    for M in Ms
        test_curvature(M; kwargs...)
    end

    # TODO: Test that distance and inner are compatible
    # TODO: Test that riemann_tensor and sectional_curvature are compatible (this test is trivial the way sectional_curvature is implemented atm, but should maybe be performed just so that the tests are complete)
    # TODO: Test second_fundamental_form
end#=}}}=#
