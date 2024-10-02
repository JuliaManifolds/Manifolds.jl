include("../header.jl")

# Tests are written for manifolds with this type
T = Union{
    Segre{V, ℝ},
    MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}}
    } where {V, A}

# Approximate derivative of f at x
function finite_difference(
    f::Function,
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
end

# Segre is a special case of warped Segre, with warping factor 1
get_warping_factor(M::Segre{V, ℝ}) = 1.0
get_warping_factor(M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}}) where {A} = A

@testset "Segre and warped Segre" begin
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

    @testset "exp maps to the manifold." begin; for M in Ms
        p = rand(M)
        v = rand(M; vector_at=p)

        @test(is_point(M, p))
        @test(is_vector(M, p, v))
        @test(is_point(M, exp(M, p, v)))
    end; end
    
    @testset "Geodesics are unit speed." begin; for M in Ms
        p = rand(M)
        p[1][1] = p[1][1] + 0.5 # Keep away from 0
        v = rand(M, vector_at=p); v = v / norm(M, p, v)

        geodesic_speed = finite_difference(
            t -> distance(M, p, exp(M, p, t * v)),
            0.5 * rand(),
            1e-5
            )
        @test(isapprox(geodesic_speed, 1.0; rtol=1e-6))
    end; end
    
    @testset "Geodesics are minimizing." begin; for M in Ms
        p = rand(M)
        v = rand(M, vector_at=p)
        v = v / norm(M, p, v)
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
        @test(isapprox(finite_difference(f, 0.0, 1e-3), 0.0; atol=1e-5))
        @test(finite_difference(f, 0.0, 1e-2; order=2) >= 0.0)
    end; end

    @testset "log is inverse of exp." begin; for M in Ms
        # TODO: This function can be written for a general manifold that implements injectivity_radius
        p = rand(M)

        # Make sure we choose p and q compatible
        m(a, b) = sqrt(sum([distance(Sphere(n - 1), x, y)^2 for (n, x, y) in zip(V, a[2:end], b[2:end])]))
        q = rand(M)
        A = get_warping_factor(M)
        while A * m(p, q) > pi
            q = rand(M)
        end
            
        v = rand(M, vector_at=p); v = v / norm(M, p, v)
                
        @test(isapprox(norm(embed(M, q) - embed(M, exp(M, p, log(M, p, q)))), 0.0,; atol=1e-10))
    end; end

    @testset "get_coordinates is left and right inverse of get_vector. " begin; for M in Ms
        p = rand(M)
        v = rand(M, vector_at=p)
        X = rand(manifold_dimension(M))

        @test(isapprox(v, get_vector(M, p, get_coordinates(M, p, v))))
        @test(isapprox(X, get_coordinates(M, p, get_vector(M, p, X))))
    end; end

    @testset "sectional_curvature is compatible with riemann_tensor." begin; for M in Ms
        p = rand(M)
        v = rand(M, vector_at=p)
        u = rand(M, vector_at=p)

        @test(isapprox(
            sectional_curvature(u, v),
            inner(M, p, riemann_tensor(M, p, u, v, v), u) / (inner(M, p, u, u) * inner(M, p, v, v) - inner(M, p, u, v)^2)
            ))
    end; end

    @testset "Sectional curvature is difference between circumference and 2 pi r for small circles" begin; for M in Ms
        p = rand(M)
        p[1][1] = p[1][1] + 0.1 # Keep away from 0.0

        u = rand(M, vector_at=p); u = u / norm(M, p, u)
        v = rand(M, vector_at=p)
        v = v - inner(M, p, u, v) * u
        v = v / norm(M, p, v)

        r = 1e-2
        ps = [exp(M, p, r * (cos(theta) * u + sin(theta) * v)) for theta in 0.0:1e-3:(2 * pi)]
        ds = [distance(M, p1, p2) for (p1, p2) in zip(ps, [ps[2:end]..., ps[1]])]
        C = sum(ds)
        K = 3 * (2 * pi * r - C) / (pi * r^3) # https://en.wikipedia.org/wiki/Bertrand%E2%80%93Diguet%E2%80%93Puiseux_theorem

        @test(isapprox(K, sectional_curvature(M, p, u, v); rtol=1e-2, atol=1e-2))
    end; end

    # TODO: Test that distance and inner are compatible
    # TODO: Test second_fundamental_form
end
