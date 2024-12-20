using Manifolds, Test, Random, LinearAlgebra, FiniteDifferences

@testset "Segre Manifold" begin
    # Manifolds to test
    Ms = [
        Segre(10),
        Segre(7, 2),
        Segre(7, 9, 9),
        Segre(9, 3, 6, 6),
        MetricManifold(Segre(10), WarpedMetric(1.2025837056880606)),
        MetricManifold(Segre(2, 9), WarpedMetric(1.1302422072971439)),
        MetricManifold(Segre(9, 6, 10), WarpedMetric(1.4545138169484464)),
        MetricManifold(Segre(9, 3, 8, 10), WarpedMetric(1.396673190458706)),
    ]

    # Vs[i] is the valence of Ms[i]
    Vs = [(10,), (7, 2), (7, 9, 9), (9, 3, 6, 6), (10,), (2, 9), (9, 6, 10), (9, 3, 8, 10)]

    # n ≥ k, for same n,k X is in TpM and can be scaled by l
    unit_p(n, k) = 1 / sqrt(k) .* [ones(k)..., zeros(n - k)...]
    unit_X(n, k; l=1.0) = l / sqrt(n - k) .* [zeros(k)..., ones(n - k)...]
    unit_c(n, k) = normalize([mod(k * i^i, 31) - 30 / 2 for i in 1:n]) # pseudo-rng

    # ps[i] is a point on Ms[i]
    ps = [
        [[0.5], unit_p(10, 4)],
        [[0.6], unit_p(7, 3), unit_p(2, 1)],
        [[0.7], unit_p(7, 3), unit_p(9, 5), unit_p(9, 4)],
        [[0.8], unit_p(9, 3), unit_p(3, 1), unit_p(6, 4), unit_p(6, 3)],
        [[0.9], unit_p(10, 4)],
        [[1.0], unit_p(2, 1), unit_p(9, 5)],
        [[1.1], unit_p(9, 3), unit_p(6, 5), unit_p(10, 4)],
        [[1.2], unit_p(9, 3), unit_p(3, 1), unit_p(8, 4), unit_p(10, 4)],
    ]

    # qs[i] is a point on Ms[i] that is connected to ps[i] by a geodesic and uses the closest representative to ps[i]
    # (tricked by only switching only one entry to zero)
    qs = [
        [[0.1], unit_p(10, 3)],
        [[0.2], unit_p(7, 2), unit_p(2, 2)],
        [[0.3], unit_p(7, 2), unit_p(9, 4), unit_p(9, 3)],
        [[0.4], unit_p(9, 3), unit_p(3, 1), unit_p(6, 3), unit_p(6, 2)],
        [[0.5], unit_p(10, 3)],
        [[0.6], unit_p(2, 2), unit_p(9, 5)],
        [[0.7], unit_p(9, 2), unit_p(6, 4), unit_p(10, 3)],
        [[0.8], unit_p(9, 2), unit_p(3, 2), unit_p(8, 3), unit_p(10, 3)],
    ]

    # Xs[i] is a tangent vector to Ms[i] at ps[i]
    Xs = [
        [[0.5], unit_X(10, 4)],
        [[0.6], unit_X(7, 3), unit_X(2, 1)],
        [[0.7], unit_X(7, 3), unit_X(9, 5), unit_X(9, 4)],
        [[0.8], unit_X(9, 3), unit_X(3, 1), unit_X(6, 4), unit_X(6, 3)],
        [[0.9], unit_X(10, 4)],
        [[1.0], unit_X(2, 1), unit_X(9, 5)],
        [[1.1], unit_X(9, 3), unit_X(6, 5), unit_X(10, 4)],
        [[1.2], unit_X(9, 3), unit_X(3, 1), unit_X(8, 4), unit_X(10, 4)],
    ]

    # Ys[i] is a tangent vector to Ms[i] at ps[i] such that exp(Ms[i], ps[i], t * vs[i]) is the closes representative to ps[i] for t in [-1, 1]
    Ys = [
        [[0.5], unit_X(10, 5)],
        [[0.6], unit_X(7, 4), unit_X(2, 1)],
        [[0.7], unit_X(7, 4), unit_X(9, 6), unit_X(9, 5)],
        [[0.8], unit_X(9, 4), unit_X(3, 1), unit_X(6, 5), unit_X(6, 4)],
        [[0.9], unit_X(10, 5)],
        [[1.0], unit_X(2, 1), unit_X(9, 6)],
        [[1.1], unit_X(9, 4), unit_X(6, 5), unit_X(10, 5)],
        [[1.2], unit_X(9, 4), unit_X(3, 1), unit_X(8, 5), unit_X(10, 5)],
    ]

    # cs[i] is coordinates for a tangent vector at ps[i]
    cs = [
        unit_c(10, 1),
        unit_c(8, 2),
        unit_c(23, 3),
        unit_c(21, 4),
        unit_c(10, 5),
        unit_c(10, 6),
        unit_c(23, 7),
        unit_c(27, 8),
    ]

    # When testing that exp(Ms[i], ps[i], t * Xs[i]) is an extremum of the length functional, we take a directional derivative along dcs[i]
    dcs = [
        unit_c(3 * 10, 9),
        unit_c(3 * 8,  10),
        unit_c(3 * 23, 11),
        unit_c(3 * 21, 12),
        unit_c(3 * 10, 13),
        unit_c(3 * 10, 14),
        unit_c(3 * 23, 15),
        unit_c(3 * 27, 16),
    ]

    for (M, V, p, q, X, Y, c, dc) in zip(Ms, Vs, ps, qs, Xs, Ys, cs, dcs)
        @testset "Manifold $M" begin
            @testset "Segre" begin
                get_manifold(::Segre{ℝ,V}) where {V} = Segre{ℝ,V}()
                get_manifold(::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}) where {V,A} =
                    Segre{ℝ,V}()
                @test Segre(V...) == get_manifold(M)
            end

            @testset "manifold_dimension" begin
                @test manifold_dimension(M) == 1 + sum(V .- 1)
            end

            @testset "is_point" begin
                @test is_point(M, p)
                @test is_point(M, q)
                @test_throws DomainError is_point(
                    M,
                    [[1.0, 0.0], p[2:end]...];
                    error=:error,
                )
                @test_throws DomainError is_point(M, [[-1.0], p[2:end]...]; error=:error)
                @test_throws DomainError is_point(M, [p[1], 2 * p[2:end]...]; error=:error)
            end

            @testset "is_vector" begin
                @test is_vector(M, p, X; error=:error)
                @test is_vector(M, p, Y; error=:error)
                @test_throws DomainError is_vector(
                    M,
                    [[1.0, 0.0], p[2:end]...],
                    X,
                    false,
                    true,
                )
                @test_throws DomainError is_vector(
                    M,
                    p,
                    [[1.0, 0.0], X[2:end]...],
                    false,
                    true,
                )
                @test_throws DomainError is_vector(M, p, p, false, true)
            end

            Random.seed!(1)
            @testset "rand" begin
                @test is_point(M, rand(M))
                @test is_vector(M, p, rand(M, vector_at=p))
            end

            @testset "get_embedding" begin
                @test get_embedding(M) == Euclidean(prod(V))
            end

            @testset "embed!" begin
                # points
                p_ = zeros(prod(V))
                p__ = zeros(prod(V))
                embed!(M, p_, p)
                embed!(M, p__, [p[1], [-x for x in p[2:end]]...])
                @test is_point(get_embedding(M), p_)
                @test isapprox(p_, (-1)^length(V) * p__)

                # vectors
                X_ = zeros(prod(V))
                embed!(M, X_, p, X)
                @test is_vector(get_embedding(M), p_, X_)
            end

            @testset "get_coordinates" begin
                @test isapprox(X, get_vector(M, p, get_coordinates(M, p, X)))
                @test isapprox(c, get_coordinates(M, p, get_vector(M, p, c)))

                # Coordinates are ON
                @test isapprox(
                    dot(c, get_coordinates(M, p, X)),
                    inner(M, p, X, get_vector(M, p, c)),
                )
            end

            @testset "exp" begin
                # Zero vector
                p_ = exp(M, p, zeros.(size.(X)))
                @test is_point(M, p_)
                @test isapprox(p, p_; atol=1e-5)

                # Tangent vector in the scaling direction
                p_ = exp(M, p, [X[1], zeros.(size.(X[2:end]))...])
                @test is_point(M, p_)
                @test isapprox([p[1] + X[1], p[2:end]...], p_; atol=1e-5)

                # Generic tangent vector
                p_ = exp(M, p, X)
                @test is_point(M, p)

                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * X)), -1.0)
                @test isapprox(geodesic_speed, -norm(M, p, X); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * X)), -0.811)
                @test isapprox(geodesic_speed, -norm(M, p, X); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * X)), -0.479)
                @test isapprox(geodesic_speed, -norm(M, p, X); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * X)), 0.181)
                @test isapprox(geodesic_speed, norm(M, p, X); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * X)), 0.703)
                @test isapprox(geodesic_speed, norm(M, p, X); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * X)), 1.0)
                @test isapprox(geodesic_speed, norm(M, p, X); atol=1e-5)

                # Geodesics are (locally) length-minizing. So let B_a be a one-parameter
                # family of curves such that B_0 is a geodesic. Then the derivative of
                # length(B_a) at a = 0 should be 0, and the second derivative should be
                # nonnegative.

                n = manifold_dimension(M)
                c0 = 0.0 * c
                c1 = 0.25 * c
                c2 = 0.5 * c
                c3 = 0.75 * c
                c4 = 1.0 * c

                function curve_length(d::Vector{Float64})
                    @assert(length(d) == 3 * n)

                    # Control points
                    d1 = d[1:n]
                    d2 = d[(n + 1):(2 * n)]
                    d3 = d[(2 * n + 1):(3 * n)]

                    # Bezier curve from 0 to v with control points y1, ..., y4
                    function b(t)
                        return (
                            (1 - t)^4 * c0 +
                            4 * t * (1 - t)^3 * (c1 + d1) +
                            6 * t^2 * (1 - t)^2 * (c2 + d2) +
                            4 * t^3 * (1 - t) * (c3 + d3) +
                            t^4 * c4
                        )
                    end

                    # Length of curve on manifold
                    ps_ = [exp(M, p, get_vector(M, p, b(t))) for t in 0.0:1e-3:1.0]
                    ds = [
                        distance(M, p1, p2) for
                        (p1, p2) in zip(ps_[1:(end - 1)], ps_[2:end])
                    ]
                    return sum(ds)
                end

                f = a -> curve_length(a * dc)
                @test isapprox(central_fdm(3, 1)(f, 0.0), 0.0; atol=1e-5)
                @test central_fdm(3, 2)(f, 0.0) >= 0.0
            end

            @testset "log" begin
                # Same point
                X_ = log(M, p, p)
                @test is_vector(M, p, X_)
                @test isapprox(zeros.(size.(X)), X_; atol=1e-5)

                # Scaled point
                X_ = log(M, p, [q[1], p[2:end]...])
                @test is_vector(M, p, X_)
                @test isapprox(X_, [q[1] - p[1], zeros.(size.(q[2:end]))...]; atol=1e-5)

                # Generic tangent vector
                X_ = log(M, p, q)
                @test is_vector(M, p, X_)
            end

            @testset "norm" begin
                @test isapprox(norm(M, p, log(M, p, q)), distance(M, p, q))
            end

            @testset "sectional_curvature" begin
                # Test that sectional curvature is difference between circumference
                # and 2 pi r for small circles.

                # Orthonormalize
                X_ = X / norm(M, p, X)
                Y_ = Y - inner(M, p, X_, Y) * X_
                Y_ = Y_ / norm(M, p, Y_)

                r = 1e-2
                ps_ = [
                    exp(M, p, r * (cos(theta) * X_ + sin(theta) * Y_)) for
                    theta in 0.0:1e-3:(2 * pi)
                ]
                ds = [distance(M, p1, p2) for (p1, p2) in zip(ps_, [ps_[2:end]..., ps_[1]])]
                C = sum(ds)
                K = 3 * (2 * pi * r - C) / (pi * r^3) # https://en.wikipedia.org/wiki/Bertrand%E2%80%93Diguet%E2%80%93Puiseux_theorem

                @test isapprox(K, sectional_curvature(M, p, X, Y); rtol=1e-2, atol=1e-2)
            end

            @testset "riemann_tensor" begin
                @test isapprox(
                    sectional_curvature(M, p, X, Y),
                    inner(M, p, riemann_tensor(M, p, X, Y, Y), X) /
                    (inner(M, p, X, X) * inner(M, p, Y, Y) - inner(M, p, X, Y)^2),
                )
            end
        end
    end

    # Test a point that does not use the closest representative
    @testset "log" begin
        M = Ms[4]
        p = ps[4]
        q = qs[4]
        q_ = [q[1], q[2], q[3], q[4], -q[5]]
        @test is_vector(M, p, log(M, p, q_))

        M = Ms[8]
        p = ps[8]
        q = qs[8]
        q_ = [q[1], q[2], q[3], q[4], -q[5]]
        @test is_vector(M, p, log(M, p, q_))
    end
end
