using Manifolds, Test

@testset "Deprecation tests" begin
    @testset "local metric in a basis" begin
        M = MetricManifold(Sphere(2), EuclideanMetric())
        p = [1.0, 0.0, 0.0]
        B = DefaultOrthonormalBasis()
        for (f, v) in [
                (local_metric, [1.0 0.0; 0.0 1.0]),
                (det_local_metric, 1.0),
                (inverse_local_metric, [1.0 0.0; 0.0 1.0]),
                (log_local_metric_density, 0.0),
            ]
            # the method that f delegates to may warn as well
            msg = Regex("^`$(nameof(f))\\(.*\\)` is deprecated")
            @test (@test_logs (:warn, msg) match_mode = :any f(M, p, B)) == v
        end
    end

    @testset "N_pts of estimated_sectional_curvature_matrix" begin
        M = Sphere(2)
        p = [1.0, 0.0, 0.0]
        B = DefaultOrthonormalBasis()
        msg = r"^The keyword `N_pts` of `estimated_sectional_curvature_matrix` is deprecated"
        cm = Manifolds.estimated_sectional_curvature_matrix(M, p, B; r = 1.0e-3, N = 200)
        @test (@test_logs (:warn, msg) Manifolds.estimated_sectional_curvature_matrix(M, p, B; r = 1.0e-3, N_pts = 200)) == cm
    end

    @testset "spd_sqrt, spd_sqrt_inv and spd_sqrt_and_sqrt_inv" begin
        p = [2.0 0.5; 0.5 1.0]
        q = Manifolds.MatrixSqrtManifoldPoint(p)
        for (f, g) in ((Manifolds.spd_sqrt, Manifolds.hpd_sqrt), (Manifolds.spd_sqrt_inv, Manifolds.hpd_sqrt_inv))
            msg = Regex("^`$(nameof(f))\\(p\\)` is deprecated")
            r = @test_logs (:warn, msg) match_mode = :any f(q)
            @test r ≈ g(q)
            @test r isa Symmetric
        end
        msg = r"^`spd_sqrt_and_sqrt_inv\(p\)` is deprecated"
        r = @test_logs (:warn, msg) match_mode = :any Manifolds.spd_sqrt_and_sqrt_inv(p)
        @test all(r .≈ Manifolds.hpd_sqrt_and_sqrt_inv(p))
        @test all(x -> x isa Symmetric, r)
    end

    @testset "SPDPoint" begin
        @test Base.isdeprecated(Manifolds, :SPDPoint)
        @test Manifolds.SPDPoint === MatrixSqrtManifoldPoint
    end
end
