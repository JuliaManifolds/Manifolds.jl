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
end
