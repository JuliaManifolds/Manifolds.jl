include("../header.jl")

@testset "Lorentz Manifold" begin
    @test_throws DomainError Lorentz(-1)
    M = Lorentz(3)
    @test is_flat(M)
    @test is_flat(Lorentz(3; parameter = :field))
    @testset "Minkowski Metric" begin
        N = MetricManifold(Euclidean(3), MinkowskiMetric())
        @test N == M
        @test local_metric(M, zeros(3)) == Diagonal([1.0, 1.0, -1.0])
        # check minkowski metric is called
        p = zeros(3)
        X = [1.0, 2.0, 3.0]
        Y = [2.0, 3.0, 4.0]
        @test minkowski_metric(X, Y) == -4
        @test inner(M, p, X, Y) == minkowski_metric(X, Y)
    end
end
