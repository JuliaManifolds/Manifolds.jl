include("../utils.jl")

@testset "Minkowski Metric" begin
    N = Euclidean(3)
    M = MetricManifold(N, MinkowskiMetric())
    @test local_metric(M, zeros(3)) == Diagonal([1.0, 1.0, -1.0])
end
