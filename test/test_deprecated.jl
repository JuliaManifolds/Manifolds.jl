using Manifolds, ManifoldsBase, Test

@testset "Deprecation tests" begin
    @test_deprecated LinearAffineMetric()
end
