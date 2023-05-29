using Manifolds, ManifoldsBase, Test

@testset "Deprecation tests" begin
    # Let's just test that for now it still works
    @test LinearAffineMetric() === AffineInvariantMetric()
end
