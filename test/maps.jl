include("utils.jl")

using Manifolds: Map, Curve, RealField

@testset "Maps" begin
    r2 = Euclidean(2)
    s2 = Sphere(2)
    m1 = Map(r2, s2)
    s2c = Curve(s2)
    s2rf = RealField(s2)

    @test domain(m1) === r2
    @test codomain(m1) === s2

    @test domain(s2c) === ManifoldsBase.ℝ
    @test codomain(s2c) === s2

    @test domain(s2rf) === s2
    @test codomain(s2rf) === ManifoldsBase.ℝ
end
