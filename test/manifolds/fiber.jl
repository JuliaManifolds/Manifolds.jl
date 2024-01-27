include("../header.jl")

using RecursiveArrayTools

struct TestVectorSpaceType <: VectorSpaceType end

@testset "spaces at point" begin
    M = Sphere(2)
    @testset "tangent and cotangent space" begin
        p = [1.0, 0.0, 0.0]
        t_p = TangentSpace(M, p)
        ct_p = CotangentSpace(M, p)
        t_ps = sprint(show, "text/plain", t_p)
        sp = sprint(show, "text/plain", p)
        sp = replace(sp, '\n' => "\n ")
        t_ps_test = "Tangent space to the manifold $(M) at point:\n $(sp)"
        @test t_ps == t_ps_test
        @test base_manifold(t_p) == M
        @test base_manifold(ct_p) == M
        @test t_p.manifold == M
        @test ct_p.manifold == M
        @test t_p.fiber_type == TangentSpaceType()
        @test ct_p.fiber_type == CotangentSpaceType()
        @test t_p.point == p
        @test ct_p.point == p
        @test injectivity_radius(t_p) == Inf
        @test representation_size(t_p) == representation_size(M)
        X = [0.0, 0.0, 1.0]
        @test embed(t_p, X) == X
        @test embed(t_p, X, X) == X
    end
end
