include("../utils.jl")

using RecursiveArrayTools

struct TestVectorSpaceType <: VectorSpaceType end

@testset "spaces at point" begin
    M = Sphere(2)
    @testset "tangent and cotangent space" begin
        p = [1.0, 0.0, 0.0]
        t_p = TangentSpaceAtPoint(M, p)
        t_p2 = TangentSpace(M, p)
        @test t_p == t_p2
        ct_p = CotangentSpaceAtPoint(M, p)
        t_ps = sprint(show, "text/plain", t_p)
        sp = sprint(show, "text/plain", p)
        sp = replace(sp, '\n' => "\n ")
        t_ps_test = "Tangent space to the manifold $(M) at point:\n $(sp)"
        @test t_ps == t_ps_test
        @test base_manifold(t_p) == M
        @test base_manifold(ct_p) == M
        @test t_p.fiber.manifold == M
        @test ct_p.fiber.manifold == M
        @test t_p.fiber.fiber == Manifolds.TangentFiber
        @test ct_p.fiber.fiber == Manifolds.CotangentFiber
        @test t_p.point == p
        @test ct_p.point == p
        @test injectivity_radius(t_p) == Inf
        @test representation_size(t_p) == representation_size(M)
        X = [0.0, 0.0, 1.0]
        @test embed(t_p, X) == X
        @test embed(t_p, X, X) == X
        # generic vector space at
        fiber = VectorBundleFibers(TestVectorSpaceType(), M)
        X_p = Manifolds.FiberAtPoint(fiber, p)
        X_ps = sprint(show, "text/plain", X_p)
        fiber_s = sprint(show, "text/plain", fiber)
        X_ps_test = "$(typeof(X_p))\nFiber:\n $(fiber_s)\nBase point:\n $(sp)"
        @test X_ps == X_ps_test
        @test_throws MethodError project(fiber, p, X)
        @test_throws MethodError norm(fiber, p, X)
        @test_throws MethodError distance(fiber, p, X, X)
    end
end
