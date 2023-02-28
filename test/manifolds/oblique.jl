include("../utils.jl")

@testset "Oblique manifold" begin
    M = Oblique(3, 2)
    @testset "Oblique manifold Basics" begin
        @test Sphere(2)^3 === Oblique(3, 3)
        @test Sphere(2)^(3,) === PowerManifold(Sphere(2), 3)
        @test ^(Sphere(2), 2) === Oblique(3, 2)
        @test typeof(^(Sphere(2), 2)) == Oblique{3,2,ℝ,2}
        @test repr(M) == "Oblique(3,2; field = ℝ)"
        @test representation_size(M) == (3, 2)
        @test manifold_dimension(M) == 4
        @test !is_flat(M)
        p = [2, 0, 0]
        p2 = [p p]
        @test !is_point(M, p)
        @test_throws DomainError is_point(M, p, true)
        @test !is_point(M, p2)
        @test_throws CompositeManifoldError is_point(M, p2, true)
        @test !is_vector(M, p2, 0.0)
        @test_throws CompositeManifoldError{ComponentManifoldError{Int64,DomainError}} is_vector(
            M,
            p2,
            [0.0, 0.0, 0.0],
            true,
        )
        @test !is_vector(M, p2, [0.0, 0.0, 0.0])
        @test_throws DomainError is_vector(M, p, [0.0, 0.0, 0.0], true) # p wrong
        @test injectivity_radius(M) ≈ π
        x = [1.0 0.0 0.0; 1.0 0.0 0.0]'
        @test_throws DomainError is_vector(M, x, [0.0, 0.0, 0.0], true) # tangent wrong
        y = [1.0 0.0 0.0; 1/sqrt(2) 1/sqrt(2) 0.0]'
        z = [1/sqrt(2) 1/sqrt(2) 0.0; 1.0 0.0 0.0]'
        basis_types = (DefaultOrthonormalBasis(),)
        transports = [ParallelTransport()]
        test_manifold(
            M,
            [x, y, z],
            test_vector_spaces=true,
            test_project_tangent=false,
            test_musical_isomorphisms=true,
            test_default_vector_transport=true,
            vector_transport_methods=transports,
            basis_types_to_from=basis_types,
            exp_log_atol_multiplier=1,
            test_inplace=true,
        )
    end
end
