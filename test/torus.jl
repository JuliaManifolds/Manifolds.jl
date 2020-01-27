include("utils.jl")

@testset "Torus" begin
    M = Torus(2)
    @testset "Torus Basics" begin
        @test representation_size(M) == (2,)
        @test manifold_dimension(M) == 2
        @test !is_manifold_point(M, 9.)
        @test_throws DomainError is_manifold_point(M, 9., true)
        @test !is_manifold_point(M, [9.;9.])
        @test_throws DomainError is_manifold_point(M, [9. 9.], true)
        @test !is_tangent_vector(M, [9.;9.], 0.)
        @test_throws DomainError is_tangent_vector(M, 9., 0., true)
        @test !is_tangent_vector(M, [9.;9.], [0.;0.])
        @test_throws DomainError is_tangent_vector(M, 9., 0., true)
        @test injectivity_radius(M) ≈ π
        x = [1.0, 2.0]
        y = [-1.0, 2.0]
        z = [0., 0.]
        basis_types = (ArbitraryOrthonormalBasis(),)
        test_manifold(M,
            [x, y, z],
            test_forward_diff = false,
            test_reverse_diff = false,
            test_vector_spaces = true,
            test_project_tangent = false,
            test_musical_isomorphisms = true,
            test_vector_transport = false,
            basis_types_to_from = basis_types,
        )
    end
end
