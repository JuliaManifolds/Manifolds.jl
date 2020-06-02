include("utils.jl")

@testset "SphereSymmetricMatrices" begin
    M = SphereSymmetricMatrices(3)
    A = [1 2 3; 2 4 -5; 3 -5 6] / norm([1 2 3; 2 4 -5; 3 -5 6])
    B = [1 2; 2 4] / norm([1 2; 2 4])                                     #wrong dimensions
    C = [-3 -im 4; -im 5 im; 4 im 0] / norm([-3 -im 4; -im 5 im; 4 im 0]) #complex
    D = [1 2 3; 4 5 6; 7 8 9] / norm([1 2 3; 4 5 6; 7 8 9])               #not symmetric
    E = [1 2 3; 2 4 -5; 3 -5 6]                                         #not of unit norm
    @testset "Sphere Symmetric Matrices Basics" begin
        @test repr(M) == "SphereSymmetricMatrices(3, ℝ)"
        @test representation_size(M) == (3, 3)
        @test base_manifold(M) === M
        @test typeof(get_embedding(M)) === ArraySphere{Tuple{3,3},ℝ}
        @test check_manifold_point(M, A) === nothing
        @test_throws DomainError is_manifold_point(M, B, true)
        @test_throws DomainError is_manifold_point(M, C, true)
        @test_throws DomainError is_manifold_point(M, D, true)
        @test_throws DomainError is_manifold_point(M, E, true)
        @test check_tangent_vector(M, A, zeros(3, 3)) === nothing
        @test_throws DomainError is_tangent_vector(M, A, B, true)
        @test_throws DomainError is_tangent_vector(M, A, C, true)
        @test_throws DomainError is_tangent_vector(M, A, D, true)
        @test_throws DomainError is_tangent_vector(M, D, A, true)
        @test_throws DomainError is_tangent_vector(M, A, E, true)
        @test manifold_dimension(M) == 5
        A2 = similar(A)
        @test A == project!(M, A2, A)
        @test A == project(M, A)
        embed!(M, A2, A)
        A3 = embed(M, A)
        @test A2 == A
        @test A3 == A
    end
    F = [2 4 -1; 4 9 7; -1 7 5] / norm([2 4 -1; 4 9 7; -1 7 5])
    G = [-10 9 -8; 9 7 6; -8 6 5] / norm([-10 9 -8; 9 7 6; -8 6 5])
    test_manifold(
        M,
        [A, F, G],
        test_injectivity_radius = false,
        test_forward_diff = false,
        test_reverse_diff = false,
        test_vector_spaces = true,
        test_project_tangent = true,
        test_musical_isomorphisms = true,
        test_vector_transport = true,
        is_tangent_atol_multiplier = 2,
        #is_point_atol_multilplier = 1,
    )
end
