include("utils.jl")

@testset "CenteredMatrices" begin
    M = CenteredMatrices(3, 2)
    M_complex = CenteredMatrices(3, 2, ℂ)
    @test repr(M_complex) == "CenteredMatrices(3, 2, ℂ)"
    @test manifold_dimension(M_complex) == 8
    A = [1 2; 4 5; -5 -7]
    B = [1 2 3; 4 5 6; -5 -7 -9]    #wrong dimensions
    C = [-3 -im; 2 im; 1 0]         #complex
    D = [1 2; 3 4; 5 6]             #not centered 
    @testset "Real Centered Matrices Basics" begin
        @test repr(M) == "CenteredMatrices(3, 2, ℝ)"
        @test representation_size(M) == (3, 2)
        @test base_manifold(M) === M
        @test typeof(get_embedding(M)) === Euclidean{Tuple{3,2},ℝ}
        @test check_manifold_point(M, A) === nothing
        @test_throws DomainError is_manifold_point(M, B, true)
        @test_throws DomainError is_manifold_point(M, C, true)
        @test_throws DomainError is_manifold_point(M, D, true)
        @test check_tangent_vector(M, A, A) === nothing
        @test_throws DomainError is_tangent_vector(M, A, D, true)
        @test_throws DomainError is_tangent_vector(M, D, A, true)
        @test_throws DomainError is_tangent_vector(M, A, B, true)
        @test manifold_dimension(M) == 4
        @test A == project!(M, A, A)
        @test A == project(M, A, A)
        A2 = similar(A)
        embed!(M, A2, A)
        A3 = embed(M, A)
        @test A2 == A
        @test A3 == A
    end
    types = [Matrix{Float64}]
    E = [0 0; 1 -1; -1 1]
    F = [0.5 1; -1 -0.7; 0.5 -0.3]
    test_manifold(
                M,
                [A, E, F],
                test_injectivity_radius = false,
                test_reverse_diff = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
            )
end 