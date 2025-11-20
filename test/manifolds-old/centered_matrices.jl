include("../header.jl")

@testset "CenteredMatrices" begin
    M = CenteredMatrices(3, 2)
    M_complex = CenteredMatrices(3, 2, ℂ)
    A = [1 2; 4 5; -5 -7]
    B = [1 2 3; 4 5 6; -5 -7 -9]    #wrong dimensions
    C = [-3 -im; 2 im; 1 0]         #complex
    D = [1 2; 3 4; 5 6]             #not centered
    types = [Matrix{Float64}]
    @testset "Complex Centered Matrices Basics" begin
        @test repr(M_complex) == "CenteredMatrices(3, 2, ℂ)"
        @test manifold_dimension(M_complex) == 8
        G = [1.0 1.0im; -1.0im 0.0; -1.0 + 1.0im -1.0im]
        H = [1.0im 0.0; -2.0im 1.0im; 1.0im -1.0im]
        Manifolds.test_manifold(
            M_complex,
            [C, G, H],
            test_injectivity_radius = false,
            test_project_tangent = true,
            test_musical_isomorphisms = true,
            test_default_vector_transport = true,
            is_tangent_atol_multiplier = 1,
            is_point_atol_multiplier = 1,
            test_inplace = true,
        )
    end
    @testset "field parameter" begin
        M = CenteredMatrices(3, 2; parameter = :field)
        @test repr(M) == "CenteredMatrices(3, 2, ℝ; parameter=:field)"
        @test typeof(get_embedding(M)) === Euclidean{ℝ, Tuple{Int, Int}}
    end
end
