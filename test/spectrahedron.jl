include("utils.jl")

@testset "Spectrahedron" begin
    M = Spectrahedron(4, 2)
    @test repr(M) == "Spectrahedron(4, 2)"
    @test manifold_dimension(M) == 6
    @test get_embedding(M) == Euclidean(4, 2)
    @test representation_size(M) == (4, 2)
    q = [1.0 0.0; 0.0 1.0; 1.0 1.0; -1.0 1.0]
    q = q / norm(q)
    @test is_manifold_point(M, q, true)
    @test base_manifold(M) === M
    qN = [2.0 0.0; 0.0 1.0; 1 / sqrt(2) -1 / sqrt(2); 1 / sqrt(2) 1 / sqrt(2)]
    @test_throws DomainError is_manifold_point(M, qN, true)
    Y = [0.0 1.0; 1.0 0.0; 0.0 0.0; 0.0 0.0]
    @test is_tangent_vector(M, q, Y, true; check_base_point = false)
    YN = [0.1 1.0; 1.0 0.1; 0.0 0.0; 0.0 0.0]
    @test_throws DomainError is_tangent_vector(M, q, YN, true; check_base_point = false)
    qE = similar(q)
    embed!(M, qE, q)
    qE2 = embed(M, q)
    @test qE == q
    @test qE2 == q
    q2 = [4.0 / 5 3.0 / 5; 3.0 / 5.0 -4.0 / 5.0; 1.0 0.0; 0.0 1.0]
    q2 = q2 ./ norm(q2)
    q3 = [12.0 / 13.0 5.0 / 13.0; 1.0 0.0; -12.0 / 13.0 5.0 / 13.0; 0.0 1.0]
    q3 = q3 ./ norm(q3)
    types = [Matrix{Float64}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{3,3,Float64,9})

    for T in types
        pts = [convert(T, q), convert(T, q2), convert(T, q3)]
        @testset "Type $T" begin
            test_manifold(
                M,
                pts,
                test_injectivity_radius = false,
                test_reverse_diff = false,
                test_forward_diff = false,
                test_project_tangent = true,
                test_exp_log = false,
                default_inverse_retraction_method = nothing,
                default_retraction_method = ProjectionRetraction(),
            )
        end
    end
end
