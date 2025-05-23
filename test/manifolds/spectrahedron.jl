include("../header.jl")

@testset "Spectrahedron" begin
    M = Spectrahedron(4, 2)
    @test repr(M) == "Spectrahedron(4, 2)"
    @test manifold_dimension(M) == 6
    @test get_embedding(M) == Euclidean(4, 2)
    @test representation_size(M) == (4, 2)
    @test !is_flat(M)
    q = [1.0 0.0; 0.0 1.0; 1.0 1.0; -1.0 1.0]
    q = q / norm(q)
    @test is_point(M, q; error=:error)
    @test base_manifold(M) === M
    qN = [2.0 0.0; 0.0 1.0; 1/sqrt(2) -1/sqrt(2); 1/sqrt(2) 1/sqrt(2)]
    @test_throws DomainError is_point(M, qN; error=:error)
    Y = [0.0 1.0; 1.0 0.0; 0.0 0.0; 0.0 0.0]
    @test is_vector(M, q, Y; error=:error)
    YN = [0.1 1.0; 1.0 0.1; 0.0 0.0; 0.0 0.0]
    @test_throws DomainError is_vector(M, q, YN; error=:error)
    qE = similar(q)
    embed!(M, qE, q)
    qE2 = embed(M, q)
    @test qE == q
    @test qE2 == q
    q2 = [4.0/5 3.0/5; 3.0/5.0 -4.0/5.0; 1.0 0.0; 0.0 1.0]
    q2 = q2 ./ norm(q2)
    q3 = [12.0/13.0 5.0/13.0; 1.0 0.0; -12.0/13.0 5.0/13.0; 0.0 1.0]
    q3 = q3 ./ norm(q3)
    @test is_vector(
        M,
        q2,
        vector_transport_to(M, q, Y, q2, ProjectionTransport());
        atol=10^-15,
    )

    types = [Matrix{Float64}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{4,2,Float64,8})

    for T in types
        pts = [convert(T, q), convert(T, q2), convert(T, q3)]
        @testset "Type $T" begin
            test_manifold(
                M,
                pts,
                test_injectivity_radius=false,
                test_project_tangent=true,
                test_exp_log=false,
                test_default_vector_transport=true,
                vector_transport_methods=[ProjectionTransport()],
                retraction_methods=[ProjectionRetraction()],
                default_inverse_retraction_method=nothing,
                default_retraction_method=ProjectionRetraction(),
                test_inplace=true,
            )
        end
    end
    @testset "field parameter" begin
        M = Spectrahedron(4, 2; parameter=:field)
        @test typeof(get_embedding(M)) === Euclidean{Tuple{Int,Int},ℝ}
        @test repr(M) == "Spectrahedron(4, 2; parameter=:field)"
    end
end
