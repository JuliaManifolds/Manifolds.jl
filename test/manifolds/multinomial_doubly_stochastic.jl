include("../utils.jl")

@testset "Multinomial doubly stochastic Matrices" begin
    M = MultinomialDoubleStochastic(3)
    @test manifold_dimension(M) == 4
    @test repr(M) == "MultinomialDoubleStochastic(3)"
    p = ones(3, 3) ./ 3
    X = zeros(3, 3)
    @test is_point(M, p)
    @test is_vector(M, p, X)
    pf1 = [0.1 0.9 0.1; 0.1 0.9 0.1; 0.1 0.1 0.9] #not sum 1
    @test_throws ManifoldDomainError is_point(M, pf1, true)
    pf2r = [0.1 0.9 0.1; 0.8 0.05 0.15; 0.1 0.05 0.75]
    @test_throws DomainError is_point(M, pf2r, true)
    @test_throws ManifoldDomainError is_point(M, pf2r', true)
    pf3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] # contains nonpositive entries
    @test_throws ManifoldDomainError is_point(M, pf3, true)
    Xf2c = [-0.1 0.0 0.1; -0.2 0.1 0.1; 0.2 -0.1 -0.1] #nonzero columns
    @test_throws ManifoldDomainError is_vector(M, p, Xf2c, true)
    @test_throws DomainError is_vector(M, p, Xf2c', true)
    @test representation_size(M) == (3, 3)
    @test !is_flat(M)
    pE = similar(p)
    embed!(M, pE, p)
    pE2 = embed(M, p)
    @test pE == p
    @test pE2 == p
    @test_throws DomainError project(M, -ones(3, 3))
    @test project(M, p) == p
    p2 = [0.1 0.2 0.7; 0.2 0.7 0.1; 0.7 0.1 0.2]
    p3 = [0.1 0.4 0.5; 0.4 0.5 0.1; 0.5 0.1 0.4]

    X2 = [-0.1 0.0 0.1; 0.0 0.2 -0.2; 0.1 -0.2 0.1]
    @test is_vector(
        M,
        p2,
        vector_transport_to(M, p, X2, p2, ProjectionTransport());
        atol=10^-15,
    )
    X3 = [1.0 1.0 1.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    @test inner(M, p, X3, X3) == 9.0

    types = [Matrix{Float64}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{4,2,Float64,8})

    for T in types
        pts = [convert(T, p), convert(T, p2), convert(T, p3)]
        @testset "Type $T" begin
            test_manifold(
                M,
                pts,
                test_injectivity_radius=false,
                test_project_tangent=true,
                test_exp_log=false,
                test_default_vector_transport=true,
                vector_transport_methods=[ProjectionTransport()],
                default_inverse_retraction_method=nothing,
                default_retraction_method=ProjectionRetraction(),
                is_tangent_atol_multiplier=20,
                is_point_atol_multiplier=20,
                test_inplace=true,
            )
        end
    end
end
