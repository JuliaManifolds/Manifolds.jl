include("../header.jl")

@testset "Multinomial symmetric matrices" begin
    M = MultinomialSymmetric(3)
    @test manifold_dimension(M) == 3
    @test repr(M) == "MultinomialSymmetric(3)"
    p = ones(3, 3) ./ 3
    X = zeros(3, 3)
    @test is_point(M, p)
    @test is_vector(M, p, X)
    pf1 = [0.1 0.9 0.1; 0.1 0.9 0.1; 0.1 0.1 0.9] #not symmetric
    @test_throws ManifoldDomainError is_point(M, pf1; error=:error)
    pf2 = [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.9] # cols do not sum to 1
    @test_throws ManifoldDomainError is_point(M, pf2; error=:error)
    pf3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] # contains nonpositive entries
    @test_throws ManifoldDomainError is_point(M, pf3; error=:error)
    Xf1 = [0.0 1.0 -1.0; 0.0 0.0 0.0; 0.0 0.0 0.0] # not symmetric
    @test_throws ManifoldDomainError is_vector(M, p, Xf1; error=:error)
    Xf2 = [0.0 -1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 0.0] # nonzero sums
    @test_throws ManifoldDomainError is_vector(M, p, Xf2; error=:error)
    @test representation_size(M) == (3, 3)
    @test !is_flat(M)
    pE = similar(p)
    embed!(M, pE, p)
    pE2 = embed(M, p)
    @test pE == p
    @test pE2 == p
    XE = similar(X)
    embed!(M, XE, p, X)
    XE2 = embed(M, p, X)
    @test XE == X
    @test XE2 == X
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
                retraction_methods=[ProjectionRetraction()],
                default_inverse_retraction_method=nothing,
                default_retraction_method=ProjectionRetraction(),
                is_tangent_atol_multiplier=20,
                is_point_atol_multiplier=20,
                test_inplace=true,
            )
        end
    end
    @testset "field parameter" begin
        M = MultinomialSymmetric(3; parameter=:field)
        @test repr(M) == "MultinomialSymmetric(3; parameter=:field)"
        @test get_embedding(M) === MultinomialMatrices(3, 3; parameter=:field)
    end
    @testset "random" begin
        Random.seed!(42)
        p = rand(M)
        @test is_point(M, p)
        X = rand(M; vector_at=p)
        @test is_vector(M, p, X)
    end
    @testset "Hessian call" begin
        p = ones(3, 3) ./ 3
        Y = one(p)
        G = zero(p)
        H = 0.5 * one(p)
        X = riemannian_Hessian(M, p, G, H, Y)
        X2 = similar(X)
        riemannian_Hessian!(M, X2, p, G, H, Y)
        @test isapprox(M, p, X, X2)
        @test is_vector(M, p, X)
    end
end
