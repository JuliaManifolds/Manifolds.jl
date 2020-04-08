include("utils.jl")

@testset "Probability simplex" begin
    M = ProbabilitySimplex(2)
    p = [0.1, 0.7, 0.2]
    q = [0.3, 0.6, 0.1]
    X = zeros(3)
    Y = [-0.1, 0.05, 0.05]
    @test is_manifold_point(M,p)
    @test_throws DomainError is_manifold_point(M,p.+1, true)
    @test_throws DomainError is_manifold_point(M, [0], true)
    @test manifold_dimension(M) == 2
    @test is_tangent_vector(M,p,X)
    @test is_tangent_vector(M,p,Y)
    @test_throws DomainError is_tangent_vector(M,p.+1,X,true)
    @test_throws DomainError is_tangent_vector(M,p,zeros(4), true)
    @test_throws DomainError is_tangent_vector(M,p,Y.+1, true)

    types = [ Vector{Float64}, ]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_STATIC_SIZED && push!(types, MVector{3, Float64})

    basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
    for T in types
        @testset "Type $T" begin
            pts = [convert(T, [0.5, 0.3, 0.2]),
                   convert(T, [0.4, 0.4, 0.2]),
                   convert(T, [0.3, 0.5, 0.2])]
            test_manifold(
                M,
                pts,
                test_injectivity_radius = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vee_hat = false,
                test_forward_diff = false,
                test_reverse_diff = false,
                is_tangent_atol_multiplier = 5.0,
                inverse_retraction_methods = [SoftmaxInverseRetraction()],
                retraction_methods = [SoftmaxRetraction()]
            )
        end
    end

    @testset "Projection testing" begin
        p = [1/3, 1/3, 1/3]
        q = [0.2, 0.3, 0.5]
        X = log(M, p, q)
        X2 = X .+ 10
        Y = project(M, p, X2)
        @test isapprox(M, p, X, Y)
    end
end
