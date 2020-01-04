include("utils.jl")

@testset "Sphere" begin
    M = Sphere(2)
    @testset "Sphere Basics" begin
        @test representation_size(M) == (3,)
        @test !is_manifold_point(M, [1., 0., 0., 0.])
        @test !is_tangent_vector(M, [1.,0.,0.], [0., 0., 1., 0.])
        @test_throws DomainError is_manifold_point(M, [2.,0.,0.],true)
        @test !is_manifold_point(M, [2.,0.,0.])
        @test !is_tangent_vector(M,[1.,0.,0.],[1.,0.,0.])
        @test_throws DomainError is_tangent_vector(M,[1.,0.,0.],[1.,0.,0.],true)
    end
    types = [
        Vector{Float64},
        MVector{3, Float64},
        Vector{Float32},
    ]
    for T in types
        @testset "Type $T" begin
            pts = [convert(T, [1.0, 0.0, 0.0]),
                   convert(T, [0.0, 1.0, 0.0]),
                   convert(T, [0.0, 0.0, 1.0])]
            test_manifold(
                M,
                pts,
                test_reverse_diff = isa(T, Vector),
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                test_mutating_rand = isa(T, Vector),
                point_distributions = [Manifolds.uniform_distribution(M, pts[1])],
                tvector_distributions = [Manifolds.normal_tvector_distribution(M, pts[1], 1.0)],
            )

            @test isapprox(-pts[1], exp(M, pts[1], log(M, pts[1], -pts[1])))
        end
    end

    @testset "Distribution tests" begin
        usd_mvector = Manifolds.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
        @test isa(rand(usd_mvector), MVector)

        gtsd_mvector = Manifolds.normal_tvector_distribution(M, (@MVector [1.0, 0.0, 0.0]), 1.0)
        @test isa(rand(gtsd_mvector), MVector)
    end

    @testset "log edge case" begin
        n = manifold_dimension(M)
        x = normalize(randn(n + 1))
        v = log(M, x, -x)
        @test norm(v) ≈ π
        @test isapprox(dot(x, v), 0; atol=1e-12)
        vexp = normalize(project_tangent(M, x, [1, zeros(n)...]))
        @test v ≈ π * vexp

        x = [1, zeros(n)...]
        v = log(M, x, -x)
        @test norm(v) ≈ π
        @test isapprox(dot(x, v), 0; atol=1e-12)
        vexp = normalize(project_tangent(M, x, [0, 1, zeros(n - 1)...]))
        @test v ≈ π * vexp
    end

end
