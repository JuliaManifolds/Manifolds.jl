include("utils.jl")

function test_arraymanifold()
    M = Sphere(2)
    A = ArrayManifold(M)
    x = [1., 0., 0.]
    y = 1/sqrt(2)*[1., 1., 0.]
    z = [0., 1., 0.]
    v = log(M,x,y)
    v2 = log(A,x,y)
    y2 = exp(A,x,v2)
    w = log(M,x,z)
    w2 = log(A,x,z; atol=10^(-15))
    @test isapprox(y2.value,y)
    @test distance(A,x,y) == distance(M,x,y)
    @test norm(A,x,v) == norm(M,x,v)
    @test inner(A,x,v2,w2; atol=10^(-15)) == inner(M,x,v,w)
    @test_throws DomainError is_manifold_point(M,2*y)
    @test_throws DomainError is_tangent_vector(M,y,v; atol=10^(-15))

    test_manifold(A, [x, y, z],
        test_tangent_vector_broadcasting = false)
end

@testset "Sphere" begin
    M = Sphere(2)
    types = [Vector{Float64},
             SizedVector{3, Float64},
             MVector{3, Float64},
             Vector{Float32},
             SizedVector{3, Float32},
             MVector{3, Float32},
             Vector{Double64},
             MVector{3, Double64},
             SizedVector{3, Double64}]
    for T in types
        @testset "Type $T" begin
            pts = [convert(T, [1.0, 0.0, 0.0]),
                   convert(T, [0.0, 1.0, 0.0]),
                   convert(T, [0.0, 0.0, 1.0])]
            test_manifold(M,
                          pts,
                          test_reverse_diff = isa(T, Vector),
                          test_project_tangent = true,
                          point_distributions = [Manifolds.uniform_distribution(M, pts[1])],
                          tvector_distributions = [Manifolds.normal_tvector_distribution(M, pts[1], 1.0)])

            @test isapprox(-pts[1], exp(M, pts[1], log(M, pts[1], -pts[1])))
        end
    end

    @testset "Distribution tests" begin
        usd_mvector = Manifolds.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
        @test isa(rand(usd_mvector), MVector)

        gtsd_mvector = Manifolds.normal_tvector_distribution(M, (@MVector [1.0, 0.0, 0.0]), 1.0)
        @test isa(rand(gtsd_mvector), MVector)
    end

    test_arraymanifold()
end
