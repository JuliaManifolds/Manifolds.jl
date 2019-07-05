include("utils.jl")

@testset "Product manifold" begin
    M1 = Sphere(2)
    M2 = Euclidean(2)
    Mse = ProductManifold(M1, M2)
    @test Mse == M1 × M2
    shape_se = Manifolds.ShapeSpecification(M1, M2)

    types = [Vector{Float64},
             SizedVector{5, Float64},
             MVector{5, Float64},
             Vector{Float32},
             SizedVector{5, Float32},
             MVector{5, Float32}]

    retraction_methods = [Manifolds.ProductRetraction(Manifolds.ExponentialRetraction(), Manifolds.ExponentialRetraction())]
    inverse_retraction_methods = [Manifolds.InverseProductRetraction(Manifolds.LogarithmicInverseRetraction(), Manifolds.LogarithmicInverseRetraction())]
    for T in types
        @testset "Type $T" begin
            pts_base = [convert(T, [1.0, 0.0, 0.0, 0.0, 0.0]),
                        convert(T, [0.0, 1.0, 0.0, 1.0, 0.0]),
                        convert(T, [0.0, 0.0, 1.0, 0.0, 0.1])]
            pts = map(p -> Manifolds.ProductArray(shape_se, p), pts_base)
            distr_M1 = Manifolds.uniform_distribution(M1, pts_base[1][1:3])
            distr_M2 = Manifolds.projected_distribution(M2, Distributions.MvNormal(zero(pts_base[1][4:5]), 1.0))
            distr_tv_M1 = Manifolds.normal_tvector_distribution(M1, pts_base[1][1:3], 1.0)
            distr_tv_M2 = Manifolds.normal_tvector_distribution(M2, pts_base[1][4:5], 1.0)
            test_manifold(Mse,
                          pts;
                          test_reverse_diff = isa(T, Vector),
                          retraction_methods = retraction_methods,
                          inverse_retraction_methods = inverse_retraction_methods,
                          point_distributions = [Manifolds.ProductPointDistribution(distr_M1, distr_M2)],
                          tvector_distributions = [Manifolds.ProductTVectorDistribution(distr_tv_M1, distr_tv_M2)])
        end
    end

    M3 = Manifolds.Rotations(2)
    Mser = ProductManifold(M1, M2, M3)
    shape_ser = Manifolds.ShapeSpecification(M1, M2, M3)

    @test submanifold(Mser, 2) == M2
    @test submanifold(Mser, Val((1, 3))) == M1 × M3
    @test submanifold(Mser, 2:3) == M2 × M3
    @test submanifold(Mser, [1, 3]) == M1 × M3
    @inferred submanifold(Mser, Val((1, 3)))

    # testing the slower generic constructor
    Mprod4 = ProductManifold(M2, M2, M2, M2)
    shape4 = Manifolds.ShapeSpecification(Mprod4.manifolds...)
    data_x4 = rand(8)
    @test Manifolds.ProductArray(shape4, data_x4).data === data_x4

    pts_sphere = [[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]
    pts_r2 = [[0.0, 0.0],
              [1.0, 0.0],
              [0.0, 0.1]]
    angles = (0.0, π/2, 2π/3)
    pts_rot = [[cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)] for ϕ in angles]
    pts = [Manifolds.prod_point(shape_ser, p[1], p[2], p[3]) for p in zip(pts_sphere, pts_r2, pts_rot)]
    test_manifold(Mser,
                  pts,
                  test_forward_diff = false,
                  test_reverse_diff = false)

    @testset "prod_point" begin
        Ts = SizedVector{3, Float64}
        Tr2 = SizedVector{2, Float64}
        T = SizedVector{5, Float64}
        pts_base = [convert(T, [1.0, 0.0, 0.0, 0.0, 0.0]),
                    convert(T, [0.0, 1.0, 0.0, 1.0, 0.0]),
                    convert(T, [0.0, 0.0, 1.0, 0.0, 0.1])]
        pts = map(p -> Manifolds.ProductArray(shape_se, p), pts_base)
        pts_sphere = [convert(Ts, [1.0, 0.0, 0.0]),
                      convert(Ts, [0.0, 1.0, 0.0]),
                      convert(Ts, [0.0, 0.0, 1.0])]
        pts_r2 = [convert(Tr2, [0.0, 0.0]),
                  convert(Tr2, [1.0, 0.0]),
                  convert(Tr2, [0.0, 0.1])]
        pts_prod = [Manifolds.prod_point(shape_se, p[1], p[2]) for p in zip(pts_sphere, pts_r2)]
        for p in zip(pts, pts_prod)
            @test isapprox(Mse, p[1], p[2])
        end
        for p in zip(pts_sphere, pts_r2, pts_prod)
            @test isapprox(M1, p[1], Manifolds.submanifold_component(p[3], 1))
            @test isapprox(M2, p[2], Manifolds.submanifold_component(p[3], 2))
        end
    end

    @testset "ProductMPoint" begin
        Ts = SizedVector{3, Float64}
        Tr2 = SizedVector{2, Float64}
        pts_sphere = [convert(Ts, [1.0, 0.0, 0.0]),
                      convert(Ts, [0.0, 1.0, 0.0]),
                      convert(Ts, [0.0, 0.0, 1.0])]
        pts_r2 = [convert(Tr2, [0.0, 0.0]),
                  convert(Tr2, [1.0, 0.0]),
                  convert(Tr2, [0.0, 0.1])]

        pts = [Manifolds.ProductMPoint(p[1], p[2]) for p in zip(pts_sphere, pts_r2)]
        test_manifold(Mse,
                      pts,
                      test_tangent_vector_broadcasting = false,
                      test_forward_diff = false,
                      test_reverse_diff = false)
    end
end
