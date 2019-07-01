include("utils.jl")

@testset "Product manifold" begin
    M1 = Manifolds.Sphere(2)
    M2 = Manifolds.Euclidean(2)
    Mse = Manifolds.ProductManifold(M1, M2)
    shape_se = Manifolds.ShapeSpecification(M1, M2)

    types = [Vector{Float64},
             SizedVector{5, Float64},
             MVector{5, Float64},
             Vector{Float32},
             SizedVector{5, Float32},
             MVector{5, Float32}]

    for T in types
        @testset "Type $T" begin
            pts_base = [convert(T, [1.0, 0.0, 0.0, 0.0, 0.0]),
                        convert(T, [0.0, 1.0, 0.0, 1.0, 0.0]),
                        convert(T, [0.0, 0.0, 1.0, 0.0, 0.1])]
            pts = map(p -> Manifolds.ProductArray(shape_se, p), pts_base)
            test_manifold(Mse,
                          pts,
                          test_reverse_diff = isa(T, Vector))
        end
    end

    M3 = Manifolds.Rotations(2)
    Mser = Manifolds.ProductManifold(M1, M2, M3)
    shape_ser = Manifolds.ShapeSpecification(M1, M2, M3)

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
            @test isapprox(M1, p[1], Manifolds.proj_product(p[3], 1))
            @test isapprox(M2, p[2], Manifolds.proj_product(p[3], 2))
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
