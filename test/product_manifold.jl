include("utils.jl")

struct NotImplementedReshaper <: Manifolds.AbstractReshaper end

@testset "Product manifold" begin
    @test_throws MethodError ProductManifold()
    M1 = Sphere(2)
    M2 = Euclidean(2)
    Mse = ProductManifold(M1, M2)
    @test Mse == M1 × M2
    @test Mse == ProductManifold(M1) × M2
    @test Mse == ProductManifold(M1) × ProductManifold(M2)
    @test Mse == M1 × ProductManifold(M2)
    @test injectivity_radius(Mse) ≈ π
    @test is_default_metric(Mse, ProductMetric())
    @test Manifolds.default_metric_dispatch(Mse, ProductMetric()) === Val{true}()
    @test_throws ErrorException Manifolds.make_reshape(NotImplementedReshaper(), Int64, zeros(2,3))
    types = [
        Vector{Float64},
        MVector{5, Float64},
    ]
    TEST_FLOAT32 && push!(types, Vector{Float32})

    retraction_methods = [Manifolds.ProductRetraction(ManifoldsBase.ExponentialRetraction(), ManifoldsBase.ExponentialRetraction())]
    inverse_retraction_methods = [Manifolds.InverseProductRetraction(ManifoldsBase.LogarithmicInverseRetraction(), ManifoldsBase.LogarithmicInverseRetraction())]

    reshapers = (Manifolds.StaticReshaper(), Manifolds.ArrayReshaper())

    @testset "Show methods" begin
        Mse2 = ProductManifold(M1, M1, M2, M2)
        @test sprint(show, Mse2) == "ProductManifold($(M1), $(M1), $(M2), $(M2))"
        withenv("LINES" => 10, "COLUMNS" => 100) do
            @test sprint(show, "text/plain", ProductManifold(M1)) == "ProductManifold with 1 submanifold:\n $(M1)"
            @test sprint(show, "text/plain", Mse2) == "ProductManifold with 4 submanifolds:\n $(M1)\n $(M1)\n $(M2)\n $(M2)"
        end
        withenv("LINES" => 7, "COLUMNS" => 100) do
            @test sprint(show, "text/plain", Mse2) == "ProductManifold with 4 submanifolds:\n $(M1)\n ⋮\n $(M2)"
        end

        @test sprint(show, "text/plain", ProductManifold(Mse, Mse)) == """
        ProductManifold with 2 submanifolds:
         ProductManifold(Sphere(2), Euclidean(2; field = ℝ))
         ProductManifold(Sphere(2), Euclidean(2; field = ℝ))"""

        shape_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M1)
        p = Manifolds.ProductArray(shape_se, Float64[1, 0, 0])
        @test sprint(show, "text/plain", p) == """
        ProductArray with 1 submanifold component:
         Component 1 =
          3-element view(::Array{Float64,1}, 1:3) with eltype Float64:
           1.0
           0.0
           0.0"""

        shape_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M1, M1, M2, M2)
        p = Manifolds.ProductArray(shape_se, Float64[1, 0, 0, 0, 1, 0, 1, 2, 3, 4])
        @test sprint(show, "text/plain", p) == """
        ProductArray with 4 submanifold components:
         Component 1 =
          3-element view(::Array{Float64,1}, 1:3) with eltype Float64:
           1.0
           0.0
           0.0
         Component 2 =
          3-element view(::Array{Float64,1}, 4:6) with eltype Float64:
           0.0
           1.0
           0.0
         Component 3 =
          2-element view(::Array{Float64,1}, 7:8) with eltype Float64:
           1.0
           2.0
         Component 4 =
          2-element view(::Array{Float64,1}, 9:10) with eltype Float64:
           3.0
           4.0"""

        shape_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M1, M1, M2, M2, M2)
        p = Manifolds.ProductArray(shape_se, Float64[1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5, 6])
        @test sprint(show, "text/plain", p) == """
        ProductArray with 5 submanifold components:
         Component 1 =
          3-element view(::Array{Float64,1}, 1:3) with eltype Float64:
           1.0
           0.0
           0.0
         Component 2 =
          3-element view(::Array{Float64,1}, 4:6) with eltype Float64:
           0.0
           1.0
           0.0
         ⋮
         Component 4 =
          2-element view(::Array{Float64,1}, 9:10) with eltype Float64:
           3.0
           4.0
         Component 5 =
          2-element view(::Array{Float64,1}, 11:12) with eltype Float64:
           5.0
           6.0"""

       p = Manifolds.ProductRepr(Float64[1, 0, 0])
       @test sprint(show, "text/plain", p) == """
       ProductRepr with 1 submanifold component:
        Component 1 =
         3-element Array{Float64,1}:
          1.0
          0.0
          0.0"""

        p = Manifolds.ProductRepr(Float64[1, 0, 0], Float64[0, 1, 0], Float64[1, 2], Float64[3, 4])
        @test sprint(show, "text/plain", p) == """
        ProductRepr with 4 submanifold components:
         Component 1 =
          3-element Array{Float64,1}:
           1.0
           0.0
           0.0
         Component 2 =
          3-element Array{Float64,1}:
           0.0
           1.0
           0.0
         Component 3 =
          2-element Array{Float64,1}:
           1.0
           2.0
         Component 4 =
          2-element Array{Float64,1}:
           3.0
           4.0"""

        p = Manifolds.ProductRepr(Float64[1, 0, 0], Float64[0, 1, 0], Float64[1, 2], Float64[3, 4], Float64[5, 6])
        @test sprint(show, "text/plain", p) == """
        ProductRepr with 5 submanifold components:
         Component 1 =
          3-element Array{Float64,1}:
           1.0
           0.0
           0.0
         Component 2 =
          3-element Array{Float64,1}:
           0.0
           1.0
           0.0
         ⋮
         Component 4 =
          2-element Array{Float64,1}:
           3.0
           4.0
         Component 5 =
          2-element Array{Float64,1}:
           5.0
           6.0"""
    end

    for T in types, reshaper in reshapers
        if reshaper == reshapers[2] && T != Vector{Float64}
            continue
        end
        shape_se = Manifolds.ShapeSpecification(reshaper, M1, M2)
        @testset "Type $T" begin
            pts_base = [convert(T, [1.0, 0.0, 0.0, 0.0, 0.0]),
                        convert(T, [0.0, 1.0, 0.0, 1.0, 0.0]),
                        convert(T, [0.0, 0.0, 1.0, 0.0, 0.1])]
            pts = map(p -> Manifolds.ProductArray(shape_se, p), pts_base)
            distr_M1 = Manifolds.uniform_distribution(M1, pts_base[1][1:3])
            distr_M2 = Manifolds.projected_distribution(M2, Distributions.MvNormal(zero(pts_base[1][4:5]), 1.0))
            distr_tv_M1 = Manifolds.normal_tvector_distribution(M1, pts_base[1][1:3], 1.0)
            distr_tv_M2 = Manifolds.normal_tvector_distribution(M2, pts_base[1][4:5], 1.0)
            @test injectivity_radius(Mse, pts[1]) ≈ π
            @test injectivity_radius(Mse) ≈ π
            @test injectivity_radius(Mse, pts[1], ExponentialRetraction()) ≈ π
            test_manifold(
                Mse,
                pts;
                test_reverse_diff = isa(T, Vector),
                test_musical_isomorphisms = true,
                test_injectivity_radius = false,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                test_mutating_rand = isa(T, Vector),
                point_distributions = [Manifolds.ProductPointDistribution(distr_M1, distr_M2)],
                tvector_distributions = [Manifolds.ProductFVectorDistribution(distr_tv_M1, distr_tv_M2)])
        end
    end

    M3 = Manifolds.Rotations(2)
    Mser = ProductManifold(M1, M2, M3)
    shape_ser = Manifolds.ShapeSpecification(reshapers[1], M1, M2, M3)

    @testset "ShapeSpecification detailed reshapers" begin
        shape_ser_3_test = Manifolds.ShapeSpecification((reshapers[1], reshapers[2], reshapers[1]), M1, M2, M3)
        @test shape_ser_3_test.reshapers[1] == reshapers[1]
        @test shape_ser_3_test.reshapers[2] == reshapers[2]
        @test shape_ser_3_test.reshapers[3] == reshapers[1]
    end

    @testset "high-dimensional product manifold" begin
        Mhigh = Euclidean(100000)
        shape_high = Manifolds.ShapeSpecification(reshapers[2], M1, Mhigh)
        a = Manifolds.ProductArray(shape_high, collect(1.0:100003.0))
        b = a.parts[2].^2
        @test b[1] == 16
    end

    @test submanifold(Mser, 2) == M2
    @test submanifold(Mser, Val((1, 3))) == M1 × M3
    @test submanifold(Mser, 2:3) == M2 × M3
    @test submanifold(Mser, [1, 3]) == M1 × M3
    @inferred submanifold(Mser, Val((1, 3)))

    # testing the slower generic constructor
    Mprod4 = ProductManifold(M2, M2, M2, M2)
    shape4 = Manifolds.ShapeSpecification(reshapers[1], Mprod4.manifolds...)
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
    test_manifold(
        Mser,
        pts,
        test_injectivity_radius = false,
        test_forward_diff = false,
        test_reverse_diff = false
    )

    @testset "prod_point" begin
        shape_se = Manifolds.ShapeSpecification(reshapers[1], M1, M2)
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
            @test isapprox(M1, p[1], submanifold_component(p[3], 1))
            @test isapprox(M2, p[2], submanifold_component(p[3], 2))
        end
        @test submanifold_component(Mse, pts[1], 1) === pts[1].parts[1]
        @test submanifold_component(Mse, pts[1], Val(1)) === pts[1].parts[1]
        @test submanifold_component(pts[1], 1) === pts[1].parts[1]
        @test submanifold_component(pts[1], Val(1)) === pts[1].parts[1]
        @test submanifold_components(Mse, pts[1]) === pts[1].parts
        @test submanifold_components(pts[1]) === pts[1].parts
    end

    @testset "ProductRepr" begin

        Ts = SizedVector{3, Float64}
        Tr2 = SizedVector{2, Float64}
        pts_sphere = [convert(Ts, [1.0, 0.0, 0.0]),
                      convert(Ts, [0.0, 1.0, 0.0]),
                      convert(Ts, [0.0, 0.0, 1.0])]
        pts_r2 = [convert(Tr2, [0.0, 0.0]),
                  convert(Tr2, [1.0, 0.0]),
                  convert(Tr2, [0.0, 0.1])]

        pts = [ProductRepr(p[1], p[2]) for p in zip(pts_sphere, pts_r2)]
        basis_types = (
            DefaultOrthonormalBasis(),
            ProjectedOrthonormalBasis(:svd),
            get_basis(Mse, pts[1], DefaultOrthonormalBasis()),
            DiagonalizingOrthonormalBasis(ProductRepr([0.0, 1.0, 0.0], [1.0, 0.0]))
        )

        test_manifold(
            Mse,
            pts,
            test_injectivity_radius = false,
            test_musical_isomorphisms = true,
            test_tangent_vector_broadcasting = false,
            test_forward_diff = false,
            test_reverse_diff = false,
            basis_types_vecs = (basis_types[1], basis_types[3], basis_types[4]),
            basis_types_to_from = basis_types,
        )
        @test number_eltype(pts[1]) === Float64
        @test submanifold_component(Mse, pts[1], 1) === pts[1].parts[1]
        @test submanifold_component(Mse, pts[1], Val(1)) === pts[1].parts[1]
        @test submanifold_component(pts[1], 1) === pts[1].parts[1]
        @test submanifold_component(pts[1], Val(1)) === pts[1].parts[1]
        @test submanifold_components(Mse, pts[1]) === pts[1].parts
        @test submanifold_components(pts[1]) === pts[1].parts
    end

    @testset "vee/hat" begin
        M1 = Rotations(3)
        M2 = Euclidean(3)
        M = M1 × M2
        reshaper = Manifolds.ArrayReshaper()
        shape_se = Manifolds.ShapeSpecification(reshaper, M1, M2)

        e = Matrix{Float64}(I, 3, 3)
        x = Manifolds.prod_point(shape_se, exp(M1, e, hat(M1, e, [1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
        v = [0.1, 0.2, 0.3, -1.0, 2.0, -3.0]
        V = hat(M, x, v)
        v2 = vee(M, x, V)
        @test isapprox(v, v2)
    end

    @testset "Basis printing" begin
        p = ProductRepr([1.0, 0.0, 0.0], [1.0, 0.0])
        B = DefaultOrthonormalBasis()
        Bc = get_basis(Mse, p, B)
        @test sprint(show, "text/plain", Bc) == """
        DefaultOrthonormalBasis(ℝ) for a product manifold with coordinates in ℝ
        Basis for component 1:
        DefaultOrthonormalBasis(ℝ) with coordinates in ℝ and 2 basis vectors:
         E1 =
          3-element Array{Int64,1}:
           0
           1
           0
         E2 =
          3-element Array{Int64,1}:
           0
           0
           1
        Basis for component 2:
        DefaultOrthonormalBasis(ℝ) with coordinates in ℝ and 2 basis vectors:
         E1 =
          2-element Array{Float64,1}:
           1.0
           0.0
         E2 =
          2-element Array{Float64,1}:
           0.0
           1.0
        """
    end
end
