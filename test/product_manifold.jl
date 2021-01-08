include("utils.jl")

struct NotImplementedReshaper <: Manifolds.AbstractReshaper end

function parray(M, x)
    return Manifolds.ProductArray(
        Manifolds.ShapeSpecification(Manifolds.StaticReshaper(), M.manifolds...),
        x,
    )
end

@testset "Product manifold" begin
    @test_throws MethodError ProductManifold()
    M1 = Sphere(2)
    M2 = Euclidean(2)
    @test (@inferred ProductManifold(M1, M2)) isa ProductManifold
    Mse = ProductManifold(M1, M2)
    @test Mse == M1 × M2
    @test Mse == ProductManifold(M1) × M2
    @test Mse == ProductManifold(M1) × ProductManifold(M2)
    @test Mse == M1 × ProductManifold(M2)
    @test injectivity_radius(Mse) ≈ π
    @test injectivity_radius(
        Mse,
        ProductRepr([0.0, 1.0, 0.0], [0.0, 0.0]),
        ProductRetraction(ExponentialRetraction(), ExponentialRetraction()),
    ) ≈ π
    @test injectivity_radius(
        Mse,
        ProductRepr([0.0, 1.0, 0.0], [0.0, 0.0]),
        ExponentialRetraction(),
    ) ≈ π
    @test is_default_metric(Mse, ProductMetric())
    @test Manifolds.default_metric_dispatch(Mse, ProductMetric()) === Val{true}()
    @test_throws ErrorException Manifolds.make_reshape(
        NotImplementedReshaper(),
        Int64,
        zeros(2, 3),
    )
    @test Manifolds.number_of_components(Mse) == 2
    types = [Vector{Float64}]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_STATIC_SIZED && push!(types, MVector{5,Float64})

    retraction_methods = [
        Manifolds.ProductRetraction(
            ManifoldsBase.ExponentialRetraction(),
            ManifoldsBase.ExponentialRetraction(),
        ),
    ]
    inverse_retraction_methods = [
        Manifolds.InverseProductRetraction(
            ManifoldsBase.LogarithmicInverseRetraction(),
            ManifoldsBase.LogarithmicInverseRetraction(),
        ),
    ]

    reshapers = (Manifolds.StaticReshaper(), Manifolds.ArrayReshaper())

    @testset "get_component, set_component!, getindex and setindex!" begin
        p1 = ProductRepr([0.0, 1.0, 0.0], [0.0, 0.0])
        @test get_component(Mse, p1, 1) == p1.parts[1]
        @test get_component(Mse, p1, Val(1)) == p1.parts[1]
        @test p1[Mse, 1] == p1.parts[1]
        @test p1[Mse, Val(1)] == p1.parts[1]
        @test p1[Mse, 1] isa Vector
        @test p1[Mse, Val(1)] isa Vector
        p2 = [10.0, 12.0]
        set_component!(Mse, p1, p2, 2)
        @test get_component(Mse, p1, 2) == p2
        p1[Mse, 2] = 2 * p2
        @test p1[Mse, 2] == 2 * p2
        p3 = [11.0, 15.0]
        set_component!(Mse, p1, p3, Val(2))
        @test get_component(Mse, p1, Val(2)) == p3
        p1[Mse, Val(2)] = 2 * p3
        @test p1[Mse, Val(2)] == 2 * p3

        shape_a_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M1, M2)
        pra1 = Manifolds.ProductArray(shape_a_se, [0.0, 1.0, 0.0, 0.0, 0.0])
        @test get_component(Mse, pra1, 1) == p1.parts[1]
        @test get_component(Mse, pra1, Val(1)) == p1.parts[1]
        @test pra1[Mse, 1] == pra1.parts[1]
        @test pra1[Mse, Val(1)] == pra1.parts[1]
        @test pra1[Mse, 1] isa Vector
        @test pra1[Mse, Val(1)] isa Vector
        set_component!(Mse, pra1, p2, 2)
        @test get_component(Mse, pra1, 2) == p2
        pra1[Mse, 2] = 2 * p2
        @test pra1[Mse, 2] == 2 * p2
        set_component!(Mse, pra1, p3, Val(2))
        @test get_component(Mse, pra1, Val(2)) == p3
        pra1[Mse, Val(2)] = 2 * p3
        @test pra1[Mse, Val(2)] == 2 * p3

        shape_s_se = Manifolds.ShapeSpecification(Manifolds.StaticReshaper(), M1, M2)
        prs1 = Manifolds.ProductArray(shape_s_se, [0.0, 1.0, 0.0, 0.0, 0.0])
        @test get_component(Mse, prs1, 1) == p1.parts[1]
        @test get_component(Mse, prs1, Val(1)) == p1.parts[1]
        @test prs1[Mse, 1] == prs1.parts[1]
        @test prs1[Mse, Val(1)] == prs1.parts[1]
        @test prs1[Mse, 1] isa Vector
        @test prs1[Mse, Val(1)] isa Vector
        set_component!(Mse, prs1, p2, 2)
        @test get_component(Mse, prs1, 2) == p2
        prs1[Mse, 2] = 2 * p2
        @test prs1[Mse, 2] == 2 * p2
        set_component!(Mse, prs1, p3, Val(2))
        @test get_component(Mse, prs1, Val(2)) == p3
        prs1[Mse, Val(2)] = 2 * p3
        @test prs1[Mse, Val(2)] == 2 * p3
    end

    @testset "CompositeManifoldError" begin
        Mpr = Sphere(2) × Sphere(2)
        p1 = [1.0, 0.0, 0.0]
        p2 = [0.0, 1.0, 0.0]
        X1 = [0.0, 1.0, 0.2]
        X2 = [1.0, 0.0, 0.2]
        p = ProductRepr(p1, p2)
        X = ProductRepr(X1, X2)
        pf = ProductRepr(p1, X1)
        Xf = ProductRepr(X1, p2)
        @test is_manifold_point(Mpr, p, true)
        @test_throws CompositeManifoldError is_manifold_point(Mpr, X, true)
        @test_throws ComponentManifoldError is_tangent_vector(Mpr, pf, X, true)
        @test_throws ComponentManifoldError is_tangent_vector(Mpr, p, Xf, true)
    end

    @testset "arithmetic" begin
        Mee = ProductManifold(Euclidean(3), Euclidean(2))
        p1 = ProductRepr([0.0, 1.0, 0.0], [0.0, 1.0])
        p2 = ProductRepr([1.0, 2.0, 0.0], [2.0, 3.0])

        @test isapprox(Mee, p1 + p2, ProductRepr([1.0, 3.0, 0.0], [2.0, 4.0]))
        @test isapprox(Mee, p1 - p2, ProductRepr([-1.0, -1.0, 0.0], [-2.0, -2.0]))
        @test isapprox(Mee, -p1, ProductRepr([0.0, -1.0, 0.0], [0.0, -1.0]))
        @test isapprox(Mee, p1 * 2, ProductRepr([0.0, 2.0, 0.0], [0.0, 2.0]))
        @test isapprox(Mee, 2 * p1, ProductRepr([0.0, 2.0, 0.0], [0.0, 2.0]))
        @test isapprox(Mee, p1 / 2, ProductRepr([0.0, 0.5, 0.0], [0.0, 0.5]))
    end

    @testset "Show methods" begin
        Mse2 = ProductManifold(M1, M1, M2, M2)
        @test sprint(show, Mse2) == "ProductManifold($(M1), $(M1), $(M2), $(M2))"
        withenv("LINES" => 10, "COLUMNS" => 100) do
            @test sprint(show, "text/plain", ProductManifold(M1)) ==
                  "ProductManifold with 1 submanifold:\n $(M1)"
            @test sprint(show, "text/plain", Mse2) ==
                  "ProductManifold with 4 submanifolds:\n $(M1)\n $(M1)\n $(M2)\n $(M2)"
            return nothing
        end
        withenv("LINES" => 7, "COLUMNS" => 100) do
            @test sprint(show, "text/plain", Mse2) ==
                  "ProductManifold with 4 submanifolds:\n $(M1)\n ⋮\n $(M2)"
            return nothing
        end

        @test sprint(show, "text/plain", ProductManifold(Mse, Mse)) == """
        ProductManifold with 2 submanifolds:
         ProductManifold(Sphere(2, ℝ), Euclidean(2; field = ℝ))
         ProductManifold(Sphere(2, ℝ), Euclidean(2; field = ℝ))"""

        shape_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M1)
        p = Manifolds.ProductArray(shape_se, Float64[1, 0, 0])
        @test sprint(show, "text/plain", p) == """
        ProductArray with 1 submanifold component:
         Component 1 =
          3-element view(::$(sprint(show, Vector{Float64})), 1:3) with eltype Float64:
           1.0
           0.0
           0.0"""

        shape_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M1, M1, M2, M2)
        p = Manifolds.ProductArray(shape_se, Float64[1, 0, 0, 0, 1, 0, 1, 2, 3, 4])
        @test sprint(show, "text/plain", p) == """
        ProductArray with 4 submanifold components:
         Component 1 =
          3-element view(::$(sprint(show, Vector{Float64})), 1:3) with eltype Float64:
           1.0
           0.0
           0.0
         Component 2 =
          3-element view(::$(sprint(show, Vector{Float64})), 4:6) with eltype Float64:
           0.0
           1.0
           0.0
         Component 3 =
          2-element view(::$(sprint(show, Vector{Float64})), 7:8) with eltype Float64:
           1.0
           2.0
         Component 4 =
          2-element view(::$(sprint(show, Vector{Float64})), 9:10) with eltype Float64:
           3.0
           4.0"""

        shape_se =
            Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M1, M1, M2, M2, M2)
        p = Manifolds.ProductArray(shape_se, Float64[1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5, 6])

        @test sprint(show, "text/plain", p) == """
        ProductArray with 5 submanifold components:
         Component 1 =
          3-element view(::$(sprint(show, Vector{Float64})), 1:3) with eltype Float64:
           1.0
           0.0
           0.0
         Component 2 =
          3-element view(::$(sprint(show, Vector{Float64})), 4:6) with eltype Float64:
           0.0
           1.0
           0.0
         ⋮
         Component 4 =
          2-element view(::$(sprint(show, Vector{Float64})), 9:10) with eltype Float64:
           3.0
           4.0
         Component 5 =
          2-element view(::$(sprint(show, Vector{Float64})), 11:12) with eltype Float64:
           5.0
           6.0"""

        p = Manifolds.ProductRepr(Float64[1, 0, 0])
        @test sprint(show, "text/plain", p) == """
        ProductRepr with 1 submanifold component:
         Component 1 =
          3-element $(sprint(show, Vector{Float64})):
           1.0
           0.0
           0.0"""

        p = Manifolds.ProductRepr(
            Float64[1, 0, 0],
            Float64[0, 1, 0],
            Float64[1, 2],
            Float64[3, 4],
        )
        @test sprint(show, "text/plain", p) == """
        ProductRepr with 4 submanifold components:
         Component 1 =
          3-element $(sprint(show, Vector{Float64})):
           1.0
           0.0
           0.0
         Component 2 =
          3-element $(sprint(show, Vector{Float64})):
           0.0
           1.0
           0.0
         Component 3 =
          2-element $(sprint(show, Vector{Float64})):
           1.0
           2.0
         Component 4 =
          2-element $(sprint(show, Vector{Float64})):
           3.0
           4.0"""

        p = Manifolds.ProductRepr(
            Float64[1, 0, 0],
            Float64[0, 1, 0],
            Float64[1, 2],
            Float64[3, 4],
            Float64[5, 6],
        )
        @test sprint(show, "text/plain", p) == """
        ProductRepr with 5 submanifold components:
         Component 1 =
          3-element $(sprint(show, Vector{Float64})):
           1.0
           0.0
           0.0
         Component 2 =
          3-element $(sprint(show, Vector{Float64})):
           0.0
           1.0
           0.0
         ⋮
         Component 4 =
          2-element $(sprint(show, Vector{Float64})):
           3.0
           4.0
         Component 5 =
          2-element $(sprint(show, Vector{Float64})):
           5.0
           6.0"""
    end

    for T in types, reshaper in reshapers
        if reshaper == reshapers[2] && T != Vector{Float64}
            continue
        end
        shape_se = Manifolds.ShapeSpecification(reshaper, M1, M2)
        @testset "Type $T" begin
            pts_base = [
                convert(T, [1.0, 0.0, 0.0, 0.0, 0.0]),
                convert(T, [0.0, 1.0, 0.0, 1.0, 0.0]),
                convert(T, [0.0, 0.0, 1.0, 0.0, 0.1]),
            ]
            pts = map(p -> Manifolds.ProductArray(shape_se, p), pts_base)
            distr_M1 = Manifolds.uniform_distribution(M1, pts_base[1][1:3])
            distr_M2 = Manifolds.projected_distribution(
                M2,
                Distributions.MvNormal(zero(pts_base[1][4:5]), 1.0),
            )
            distr_tv_M1 = Manifolds.normal_tvector_distribution(M1, pts_base[1][1:3], 1.0)
            distr_tv_M2 = Manifolds.normal_tvector_distribution(M2, pts_base[1][4:5], 1.0)
            @test injectivity_radius(Mse, pts[1]) ≈ π
            @test injectivity_radius(Mse) ≈ π
            @test injectivity_radius(Mse, pts[1], ExponentialRetraction()) ≈ π
            @test injectivity_radius(Mse, ExponentialRetraction()) ≈ π
            test_manifold(
                Mse,
                pts;
                test_reverse_diff=isa(T, Vector),
                test_musical_isomorphisms=true,
                test_injectivity_radius=true,
                test_project_point=true,
                test_project_tangent=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                test_mutating_rand=isa(T, Vector),
                point_distributions=[
                    Manifolds.ProductPointDistribution(distr_M1, distr_M2),
                ],
                tvector_distributions=[
                    Manifolds.ProductFVectorDistribution(distr_tv_M1, distr_tv_M2),
                ],
                is_tangent_atol_multiplier=1,
                exp_log_atol_multiplier=1,
            )
        end
    end

    M3 = Manifolds.Rotations(2)
    Mser = ProductManifold(M1, M2, M3)
    shape_ser = Manifolds.ShapeSpecification(reshapers[1], M1, M2, M3)

    @testset "ShapeSpecification detailed reshapers" begin
        shape_ser_3_test = Manifolds.ShapeSpecification(
            (reshapers[1], reshapers[2], reshapers[1]),
            M1,
            M2,
            M3,
        )
        @test shape_ser_3_test.reshapers[1] == reshapers[1]
        @test shape_ser_3_test.reshapers[2] == reshapers[2]
        @test shape_ser_3_test.reshapers[3] == reshapers[1]
    end

    @testset "high-dimensional product manifold" begin
        Mhigh = Euclidean(100000)
        shape_high = Manifolds.ShapeSpecification(reshapers[2], M1, Mhigh)
        a = Manifolds.ProductArray(shape_high, collect(1.0:100003.0))
        b = a.parts[2] .^ 2
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

    pts_sphere = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pts_r2 = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.1]]
    angles = (0.0, π / 2, 2π / 3)
    pts_rot = [[cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)] for ϕ in angles]
    pts = [
        Manifolds.prod_point(shape_ser, p[1], p[2], p[3]) for
        p in zip(pts_sphere, pts_r2, pts_rot)
    ]
    test_manifold(
        Mser,
        pts,
        test_injectivity_radius=false,
        test_forward_diff=false,
        test_reverse_diff=false,
        is_tangent_atol_multiplier=1,
        exp_log_atol_multiplier=1,
    )

    @testset "product vector transport" begin
        p = ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0])
        q = ProductRepr([0.0, 1.0, 0.0], [2.0, 0.0])
        X = log(Mse, p, q)
        m = ProductVectorTransport(ParallelTransport(), ParallelTransport())
        Y = vector_transport_to(Mse, p, X, q, m)
        Z = -log(Mse, q, p)
        @test isapprox(Mse, q, Y, Z)
    end

    @testset "prod_point" begin
        shape_se = Manifolds.ShapeSpecification(reshapers[1], M1, M2)
        Ts = SizedVector{3,Float64}
        Tr2 = SizedVector{2,Float64}
        T = SizedVector{5,Float64}
        pts_base = [
            convert(T, [1.0, 0.0, 0.0, 0.0, 0.0]),
            convert(T, [0.0, 1.0, 0.0, 1.0, 0.0]),
            convert(T, [0.0, 0.0, 1.0, 0.0, 0.1]),
        ]
        pts = map(p -> Manifolds.ProductArray(shape_se, p), pts_base)
        pts_sphere = [
            convert(Ts, [1.0, 0.0, 0.0]),
            convert(Ts, [0.0, 1.0, 0.0]),
            convert(Ts, [0.0, 0.0, 1.0]),
        ]
        pts_r2 =
            [convert(Tr2, [0.0, 0.0]), convert(Tr2, [1.0, 0.0]), convert(Tr2, [0.0, 0.1])]
        pts_prod =
            [Manifolds.prod_point(shape_se, p[1], p[2]) for p in zip(pts_sphere, pts_r2)]
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

        p_inf = parray(Mse, randn(5))
        @test (@inferred ManifoldsBase.allocate_result_type(
            Mse,
            Manifolds.log,
            (p_inf, p_inf),
        )) === Float64
        @test (@inferred number_eltype(typeof(p_inf))) === Float64
        @test pts_prod[1] .+ fill(1.0, 5) == pts_prod[1] .+ 1.0
    end

    @testset "ProductRepr" begin
        Ts = SizedVector{3,Float64}
        Tr2 = SizedVector{2,Float64}
        pts_sphere = [
            convert(Ts, [1.0, 0.0, 0.0]),
            convert(Ts, [0.0, 1.0, 0.0]),
            convert(Ts, [0.0, 0.0, 1.0]),
        ]
        pts_r2 =
            [convert(Tr2, [0.0, 0.0]), convert(Tr2, [1.0, 0.0]), convert(Tr2, [0.0, 0.1])]

        pts = [ProductRepr(p[1], p[2]) for p in zip(pts_sphere, pts_r2)]
        basis_types = (
            DefaultOrthonormalBasis(),
            ProjectedOrthonormalBasis(:svd),
            get_basis(Mse, pts[1], DefaultOrthonormalBasis()),
            DiagonalizingOrthonormalBasis(ProductRepr([0.0, 1.0, 0.0], [1.0, 0.0])),
        )

        @test (@inferred convert(
            ProductRepr{Tuple{T,Float64,T} where T},
            ProductRepr(9, 10, 11),
        )) == ProductRepr(9, 10.0, 11)

        test_manifold(
            Mse,
            pts,
            test_injectivity_radius=false,
            test_musical_isomorphisms=true,
            test_tangent_vector_broadcasting=false,
            test_forward_diff=false,
            test_reverse_diff=false,
            test_project_tangent=true,
            test_project_point=true,
            test_default_vector_transport=true,
            vector_transport_methods=[
                ProductVectorTransport(ParallelTransport(), ParallelTransport()),
                ProductVectorTransport(SchildsLadderTransport(), SchildsLadderTransport()),
                ProductVectorTransport(PoleLadderTransport(), PoleLadderTransport()),
            ],
            basis_types_vecs=(basis_types[1], basis_types[3], basis_types[4]),
            basis_types_to_from=basis_types,
            is_tangent_atol_multiplier=1,
            exp_log_atol_multiplier=1,
        )
        @test number_eltype(pts[1]) === Float64
        @test submanifold_component(Mse, pts[1], 1) === pts[1].parts[1]
        @test submanifold_component(Mse, pts[1], Val(1)) === pts[1].parts[1]
        @test submanifold_component(pts[1], 1) === pts[1].parts[1]
        @test submanifold_component(pts[1], Val(1)) === pts[1].parts[1]
        @test submanifold_components(Mse, pts[1]) === pts[1].parts
        @test submanifold_components(pts[1]) === pts[1].parts
        @test (@inferred ManifoldsBase._get_vector_cache_broadcast(pts[1])) === Val(false)
    end

    @testset "vee/hat" begin
        M1 = Rotations(3)
        M2 = Euclidean(3)
        M = M1 × M2
        reshaper = Manifolds.ArrayReshaper()
        shape_se = Manifolds.ShapeSpecification(reshaper, M1, M2)

        e = Matrix{Float64}(I, 3, 3)
        x = Manifolds.prod_point(
            shape_se,
            exp(M1, e, hat(M1, e, [1.0, 2.0, 3.0])),
            [1.0, 2.0, 3.0],
        )
        v = [0.1, 0.2, 0.3, -1.0, 2.0, -3.0]
        V = hat(M, x, v)
        v2 = vee(M, x, V)
        @test isapprox(v, v2)

        xr = ProductRepr(exp(M1, e, hat(M1, e, [1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])

        Vr = hat(M, xr, v)
        v2r = vee(M, xr, V)
        @test isapprox(v, v2r)
    end

    @testset "Basis printing" begin
        p = ProductRepr([1.0, 0.0, 0.0], [1.0, 0.0])
        B = DefaultOrthonormalBasis()
        Bc = get_basis(Mse, p, B)
        Bc_components_s = sprint.(show, "text/plain", Bc.data.parts)
        @test sprint(show, "text/plain", Bc) == """
        DefaultOrthonormalBasis(ℝ) for a product manifold
        Basis for component 1:
        $(Bc_components_s[1])
        Basis for component 2:
        $(Bc_components_s[2])
        """
    end

    @testset "Basis-related errors" begin
        a = ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0])
        @test_throws ErrorException get_vector!(
            Mse,
            a,
            ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0]),
            [1.0, 2.0, 3.0, 4.0, 5.0],
            CachedBasis(DefaultOrthonormalBasis(), []),
        )
    end

    @testset "allocation promotion" begin
        M2c = Euclidean(2; field=ℂ)
        Msec = ProductManifold(M1, M2c)
        @test Manifolds.allocation_promotion_function(Msec, get_vector, ()) === complex
        @test Manifolds.allocation_promotion_function(Mse, get_vector, ()) === identity
    end
end
