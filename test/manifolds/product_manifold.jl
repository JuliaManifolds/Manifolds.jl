include("../utils.jl")

using RecursiveArrayTools: ArrayPartition

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
    @test Mse[1] == M1
    @test Mse[2] == M2
    @test injectivity_radius(Mse) ≈ π
    @test injectivity_radius(
        Mse,
        ProductRetraction(ExponentialRetraction(), ExponentialRetraction()),
    ) ≈ π
    @test injectivity_radius(Mse, ExponentialRetraction()) ≈ π
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

    @test Manifolds.number_of_components(Mse) == 2
    # test that arrays are not points
    @test_throws DomainError is_point(Mse, [1, 2], true)
    @test check_point(Mse, [1, 2]) isa DomainError
    @test_throws DomainError is_vector(Mse, 1, [1, 2], true; check_base_point=false)
    @test check_vector(Mse, 1, [1, 2]; check_base_point=false) isa DomainError
    #default fallbacks for check_size, Product not working with Arrays
    @test Manifolds.check_size(Mse, zeros(2)) isa DomainError
    @test Manifolds.check_size(Mse, zeros(2), zeros(3)) isa DomainError
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

        p1ap = ArrayPartition([0.0, 1.0, 0.0], [0.0, 0.0])
        @test get_component(Mse, p1ap, 1) == p1ap.x[1]
        @test get_component(Mse, p1ap, Val(1)) == p1ap.x[1]
        @test p1ap[Mse, 1] == p1ap.x[1]
        @test p1ap[Mse, Val(1)] == p1ap.x[1]
        @test p1ap[Mse, 1] isa Vector
        @test p1ap[Mse, Val(1)] isa Vector
        set_component!(Mse, p1ap, p2, 2)
        @test get_component(Mse, p1ap, 2) == p2
        p1ap[Mse, 2] = 2 * p2
        @test p1ap[Mse, 2] == 2 * p2
        p3 = [11.0, 15.0]
        set_component!(Mse, p1ap, p3, Val(2))
        @test get_component(Mse, p1ap, Val(2)) == p3
        p1ap[Mse, Val(2)] = 2 * p3
        @test p1ap[Mse, Val(2)] == 2 * p3

        p1c = copy(p1)
        p1c.parts[1][1] = -123.0
        @test p1c.parts[1][1] == -123.0
        @test p1.parts[1][1] == 0.0
        copyto!(p1c, p1)
        @test p1c.parts[1][1] == 0.0
    end

    @testset "copyto!" begin
        p = ProductRepr([0.0, 1.0, 0.0], [0.0, 0.0])
        X = ProductRepr([1.0, 0.0, 0.0], [1.0, 0.0])
        q = allocate(p)
        copyto!(Mse, q, p)
        @test p.parts == q.parts
        Y = allocate(X)
        copyto!(Mse, Y, p, X)
        @test Y.parts == X.parts
    end

    @testset "Broadcasting" begin
        p1 = ProductRepr([0.0, 1.0, 0.0], [0.0, 1.0])
        p2 = ProductRepr([3.0, 4.0, 5.0], [2.0, 5.0])
        br_result = p1 .+ 2.0 .* p2
        @test br_result isa ProductRepr
        @test br_result.parts[1] ≈ [6.0, 9.0, 10.0]
        @test br_result.parts[2] ≈ [4.0, 11.0]

        br_result .= 2.0 .* p1 .+ p2
        @test br_result.parts[1] ≈ [3.0, 6.0, 5.0]
        @test br_result.parts[2] ≈ [2.0, 7.0]

        br_result .= p1
        @test br_result.parts[1] ≈ [0.0, 1.0, 0.0]
        @test br_result.parts[2] ≈ [0.0, 1.0]

        @test axes(p1) == (Base.OneTo(2),)

        # errors
        p3 = ProductRepr([3.0, 4.0, 5.0], [2.0, 5.0], [3.0, 2.0])
        @test_throws DimensionMismatch p1 .+ p3
        @test_throws DimensionMismatch p1 .= p3
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
        @test is_point(Mpr, p, true)
        @test_throws CompositeManifoldError is_point(Mpr, X, true)
        @test_throws ComponentManifoldError is_vector(Mpr, pf, X, true)
        @test_throws ComponentManifoldError is_vector(Mpr, p, Xf, true)
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

    M3 = Manifolds.Rotations(2)
    Mser = ProductManifold(M1, M2, M3)

    @test submanifold(Mser, 2) == M2
    @test (@inferred submanifold(Mser, Val((1, 3)))) == M1 × M3
    @test submanifold(Mser, 2:3) == M2 × M3
    @test submanifold(Mser, [1, 3]) == M1 × M3

    pts_sphere = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pts_r2 = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.1]]
    angles = (0.0, π / 2, 2π / 3)
    pts_rot = [[cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)] for ϕ in angles]
    pts = [ProductRepr(p[1], p[2], p[3]) for p in zip(pts_sphere, pts_r2, pts_rot)]
    test_manifold(
        Mser,
        pts,
        test_injectivity_radius=false,
        test_forward_diff=false,
        test_reverse_diff=false,
        is_tangent_atol_multiplier=1,
        exp_log_atol_multiplier=1,
        test_inplace=true,
        test_rand_point=true,
        test_rand_tvector=true,
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

    @testset "Implicit product vector transport" begin
        p = ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0])
        q = ProductRepr([0.0, 1.0, 0.0], [2.0, 0.0])
        X = log(Mse, p, q)
        for m in [ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport()]
            Y = vector_transport_to(Mse, p, X, q, m)
            Z1 = vector_transport_to(
                Mse.manifolds[1],
                submanifold_component.([p, X, q], Ref(1))...,
                m,
            )
            Z2 = vector_transport_to(
                Mse.manifolds[2],
                submanifold_component.([p, X, q], Ref(2))...,
                m,
            )
            Z = ProductRepr(Z1, Z2)
            @test isapprox(Mse, q, Y, Z)
        end
        for m in [ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport()]
            Y = vector_transport_direction(Mse, p, X, X, m)
            Z1 = vector_transport_direction(
                Mse.manifolds[1],
                submanifold_component.([p, X, X], Ref(1))...,
                m,
            )
            Z2 = vector_transport_direction(
                Mse.manifolds[2],
                submanifold_component.([p, X, X], Ref(2))...,
                m,
            )
            Z = ProductRepr(Z1, Z2)
            @test isapprox(Mse, q, Y, Z)
        end
    end
    @testset "Parallel transport" begin
        p = ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0])
        q = ProductRepr([0.0, 1.0, 0.0], [2.0, 0.0])
        X = log(Mse, p, q)
        # to
        Y = parallel_transport_to(Mse, p, X, q)
        Z1 = parallel_transport_to(
            Mse.manifolds[1],
            submanifold_component.([p, X, q], Ref(1))...,
        )
        Z2 = parallel_transport_to(
            Mse.manifolds[2],
            submanifold_component.([p, X, q], Ref(2))...,
        )
        Z = ProductRepr(Z1, Z2)
        @test isapprox(Mse, q, Y, Z)
        Ym = allocate(Y)
        parallel_transport_to!(Mse, Ym, p, X, q)
        @test isapprox(Mse, q, Y, Z)

        # direction
        Y = parallel_transport_direction(Mse, p, X, X)
        Z1 = parallel_transport_direction(
            Mse.manifolds[1],
            submanifold_component.([p, X, X], Ref(1))...,
        )
        Z2 = parallel_transport_direction(
            Mse.manifolds[2],
            submanifold_component.([p, X, X], Ref(2))...,
        )
        Z = ProductRepr(Z1, Z2)
        @test isapprox(Mse, q, Y, Z)
        Ym = allocate(Y)
        parallel_transport_direction!(Mse, Ym, p, X, X)
        @test isapprox(Mse, q, Ym, Z)
    end

    @testset "ProductRepr" begin
        @test (@inferred convert(
            ProductRepr{Tuple{T,Float64,T} where T},
            ProductRepr(9, 10, 11),
        )) == ProductRepr(9, 10.0, 11)

        p = ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0])
        @test p == ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0])
        @test submanifold_component(Mse, p, 1) === p.parts[1]
        @test submanifold_component(Mse, p, Val(1)) === p.parts[1]
        @test submanifold_component(p, 1) === p.parts[1]
        @test submanifold_component(p, Val(1)) === p.parts[1]
        @test submanifold_components(Mse, p) === p.parts
        @test submanifold_components(p) === p.parts
    end

    @testset "ArrayPartition" begin
        p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 0.0])
        @test submanifold_component(Mse, p, 1) === p.x[1]
        @test submanifold_component(Mse, p, Val(1)) === p.x[1]
        @test submanifold_component(p, 1) === p.x[1]
        @test submanifold_component(p, Val(1)) === p.x[1]
        @test submanifold_components(Mse, p) === p.x
        @test submanifold_components(p) === p.x
    end

    for TP in [ProductRepr, ArrayPartition]
        @testset "TP=$TP" begin
            Ts = SizedVector{3,Float64}
            Tr2 = SizedVector{2,Float64}
            pts_sphere = [
                convert(Ts, [1.0, 0.0, 0.0]),
                convert(Ts, [0.0, 1.0, 0.0]),
                convert(Ts, [0.0, 0.0, 1.0]),
            ]
            pts_r2 = [
                convert(Tr2, [0.0, 0.0]),
                convert(Tr2, [1.0, 0.0]),
                convert(Tr2, [0.0, 0.1]),
            ]

            pts = [TP(p[1], p[2]) for p in zip(pts_sphere, pts_r2)]
            basis_types = (
                DefaultOrthonormalBasis(),
                ProjectedOrthonormalBasis(:svd),
                get_basis(Mse, pts[1], DefaultOrthonormalBasis()),
                DiagonalizingOrthonormalBasis(
                    TP(SizedVector{3}([0.0, 1.0, 0.0]), SizedVector{2}([1.0, 0.0])),
                ),
            )
            distr_M1 = Manifolds.uniform_distribution(M1, pts_sphere[1])
            distr_M2 = Manifolds.projected_distribution(
                M2,
                Distributions.MvNormal(zero(pts_r2[1]), 1.0),
            )
            distr_tv_M1 = Manifolds.normal_tvector_distribution(M1, pts_sphere[1], 1.0)
            distr_tv_M2 = Manifolds.normal_tvector_distribution(M2, pts_r2[1], 1.0)
            @test injectivity_radius(Mse, pts[1]) ≈ π
            @test injectivity_radius(Mse) ≈ π
            @test injectivity_radius(Mse, pts[1], ExponentialRetraction()) ≈ π
            @test injectivity_radius(Mse, ExponentialRetraction()) ≈ π

            test_manifold(
                Mse,
                pts;
                point_distributions=[
                    Manifolds.ProductPointDistribution(distr_M1, distr_M2),
                ],
                tvector_distributions=[
                    Manifolds.ProductFVectorDistribution(distr_tv_M1, distr_tv_M2),
                ],
                test_injectivity_radius=true,
                test_musical_isomorphisms=true,
                musical_isomorphism_bases=[DefaultOrthonormalBasis()],
                test_tangent_vector_broadcasting=true,
                test_forward_diff=true,
                test_reverse_diff=true,
                test_project_tangent=true,
                test_project_point=true,
                test_mutating_rand=false,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                test_riesz_representer=true,
                test_default_vector_transport=true,
                test_rand_point=true,
                test_rand_tvector=true,
                vector_transport_methods=[
                    ProductVectorTransport(ParallelTransport(), ParallelTransport()),
                    ProductVectorTransport(
                        SchildsLadderTransport(),
                        SchildsLadderTransport(),
                    ),
                    ProductVectorTransport(PoleLadderTransport(), PoleLadderTransport()),
                ],
                basis_types_vecs=(basis_types[1], basis_types[3], basis_types[4]),
                basis_types_to_from=basis_types,
                is_tangent_atol_multiplier=1,
                exp_log_atol_multiplier=1,
            )
            @test number_eltype(pts[1]) === Float64

            @test (@inferred ManifoldsBase._get_vector_cache_broadcast(pts[1])) ===
                  Val(false)
        end
    end

    @testset "vee/hat" begin
        M1 = Rotations(3)
        M2 = Euclidean(3)
        M = M1 × M2

        e = Matrix{Float64}(I, 3, 3)
        p = ProductRepr(exp(M1, e, hat(M1, e, [1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
        X = [0.1, 0.2, 0.3, -1.0, 2.0, -3.0]

        Xc = hat(M, p, X)
        X2 = vee(M, p, Xc)
        @test isapprox(X, X2)
    end

    @testset "get_coordinates" begin
        # make sure `get_coordinates` does not return an `ArrayPartition`
        p1 = ProductRepr([0.0, 1.0, 0.0], [0.0, 0.0])
        X1 = ProductRepr([1.0, 0.0, -1.0], [1.0, 0.0])
        Tp1Mse = TangentSpaceAtPoint(Mse, p1)
        c = get_coordinates(Tp1Mse, p1, X1, DefaultOrthonormalBasis())
        @test c isa Vector

        p1ap = ArrayPartition([0.0, 1.0, 0.0], [0.0, 0.0])
        X1ap = ArrayPartition([1.0, 0.0, -1.0], [1.0, 0.0])
        Tp1apMse = TangentSpaceAtPoint(Mse, p1ap)
        cap = get_coordinates(Tp1apMse, p1ap, X1ap, DefaultOrthonormalBasis())
        @test cap isa Vector
    end

    @testset "Basis printing" begin
        p = ProductRepr([1.0, 0.0, 0.0], [1.0, 0.0])
        B = DefaultOrthonormalBasis()
        Bc = get_basis(Mse, p, B)
        Bc_components_s = sprint.(show, "text/plain", Bc.data.parts)
        @test sprint(show, "text/plain", Bc) == """
        $(typeof(B)) for a product manifold
        Basis for component 1:
        $(Bc_components_s[1])
        Basis for component 2:
        $(Bc_components_s[2])
        """
    end

    @testset "Basis-related errors" begin
        a = ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0])
        B = CachedBasis(DefaultOrthonormalBasis(), ProductBasisData(([],)))
        @test_throws AssertionError get_vector!(
            Mse,
            a,
            ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0]),
            [1.0, 2.0, 3.0, 4.0, 5.0], # this is one element too long, hence assertionerror
            B,
        )
        @test_throws MethodError get_vector!(
            Mse,
            a,
            ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0]),
            [1.0, 2.0, 3.0, 4.0],
            B, # empty elements yield a submanifold MethodError
        )
    end

    @testset "allocation promotion" begin
        M2c = Euclidean(2; field=ℂ)
        Msec = ProductManifold(M1, M2c)
        @test Manifolds.allocation_promotion_function(Msec, get_vector, ()) === complex
        @test Manifolds.allocation_promotion_function(Mse, get_vector, ()) === identity
    end

    @testset "empty allocation" begin
        p = allocate_result(Mse, uniform_distribution)
        @test isa(p, ProductRepr)
        @test size(p[Mse, 1]) == (3,)
        @test size(p[Mse, 2]) == (2,)
    end

    @testset "Uniform distribution" begin
        Mss = ProductManifold(Sphere(2), Sphere(2))
        p = rand(uniform_distribution(Mss))
        @test is_point(Mss, p)
        @test is_point(Mss, rand(uniform_distribution(Mss, p)))
    end

    @testset "Atlas & Induced Basis" begin
        M = ProductManifold(Euclidean(2), Euclidean(2))
        p = ProductRepr(zeros(2), ones(2))
        X = ProductRepr(ones(2), 2 .* ones(2))
        A = RetractionAtlas()
        a = get_parameters(M, A, p, p)
        p2 = get_point(M, A, p, a)
        @test all(p2.parts .== p.parts)
    end

    @testset "metric conversion" begin
        M = SymmetricPositiveDefinite(3)
        N = ProductManifold(M, M)
        e = EuclideanMetric()
        p = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1]
        q = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1]
        P = ProductRepr(p, q)
        X = ProductRepr(log(M, p, q), log(M, q, p))
        Y = change_metric(N, e, P, X)
        Yc = ProductRepr(
            change_metric(M, e, p, log(M, p, q)),
            change_metric(M, e, q, log(M, q, p)),
        )
        @test norm(N, P, Y - Yc) ≈ 0
        Z = change_representer(N, e, P, X)
        Zc = ProductRepr(
            change_representer(M, e, p, log(M, p, q)),
            change_representer(M, e, q, log(M, q, p)),
        )
        @test norm(N, P, Z - Zc) ≈ 0
    end
end
