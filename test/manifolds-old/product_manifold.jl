include("../header.jl")

using RecursiveArrayTools: ArrayPartition

@testset "Product manifold" begin
    @test_throws MethodError ProductManifold()
    M1 = Sphere(2)
    M2 = Euclidean(2)
    @test (@inferred ProductManifold(M1, M2)) isa ProductManifold
    Mse = ProductManifold(M1, M2)
    @test Mse == M1 × M2
    @test !is_flat(Mse)
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
        ArrayPartition([0.0, 1.0, 0.0], [0.0, 0.0]),
        ProductRetraction(ExponentialRetraction(), ExponentialRetraction()),
    ) ≈ π
    @test injectivity_radius(
        Mse,
        ArrayPartition([0.0, 1.0, 0.0], [0.0, 0.0]),
        ExponentialRetraction(),
    ) ≈ π
    @test is_default_metric(Mse, ProductMetric())

    @test Manifolds.number_of_components(Mse) == 2
    # test that arrays are not points
    @test_throws DomainError is_point(Mse, [1, 2]; error = :error)
    @test check_point(Mse, [1, 2]) isa DomainError
    @test_throws DomainError is_vector(Mse, 1, [1, 2]; error = :error, check_base_point = false)
    @test check_vector(Mse, 1, [1, 2]; check_base_point = false) isa DomainError
    #default fallbacks for check_size, Product not working with Arrays
    @test Manifolds.check_size(Mse, zeros(2)) isa DomainError
    @test Manifolds.check_size(Mse, zeros(2), zeros(3)) isa DomainError
    types = [Vector{Float64}]

    retraction_methods = [
        ProductRetraction(ExponentialRetraction(), ExponentialRetraction()),
        ExponentialRetraction(),
    ]
    inverse_retraction_methods = [
        InverseProductRetraction(
            LogarithmicInverseRetraction(),
            LogarithmicInverseRetraction(),
        ),
        LogarithmicInverseRetraction(),
    ]

    @testset "arithmetic" begin
        Mee = ProductManifold(Euclidean(3), Euclidean(2))
        p1 = ArrayPartition([0.0, 1.0, 0.0], [0.0, 1.0])
        p2 = ArrayPartition([1.0, 2.0, 0.0], [2.0, 3.0])

        @test isapprox(Mee, p1 + p2, ArrayPartition([1.0, 3.0, 0.0], [2.0, 4.0]))
        @test isapprox(Mee, p1 - p2, ArrayPartition([-1.0, -1.0, 0.0], [-2.0, -2.0]))
        @test isapprox(Mee, -p1, ArrayPartition([0.0, -1.0, 0.0], [0.0, -1.0]))
        @test isapprox(Mee, p1 * 2, ArrayPartition([0.0, 2.0, 0.0], [0.0, 2.0]))
        @test isapprox(Mee, 2 * p1, ArrayPartition([0.0, 2.0, 0.0], [0.0, 2.0]))
        @test isapprox(Mee, p1 / 2, ArrayPartition([0.0, 0.5, 0.0], [0.0, 0.5]))
    end

    M3 = Rotations(2)
    Mser = ProductManifold(M1, M2, M3)

    @test submanifold(Mser, 2) == M2
    @test (@inferred submanifold(Mser, Val((1, 3)))) == M1 × M3
    @test submanifold(Mser, 2:3) == M2 × M3
    @test submanifold(Mser, [1, 3]) == M1 × M3

    pts_sphere = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pts_r2 = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.1]]
    angles = (0.0, π / 2, 2π / 3)
    pts_rot = [[cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)] for ϕ in angles]
    pts = [ArrayPartition(p[1], p[2], p[3]) for p in zip(pts_sphere, pts_r2, pts_rot)]
    Manifolds.test_manifold(
        Mser,
        pts,
        test_injectivity_radius = false,
        is_tangent_atol_multiplier = 1,
        exp_log_atol_multiplier = 1,
        test_inplace = true,
        test_rand_point = true,
        test_rand_tvector = true,
        test_representation_size = false,
    )

    @testset "manifold tests (static size)" begin
        Ts = SizedVector{3, Float64}
        Tr2 = SizedVector{2, Float64}
        pts_sphere = [
            convert(Ts, [1.0, 0.0, 0.0]),
            convert(Ts, [0.0, 1.0, 0.0]),
            convert(Ts, [0.0, 0.0, 1.0]),
        ]
        pts_r2 =
            [convert(Tr2, [0.0, 0.0]), convert(Tr2, [1.0, 0.0]), convert(Tr2, [0.0, 0.1])]

        pts = [ArrayPartition(p[1], p[2]) for p in zip(pts_sphere, pts_r2)]
        basis_types = (
            DefaultOrthonormalBasis(),
            ProjectedOrthonormalBasis(:svd),
            get_basis(Mse, pts[1], DefaultOrthonormalBasis()),
            DiagonalizingOrthonormalBasis(
                ArrayPartition(SizedVector{3}([0.0, 1.0, 0.0]), SizedVector{2}([1.0, 0.0])),
            ),
        )
        distr_M1 = Manifolds.uniform_distribution(M1, pts_sphere[1])
        distr_M2 = Manifolds.projected_distribution(
            M2,
            Distributions.MvNormal(zero(pts_r2[1]), 1.0 * I),
        )
        @test injectivity_radius(Mse, pts[1]) ≈ π
        @test injectivity_radius(Mse) ≈ π
        @test injectivity_radius(Mse, pts[1], ExponentialRetraction()) ≈ π
        @test injectivity_radius(Mse, ExponentialRetraction()) ≈ π

        @test ManifoldsBase.allocate_coordinates(
            Mse,
            pts[1],
            Float64,
            number_of_coordinates(Mse, DefaultOrthogonalBasis()),
        ) isa Vector{Float64}

        Y = allocate(pts[1])
        inverse_retract!(Mse, Y, pts[1], pts[2], default_inverse_retraction_method(Mse))
        @test isapprox(
            Mse,
            pts[1],
            Y,
            inverse_retract(Mse, pts[1], pts[2], default_inverse_retraction_method(Mse)),
        )

        Mse_point_distributions = []
        Mse_tvector_distributions = []

        if VERSION >= v"1.9"
            distr_tv_M1 = Manifolds.normal_tvector_distribution(M1, pts_sphere[1], 1.0)
            distr_tv_M2 = Manifolds.normal_tvector_distribution(M2, pts_r2[1], 1.0)

            ProductPointDistribution =
                Base.get_extension(
                Manifolds,
                :ManifoldsDistributionsExt,
            ).ProductPointDistribution
            push!(Mse_point_distributions, ProductPointDistribution(distr_M1, distr_M2))

            ProductFVectorDistribution =
                Base.get_extension(
                Manifolds,
                :ManifoldsDistributionsExt,
            ).ProductFVectorDistribution
            push!(
                Mse_tvector_distributions,
                ProductFVectorDistribution(distr_tv_M1, distr_tv_M2),
            )

            MPointSupport =
                Base.get_extension(Manifolds, :ManifoldsDistributionsExt).MPointSupport
            FVectorSupport =
                Base.get_extension(Manifolds, :ManifoldsDistributionsExt).FVectorSupport

            Test.@test Distributions.support(Mse_point_distributions[1]) isa MPointSupport
            Test.@test Distributions.support(Mse_tvector_distributions[1]) isa
                FVectorSupport
        end

        Manifolds.test_manifold(
            Mse,
            pts;
            point_distributions = Mse_point_distributions,
            tvector_distributions = Mse_tvector_distributions,
            test_injectivity_radius = true,
            test_musical_isomorphisms = true,
            musical_isomorphism_bases = [DefaultOrthonormalBasis()],
            test_tangent_vector_broadcasting = true,
            test_project_tangent = true,
            test_project_point = true,
            test_mutating_rand = true,
            retraction_methods = retraction_methods,
            inverse_retraction_methods = inverse_retraction_methods,
            test_riesz_representer = true,
            test_default_vector_transport = true,
            test_rand_point = true,
            test_rand_tvector = true,
            vector_transport_methods = [
                ProductVectorTransport(ParallelTransport(), ParallelTransport()),
                ProductVectorTransport(SchildsLadderTransport(), SchildsLadderTransport()),
                ProductVectorTransport(PoleLadderTransport(), PoleLadderTransport()),
            ],
            basis_types_vecs = (basis_types[1], basis_types[3], basis_types[4]),
            basis_types_to_from = basis_types,
            is_tangent_atol_multiplier = 1,
            exp_log_atol_multiplier = 1,
            test_representation_size = false,
        )
        @test number_eltype(pts[1]) === Float64

        @test (@inferred ManifoldsBase._get_vector_cache_broadcast(pts[1])) === Val(false)
    end

    @testset "vee/hat" begin
        M1 = Rotations(3)
        M2 = Euclidean(3)
        M = M1 × M2

        e = Matrix{Float64}(I, 3, 3)
        p = ArrayPartition(exp(M1, e, hat(M1, e, [1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
        X = [0.1, 0.2, 0.3, -1.0, 2.0, -3.0]

        Xc = hat(M, p, X)
        X2 = vee(M, p, Xc)
        @test isapprox(X, X2)
    end

    @testset "empty allocation" begin
        p = allocate_result(Mse, Manifolds.uniform_distribution)
        @test isa(p, ArrayPartition)
        @test size(p[Mse, 1]) == (3,)
        @test size(p[Mse, 2]) == (2,)
    end

    @testset "Uniform distribution" begin
        Mss = ProductManifold(Sphere(2), Sphere(2))
        p = rand(Manifolds.uniform_distribution(Mss))
        @test is_point(Mss, p)
        @test is_point(Mss, rand(Manifolds.uniform_distribution(Mss, p)))
    end

    @testset "Atlas & Induced Basis" begin
        M = ProductManifold(Euclidean(2), Euclidean(2))
        p = ArrayPartition(zeros(2), ones(2))
        X = ArrayPartition(ones(2), 2 .* ones(2))
        A = RetractionAtlas()
        a = get_parameters(M, A, p, p)
        p2 = get_point(M, A, p, a)
        @test all(submanifold_components(p2) .== submanifold_components(p))
    end

    @testset "metric conversion" begin
        M = SymmetricPositiveDefinite(3)
        N = ProductManifold(M, M)
        e = EuclideanMetric()
        p = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1]
        q = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1]
        P = ArrayPartition(p, q)
        X = ArrayPartition(log(M, p, q), log(M, q, p))
        Y = change_metric(N, e, P, X)
        Yc = ArrayPartition(
            change_metric(M, e, p, log(M, p, q)),
            change_metric(M, e, q, log(M, q, p)),
        )
        @test norm(N, P, Y - Yc) ≈ 0
        Z = change_representer(N, e, P, X)
        Zc = ArrayPartition(
            change_representer(M, e, p, log(M, p, q)),
            change_representer(M, e, q, log(M, q, p)),
        )
        @test norm(N, P, Z - Zc) ≈ 0
    end

    @testset "default retraction, inverse retraction and VT" begin
        Mstb = ProductManifold(M1, TangentBundle(M1))
        T_p_ap = ArrayPartition{
            Float64,
            Tuple{
                Matrix{Float64},
                ArrayPartition{Float64, Tuple{Matrix{Float64}, Matrix{Float64}}},
            },
        }
        @test Manifolds.default_retraction_method(Mstb) === ProductRetraction(
            ExponentialRetraction(),
            Manifolds.FiberBundleProductRetraction(),
        )
        @test Manifolds.default_retraction_method(Mstb, T_p_ap) === ProductRetraction(
            ExponentialRetraction(),
            Manifolds.FiberBundleProductRetraction(),
        )

        @test Manifolds.default_inverse_retraction_method(Mstb) ===
            Manifolds.InverseProductRetraction(
            LogarithmicInverseRetraction(),
            Manifolds.FiberBundleInverseProductRetraction(),
        )
        @test Manifolds.default_inverse_retraction_method(Mstb, T_p_ap) ===
            Manifolds.InverseProductRetraction(
            LogarithmicInverseRetraction(),
            Manifolds.FiberBundleInverseProductRetraction(),
        )

        @test Manifolds.default_vector_transport_method(Mstb) === ProductVectorTransport(
            ParallelTransport(),
            Manifolds.FiberBundleProductVectorTransport(
                ParallelTransport(),
                ParallelTransport(),
            ),
        )
        @test Manifolds.default_vector_transport_method(Mstb, T_p_ap) ===
            ProductVectorTransport(
            ParallelTransport(),
            Manifolds.FiberBundleProductVectorTransport(
                ParallelTransport(),
                ParallelTransport(),
            ),
        )
        @test Manifolds.default_vector_transport_method(Mstb, T_p_ap) ===
            ProductVectorTransport(
            ParallelTransport(),
            Manifolds.FiberBundleProductVectorTransport(
                ParallelTransport(),
                ParallelTransport(),
            ),
        )
    end

    @testset "ManifoldDiff" begin
        p = ArrayPartition([0.0, 1.0, 0.0], [2.0, 3.0])
        q = ArrayPartition([1.0, 0.0, 0.0], [-2.0, 3.0])
        X = ArrayPartition([1.0, 0.0, 0.0], [2.0, 3.0])
        # ManifoldDiff
        @test ManifoldDiff.adjoint_Jacobi_field(
            Mse,
            p,
            q,
            0.5,
            X,
            ManifoldDiff.βdifferential_shortest_geodesic_startpoint,
        ) == ArrayPartition([0.5, 0.0, 0.0], [1.0, 1.5])
        X2 = allocate(X)
        ManifoldDiff.adjoint_Jacobi_field!(
            Mse,
            X2,
            p,
            q,
            0.5,
            X,
            ManifoldDiff.βdifferential_shortest_geodesic_startpoint,
        )
        @test X2 == ArrayPartition([0.5, 0.0, 0.0], [1.0, 1.5])
        @test ManifoldDiff.jacobi_field(
            Mse,
            p,
            q,
            0.5,
            X,
            ManifoldDiff.βdifferential_shortest_geodesic_startpoint,
        ) == ArrayPartition([0.3535533905932738, -0.35355339059327373, 0.0], [1.0, 1.5])
        X2 = allocate(X)
        ManifoldDiff.jacobi_field!(
            Mse,
            X2,
            p,
            q,
            0.5,
            X,
            ManifoldDiff.βdifferential_shortest_geodesic_startpoint,
        )
        @test X2 ==
            ArrayPartition([0.3535533905932738, -0.35355339059327373, 0.0], [1.0, 1.5])
    end
    @testset "gradient conversion" begin
        M = ProbabilitySimplex(3)
        p = [
            0.011143450837652447,
            0.0010060921724131108,
            0.41619787908924166,
            0.571652577900693,
        ]
        X_e = [
            0.062102217532734615,
            0.04344131268205693,
            0.4262317935277549,
            -0.5317753237425427,
        ]
        X_r = riemannian_gradient(M, p, X_e)

        N = M × M

        p_2 = ArrayPartition(p, p)
        X_e_2 = ArrayPartition(X_e, X_e)
        X_r_2 = riemannian_gradient(N, p_2, X_e_2)
        @test isapprox(X_r_2, ArrayPartition(X_r, X_r))

        Y_r_2 = similar(X_r_2)
        riemannian_gradient!(N, Y_r_2, p_2, X_e_2)
        @test isapprox(Y_r_2, ArrayPartition(X_r, X_r))
    end
    @testset "Hessian conversion" begin
        M = Sphere(2)
        N = M × M
        p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        q = 1 / sqrt(2) * ArrayPartition([1.0, 1.0, 0.0], [0.0, 1.0, 1.0])
        q = 1 / sqrt(2) * ArrayPartition([0.0, 1.0, 1.0], [1.0, 1.0, 0.0])
        r = 1 / sqrt(3) * ArrayPartition([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        X = log(M, p, q)
        Y = log(M, p, r)
        Z = -X
        H1 = riemannian_Hessian(N, p, Y, Z, X)
        H2 = ArrayPartition(
            [riemannian_Hessian(M, p.x[i], Y.x[i], Z.x[i], X.x[i]) for i in 1:2]...,
        )
        @test H1 == H2
        V = ArrayPartition([0.2, 0.0, 0.0], [0.0, 0.0, 0.3])
        W1 = Weingarten(N, p, X, V)
        W2 = ArrayPartition([Weingarten(M, p.x[i], X.x[i], V.x[i]) for i in 1:2]...)
        @test W1 == W2
    end
    @testset "Manifold volume" begin
        MS2 = Sphere(2)
        MS3 = Sphere(3)
        PM = ProductManifold(MS2, MS3)
        @test manifold_volume(PM) ≈ manifold_volume(MS2) * manifold_volume(MS3)
        p1 = [-0.9171596991960276, 0.39792260844341604, -0.02181017790481868]
        p2 = [
            -0.5427653626654726,
            5.420303965772687e-5,
            -0.8302022885580579,
            -0.12716099333369416,
        ]
        X1 = [-0.35333565579879633, -0.7896159441709865, 0.45204526334685574]
        X2 = [
            -0.33940201562492356,
            0.8092470417550779,
            0.18290591742514573,
            0.2548785571950708,
        ]
        @test volume_density(PM, ArrayPartition(p1, p2), ArrayPartition(X1, X2)) ≈
            volume_density(MS2, p1, X1) * volume_density(MS3, p2, X2)
    end
end
