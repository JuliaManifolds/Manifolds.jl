include("../header.jl")

@testset "Rotations" begin
    M = Rotations(2)
    @test repr(M) == "Rotations(2)"
    @test representation_size(M) == (2, 2)
    @test is_flat(M)
    @test injectivity_radius(M) == π * sqrt(2.0)
    @test injectivity_radius(M, [1.0 0.0; 0.0 1.0]) == π * sqrt(2.0)
    @test injectivity_radius(M, ExponentialRetraction()) == π * sqrt(2.0)
    @test injectivity_radius(M, [1.0 0.0; 0.0 1.0], ExponentialRetraction()) ==
        π * sqrt(2.0)
    @test injectivity_radius(M, PolarRetraction()) ≈ π / sqrt(2)
    @test injectivity_radius(M, [1.0 0.0; 0.0 1.0], PolarRetraction()) ≈ π / sqrt(2)
    @test get_embedding(M) == Euclidean(2, 2)
    types = [Matrix{Float64}, SMatrix{2, 2, Float64, 4}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{2, 2, Float64, 4})
    retraction_methods = [PolarRetraction(), QRRetraction()]
    @test default_vector_transport_method(M) === ParallelTransport()

    inverse_retraction_methods = [PolarInverseRetraction(), QRInverseRetraction()]

    basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))

    @testset "vee/hat" begin
        M = Rotations(2)
        Xf = [1.23]
        p = Matrix{Float64}(I, 2, 2)
        X = hat(M, p, Xf)
        @test isa(X, AbstractMatrix)
        @test norm(M, p, X) / sqrt(2) ≈ norm(Xf)
        @test vee(M, p, X) == Xf

        X = project(M, p, randn(2, 2))
        Xf = vee(M, p, X)
        @test isa(Xf, AbstractVector)
        @test hat(M, p, Xf) == X
    end

    for T in types
        angles = (0.0, π / 2, 2π / 3, π / 4)
        pts = [convert(T, [cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]) for ϕ in angles]
        point_distributions =
            (T <: SMatrix) ? [] : [Manifolds.normal_rotation_distribution(M, pts[1], 1.0)]
        tvector_distributions =
            (T <: SMatrix) ? [] : [Manifolds.normal_tvector_distribution(M, pts[1], 1.0)]
        Manifolds.test_manifold(
            M,
            pts;
            is_mutating = !(T <: SMatrix),
            test_injectivity_radius = false,
            test_project_tangent = true,
            test_musical_isomorphisms = true,
            test_default_vector_transport = true,
            retraction_methods = retraction_methods,
            inverse_retraction_methods = inverse_retraction_methods,
            point_distributions,
            tvector_distributions,
            basis_types_to_from = basis_types,
            test_inplace = true,
            test_rand_point = true,
            retraction_atol_multiplier = 2,
            exp_log_atol_multiplier = 10,
        )

        @testset "log edge cases" begin
            X = Manifolds.hat(M, pts[1], [Float64(π)])
            p = exp(M, pts[1], X)
            @test isapprox(p, exp(M, pts[1], log(M, pts[1], p)))
        end

        X = log(M, pts[1], pts[2])
        @test norm(M, pts[1], X) ≈ (angles[2] - angles[1]) * sqrt(2)

        # check that exp! does not have a side effect
        q = allocate(pts[1])
        copyto!(M, q, pts[1])
        q2 = exp(M, pts[1], X)
        exp!(M, q, q, X)
        @test norm(q - q2) ≈ 0

        X14_polar = inverse_retract(M, pts[1], pts[4], PolarInverseRetraction())
        p4_polar = retract(M, pts[1], X14_polar, PolarRetraction())
        @test isapprox(M, pts[4], p4_polar)

        X14_qr = inverse_retract(M, pts[1], pts[4], QRInverseRetraction())
        p4_qr = retract(M, pts[1], X14_qr, QRRetraction())
        @test isapprox(M, pts[4], p4_qr)
    end

    @testset "Distribution tests" begin
        usd_mmatrix =
            Manifolds.normal_rotation_distribution(M, (@MMatrix [1.0 0.0; 0.0 1.0]), 1.0)
        @test isa(rand(usd_mmatrix), MMatrix)

        usd1_mmatrix =
            Manifolds.normal_rotation_distribution(Rotations(1), (@MMatrix [1.0]), 1.0)
        @test isa(rand(usd1_mmatrix), MMatrix)
        @test rand(usd1_mmatrix) == @MMatrix [1.0]

        gtsd_mvector =
            Manifolds.normal_tvector_distribution(M, (@MMatrix [1.0 0.0; 0.0 1.0]), 1.0)
        @test isa(rand(gtsd_mvector), MMatrix)

        Test.@test Distributions.support(usd_mmatrix).manifold == M
    end

    Random.seed!(42)
    for n in (3, 4, 5)
        @testset "Rotations: SO($n)" begin
            SOn = Rotations(n)
            @test !is_flat(SOn)
            ptd = Manifolds.normal_rotation_distribution(SOn, Matrix(1.0I, n, n), 1.0)
            tvd = Manifolds.normal_tvector_distribution(SOn, Matrix(1.0I, n, n), 1.0)
            pts = [rand(ptd) for _ in 1:3]
            diag_basis_1 = if n == 3
                DiagonalizingOrthonormalBasis(
                    [
                        0.0 0.24800271831269094 0.30019597622794186
                        -0.24800271831269094 0.0 -0.5902347224334308
                        -0.30019597622794186 0.5902347224334308 0.0
                    ],
                )
            else
                DiagonalizingOrthonormalBasis(rand(SOn; vector_at = pts[1]))
            end
            diag_basis_2 = DiagonalizingOrthonormalBasis(
                hat(SOn, pts[1], [1.0, zeros(manifold_dimension(SOn) - 1)...]),
            )
            Manifolds.test_manifold(
                SOn,
                pts;
                test_injectivity_radius = false,
                test_musical_isomorphisms = true,
                test_mutating_rand = true,
                test_default_vector_transport = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                point_distributions = [ptd],
                tvector_distributions = [tvd],
                basis_types_to_from = basis_types,
                basis_types_vecs = (diag_basis_1, diag_basis_2),
                exp_log_atol_multiplier = 250,
                retraction_atol_multiplier = 12,
                test_inplace = true,
                test_rand_point = true,
                test_rand_tvector = true,
            )

            @testset "vee/hat" begin
                p = Matrix(1.0I, n, n)
                Xf = randn(manifold_dimension(SOn))
                X = Manifolds.hat(SOn, p, Xf)
                @test isa(X, AbstractMatrix)
                @test norm(SOn, p, X) / sqrt(2) ≈ norm(Xf)
                @test Manifolds.vee(SOn, p, X) == Xf

                X = project(SOn, p, randn(n, n))
                Xf = Manifolds.vee(SOn, p, X)
                @test isa(Xf, AbstractVector)
                @test Manifolds.hat(SOn, p, Xf) ≈ X
            end
            X = Matrix(
                Manifolds.hat(SOn, pts[1], π * normalize(randn(manifold_dimension(SOn)))),
            )
            p = exp(SOn, pts[1], X)
            X2 = log(SOn, pts[1], p)
            @test distance(SOn, p, exp(SOn, pts[1], X2)) < 25 * eps()
            p2 = ManifoldsBase.exp_fused(SOn, pts[1], X, 1.0)
            X3 = log(SOn, pts[1], p)
            @test distance(SOn, p, exp(SOn, pts[1], X3)) < 25 * eps()
        end
    end
    @testset "Test AbstractManifold Point and Tangent Vector checks" begin
        M = Rotations(2)
        for p in [1, [2.0 0.0; 0.0 1.0], [1.0 0.5; 0.0 1.0]]
            @test_throws DomainError is_point(M, p; error = :error)
            @test !is_point(M, p)
        end
        p = one(zeros(2, 2))
        @test is_point(M, p)
        @test is_point(M, p; error = :error)
        for X in [1, [0.0 1.0; 0.0 0.0]]
            @test_throws DomainError is_vector(M, p, X; error = :error)
            @test !is_vector(M, p, X)
        end
        X = [0.0 1.0; -1.0 0.0]
        @test is_vector(M, p, X)
        @test is_vector(M, p, X; error = :error)
    end
    @testset "Project point" begin
        M = Rotations(2)
        p = Matrix{Float64}(I, 2, 2)
        p1 = project(M, p)
        @test is_point(M, p1; error = :error)

        M = Rotations(3)
        p = collect(reshape(1.0:9.0, (3, 3)))
        p2 = project(M, p)
        @test is_point(M, p2; error = :error)

        rng = MersenneTwister(44)
        x3 = project(M, randn(rng, 3, 3))
        @test is_point(M, x3; error = :error)
    end
    @testset "Convert from Lie algebra representation of tangents to Riemannian submanifold representation" begin
        M = Rotations(3)
        p = project(M, collect(reshape(1.0:9.0, (3, 3))))
        x = [[0, -1, 3] [1, 0, 2] [-3, -2, 0]]
        @test is_vector(M, p, x; error = :error)
        @test embed(M, p, x) == p * x
        Y = zeros((3, 3))
        embed!(M, Y, p, x)
        @test Y == p * x
        @test Y ≈ p * (p'Y - Y'p) / 2
    end
    @testset "Edge cases of Rotations" begin
        @test_throws OutOfInjectivityRadiusError inverse_retract(
            Rotations(2),
            [1.0 0.0; 0.0 1.0],
            [1.0 0.0; 0.0 -1.0],
            PolarInverseRetraction(),
        )
    end

    @testset "Rotations(1)" begin
        M = Rotations(1)
        p = fill(1.0, 1, 1)
        X = get_vector(M, p, Float64[], DefaultOrthonormalBasis())
        @test X isa Matrix{Float64}
        @test X == fill(0.0, 1, 1)
        Xc = get_coordinates(M, p, X, DefaultOrthonormalBasis())
        @test length(Xc) == 0
        @test Xc isa Vector{Float64}

        @test injectivity_radius(M) == 0.0
        @test injectivity_radius(M, p) == 0.0
        @test injectivity_radius(M, ExponentialRetraction()) == 0.0
        @test injectivity_radius(M, p, ExponentialRetraction()) == 0.0
        @test injectivity_radius(M, PolarRetraction()) == 0.0
        @test injectivity_radius(M, p, PolarRetraction()) == 0.0
    end
    @testset "Riemannian Hessian" begin
        M = Rotations(2)
        p = Matrix{Float64}(I, 2, 2)
        X = [0.0 3.0; -3.0 0.0]
        V = [1.0 0.0; 1.0 0.0]
        @test Weingarten(M, p, X, V) == -1 / 2 * p * (V' * X - X' * V)
        G = [0.0 1.0; 0.0 0.0]
        H = [0.0 0.0; 2.0 0.0]
        @test riemannian_Hessian(M, p, G, H, X) == [0.0 -1.0; 1.0 0.0]
    end

    @testset "riemann_tensor" begin
        M = Rotations(3)
        p = [
            -0.5908399013383766 -0.6241917041179139 0.5111681988316876
            -0.7261666986267721 0.13535732881097293 -0.6740625485388226
            0.35155388888753836 -0.7694563730631729 -0.5332417398896261
        ]
        X = [
            0.0 -0.30777760628130063 0.5499897386953444
            0.30777760628130063 0.0 -0.32059980100053004
            -0.5499897386953444 0.32059980100053004 0.0
        ]
        Y = [
            0.0 -0.4821890003925358 -0.3513148535122392
            0.4821890003925358 0.0 0.37956770358148356
            0.3513148535122392 -0.37956770358148356 0.0
        ]
        Z = [
            0.0 0.3980141785048982 0.09735377380829331
            -0.3980141785048982 0.0 -0.576287216962475
            -0.09735377380829331 0.576287216962475 0.0
        ]
        @test riemann_tensor(M, p, X, Y, Z) ≈ [
            0.0 0.04818900625787811 -0.050996416671166
            -0.04818900625787811 0.0 0.024666891276861697
            0.050996416671166 -0.024666891276861697 0.0
        ]
    end

    @testset "sectional curvature" begin
        @test sectional_curvature_min(Rotations(3)) == 0.0
        @test sectional_curvature_max(Rotations(1)) == 0.0
        @test sectional_curvature_max(Rotations(2)) == 0.0
        @test sectional_curvature_max(Rotations(3)) == 1 / 8
        @test sectional_curvature_max(Rotations(4)) == 1 / 4
    end

    @testset "field parameter" begin
        M = Rotations(2; parameter = :field)
        @test is_flat(M)
        @test repr(M) == "Rotations(2; parameter=:field)"

        M = Rotations(1; parameter = :field)
        p = fill(1.0, 1, 1)
        X = get_vector(M, p, Float64[], DefaultOrthonormalBasis())
        @test X isa Matrix{Float64}
        @test X == fill(0.0, 1, 1)
    end

    @testset "Specializations" begin
        M = Rotations(2)
        p = Matrix{Float64}(I, 2, 2)
        X = [0.0 3.0; -3.0 0.0]
        @test parallel_transport_direction(M, p, X, X) === X

        M = Rotations(3)
        p = @SMatrix [
            -0.5908399013383766 -0.6241917041179139 0.5111681988316876
            -0.7261666986267721 0.13535732881097293 -0.6740625485388226
            0.35155388888753836 -0.7694563730631729 -0.5332417398896261
        ]
        X = @SMatrix [
            0.0 -0.30777760628130063 0.5499897386953444
            0.30777760628130063 0.0 -0.32059980100053004
            -0.5499897386953444 0.32059980100053004 0.0
        ]
        d = @SMatrix [
            0.0 -0.4821890003925358 -0.3513148535122392
            0.4821890003925358 0.0 0.37956770358148356
            0.3513148535122392 -0.37956770358148356 0.0
        ]
        @test parallel_transport_direction(M, p, X, d) ≈ [
            0.0 -0.3258778314599828 0.3903114578816008
            0.32587783145998306 0.0 -0.49138641089195584
            -0.3903114578816011 0.4913864108919558 0.0
        ]
    end
    @testset "Jacobians" begin
        M = Rotations(2)
        p = [
            0.38024046142595025 0.9248876642568981
            -0.9248876642568981 0.38024046142595014
        ]
        X = [
            0.0 0.40294834454872025
            -0.40294834454872025 0.0
        ]
        @test ManifoldDiff.jacobian_exp_argument(M, p, X) == @SMatrix [1]
        J = fill(0.0, 1, 1)
        ManifoldDiff.jacobian_exp_argument!(M, J, p, X)
        @test J == @SMatrix [1]

        M = Rotations(3)
        p = [
            0.8795914880107569 -0.39921238866388364 -0.2587436626398777
            0.4643792859455722 0.6024065774550045 0.6491981163124461
            -0.10329904648012433 -0.6911843344406933 0.7152576618394751
        ]
        X = [
            0.0 -0.5656000980668913 0.7650118907017176
            0.5656000980668913 0.0 0.9710556005789448
            -0.7650118907017176 -0.9710556005789448 0.0
        ]
        Jref = [
            0.8624842949736633 0.12898141742603422 -0.41055104894342076
            -0.3547043184601566 0.8081393303537744 -0.3494729262397322
            0.24366619850830287 0.4809472670341492 0.7678272125532423
        ]
        @test ManifoldDiff.jacobian_exp_argument(M, p, X) ≈ Jref
        J = fill(0.0, 3, 3)
        ManifoldDiff.jacobian_exp_argument!(M, J, p, X)
        @test J ≈ Jref
    end

    @testset "manifold_volume and volume_density" begin
        @test manifold_volume(Rotations(1)) ≈ 1
        @test manifold_volume(Rotations(2)) ≈ 2 * π * sqrt(2)
        @test manifold_volume(Rotations(3)) ≈ 8 * π^2 * sqrt(2)
        @test manifold_volume(Rotations(4)) ≈ (2 * π)^4 * sqrt(2)
        @test manifold_volume(Rotations(5)) ≈ 4 * (2 * π)^6 / 6 * sqrt(2)

        M = Rotations(3)
        p = [
            -0.5908399013383766 -0.6241917041179139 0.5111681988316876
            -0.7261666986267721 0.13535732881097293 -0.6740625485388226
            0.35155388888753836 -0.7694563730631729 -0.5332417398896261
        ]
        X = [
            0.0 -0.30777760628130063 0.5499897386953444
            0.30777760628130063 0.0 -0.32059980100053004
            -0.5499897386953444 0.32059980100053004 0.0
        ]
        @test volume_density(M, p, X) ≈ 0.8440563052346255
        @test volume_density(M, p, zero(X)) ≈ 1.0

        M = Rotations(4)
        p = [
            -0.09091199873970474 -0.5676546886791307 -0.006808638869334249 0.8182034009599919
            -0.8001176365300662 0.3161567169523502 -0.4938592872334223 0.12633171594159726
            -0.5890394255366699 -0.2597679221590146 0.7267279425385695 -0.23962403743004465
            -0.0676707570677516 -0.7143764493344514 -0.4774129704812182 -0.5071132150619608
        ]
        X = [
            0.0 0.2554704296965055 0.26356215573144676 -0.4070678736115306
            -0.2554704296965055 0.0 -0.04594199053786204 -0.10586374034761421
            -0.26356215573144676 0.04594199053786204 0.0 0.43156436122007846
            0.4070678736115306 0.10586374034761421 -0.43156436122007846 0.0
        ]
        @test volume_density(M, p, X) ≈ 0.710713830700454
    end
end
