include("../utils.jl")

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
    types = [Matrix{Float64}, SMatrix{2,2,Float64,4}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{2,2,Float64,4})
    retraction_methods = [PolarRetraction(), QRRetraction()]

    inverse_retraction_methods = [PolarInverseRetraction(), QRInverseRetraction()]

    basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))

    @testset "vee/hat" begin
        M = Rotations(2)
        Xf = [1.23]
        p = Matrix{Float64}(I, 2, 2)
        X = Manifolds.hat(M, p, Xf)
        @test isa(X, AbstractMatrix)
        @test norm(M, p, X) / sqrt(2) ≈ norm(Xf)
        @test Manifolds.vee(M, p, X) == Xf

        X = project(M, p, randn(2, 2))
        Xf = Manifolds.vee(M, p, X)
        @test isa(Xf, AbstractVector)
        @test Manifolds.hat(M, p, Xf) == X
    end

    for T in types
        angles = (0.0, π / 2, 2π / 3, π / 4)
        pts = [convert(T, [cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]) for ϕ in angles]
        point_distributions =
            (T <: SMatrix) ? [] : [Manifolds.normal_rotation_distribution(M, pts[1], 1.0)]
        tvector_distributions =
            (T <: SMatrix) ? [] : [Manifolds.normal_tvector_distribution(M, pts[1], 1.0)]
        test_manifold(
            M,
            pts;
            is_mutating=!(T <: SMatrix),
            test_injectivity_radius=false,
            test_project_tangent=true,
            test_musical_isomorphisms=true,
            test_default_vector_transport=true,
            retraction_methods=retraction_methods,
            inverse_retraction_methods=inverse_retraction_methods,
            point_distributions,
            tvector_distributions,
            basis_types_to_from=basis_types,
            test_inplace=true,
            test_rand_point=true,
            retraction_atol_multiplier=2,
            exp_log_atol_multiplier=10,
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
                DiagonalizingOrthonormalBasis(rand(SOn; vector_at=pts[1]))
            end
            diag_basis_2 = DiagonalizingOrthonormalBasis(
                hat(SOn, pts[1], [1.0, zeros(manifold_dimension(SOn) - 1)...]),
            )
            test_manifold(
                SOn,
                pts;
                test_injectivity_radius=false,
                test_musical_isomorphisms=true,
                test_mutating_rand=true,
                test_default_vector_transport=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                point_distributions=[ptd],
                tvector_distributions=[tvd],
                basis_types_to_from=basis_types,
                basis_types_vecs=(diag_basis_1, diag_basis_2),
                exp_log_atol_multiplier=250,
                retraction_atol_multiplier=12,
                test_inplace=true,
                test_rand_point=true,
                test_rand_tvector=true,
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
            @test p ≈ exp(SOn, pts[1], X2)
        end
    end
    @testset "Test AbstractManifold Point and Tangent Vector checks" begin
        M = Rotations(2)
        for p in [1, [2.0 0.0; 0.0 1.0], [1.0 0.5; 0.0 1.0]]
            @test_throws DomainError is_point(M, p, true)
            @test !is_point(M, p)
        end
        p = one(zeros(2, 2))
        @test is_point(M, p)
        @test is_point(M, p, true)
        for X in [1, [0.0 1.0; 0.0 0.0]]
            @test_throws DomainError is_vector(M, p, X, true)
            @test !is_vector(M, p, X)
        end
        X = [0.0 1.0; -1.0 0.0]
        @test is_vector(M, p, X)
        @test is_vector(M, p, X, true)
    end
    @testset "Project point" begin
        M = Rotations(2)
        p = Matrix{Float64}(I, 2, 2)
        p1 = project(M, p)
        @test is_point(M, p1, true)

        M = Rotations(3)
        p = collect(reshape(1.0:9.0, (3, 3)))
        p2 = project(M, p)
        @test is_point(M, p2, true)

        rng = MersenneTwister(44)
        x3 = project(M, randn(rng, 3, 3))
        @test is_point(M, x3, true)
    end
    @testset "Convert from Lie algebra representation of tangents to Riemannian submanifold representation" begin
        M = Rotations(3)
        p = project(M, collect(reshape(1.0:9.0, (3, 3))))
        x = [[0, -1, 3] [1, 0, 2] [-3, -2, 0]]
        @test is_vector(M, p, x, true)
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
end
