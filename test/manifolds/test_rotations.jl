using Distributions, LinearAlgebra, ManifoldDiff, Manifolds, ManifoldsBase, Random
using RecursiveArrayTools, StaticArrays, Test

@testset "The rotations manifold" begin
    @test_throws DomainError Rotations(-1)

    rot(ϕ) = [cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]

    @testset "SO(2) for $T" for T in [Matrix{Float64}, SMatrix{2, 2, Float64, 4}]
        M = Rotations(2)
        mutating = !(T <: SMatrix)
        p1, p2, p3, p4 = [convert(T, rot(ϕ)) for ϕ in (0.0, π / 4, π / 2, 2π / 3)]
        X = convert(T, [0.0 -π / 4; π / 4 0.0]) # log(M, p1, p2)
        Y = convert(T, [0.0 -1.0; 1.0 0.0])
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [[1.0]],
                :Covectors => [flat(M, p1, X)],
                :EmbeddedPoints => [2 .* p1],
                :EmbeddedVectors => [convert(T, [0.0 -0.5; 1.5 0.3])],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                    default_vector_transport_method, distance, embed, exp, flat, geodesic,
                    get_basis, get_coordinates, get_embedding, get_vector, get_vectors,
                    injectivity_radius, inner, inverse_retract, is_flat, is_point, is_vector,
                    log, manifold_dimension, manifold_volume, mid_point, norm,
                    parallel_transport_direction, parallel_transport_to, project, rand, repr,
                    representation_size, retract, sharp, shortest_geodesic,
                    vector_transport_direction, vector_transport_to, volume_density,
                    Weingarten, zero_vector,
                ],
                :InvalidPoints => [convert(T, [2.0 0.0; 0.0 1.0]), convert(T, [1.0 0.5; 0.0 1.0])],
                :InvalidVectors => [convert(T, [0.0 1.0; 0.0 0.0])],
                :InverseRetractionMethods => [
                    LogarithmicInverseRetraction(), PolarInverseRetraction(), QRInverseRetraction(),
                ],
                :Mutating => mutating,
                :NormalVectors => [p1 * convert(T, [1.0 0.0; 0.0 2.0])],
                :Points => [p1, p2, p3, p4],
                :RetractionMethods => [ExponentialRetraction(), PolarRetraction(), QRRetraction()],
                :SecondVector => Y,
                :Vectors => [X],
                :VectorTransportMethods => [ParallelTransport()],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-13,
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => ExponentialRetraction(),
                default_vector_transport_method => ParallelTransport(),
                distance => π / 4 * sqrt(2),
                exp => p2,
                get_coordinates => [π / 4 * sqrt(2)],
                get_embedding => Euclidean(2, 2),
                (get_vector, [1.0], DefaultOrthonormalBasis()) => Y ./ sqrt(2),
                (get_vectors, DefaultOrthonormalBasis()) => :Orthonormal,
                injectivity_radius => π * sqrt(2),
                (injectivity_radius, ExponentialRetraction()) => π * sqrt(2),
                (injectivity_radius, PolarRetraction()) => π / sqrt(2),
                inner => π / 2,
                is_flat => true,
                log => X,
                manifold_dimension => 1,
                manifold_volume => 2 * π * sqrt(2),
                mid_point => convert(T, rot(π / 8)),
                norm => π / 4 * sqrt(2),
                parallel_transport_to => X,
                parallel_transport_direction => X,
                (project, :Point) => p1,
                (project, :Vector) => convert(T, [0.0 -1.0; 1.0 0.0]),
                repr => "Rotations(2)",
                representation_size => (2, 2),
                sharp => X,
                (vector_transport_to, ParallelTransport()) => X,
                volume_density => 1.0,
                Weingarten => convert(T, [0.0 3π / 8; -3π / 8 0.0]),
                :IsPointErrors => [DomainError, DomainError],
                :IsVectorErrors => [DomainError],
                :IsVectorBasepointError => ManifoldDomainError,
            ),
        )

        @testset "log edge cases" begin
            Z = hat(M, p1, [Float64(π)])
            p = exp(M, p1, Z)
            @test isapprox(p, exp(M, p1, log(M, p1, p)))
        end
        @test norm(M, p1, log(M, p1, p4)) ≈ 2π / 3 * sqrt(2)
    end

    @testset "SO(2) injectivity radius at a point" begin
        M = Rotations(2)
        p = [1.0 0.0; 0.0 1.0]
        @test injectivity_radius(M, p, ExponentialRetraction()) == π * sqrt(2.0)
        @test injectivity_radius(M, p, PolarRetraction()) ≈ π / sqrt(2)
    end

    @testset "vee/hat" begin
        for n in (2, 3, 4, 5)
            M = Rotations(n)
            p = Matrix(1.0I, n, n)
            Xf = randn(MersenneTwister(n), manifold_dimension(M))
            X = hat(M, p, Xf)
            @test isa(X, AbstractMatrix)
            @test norm(M, p, X) / sqrt(2) ≈ norm(Xf)
            @test vee(M, p, X) == Xf

            X = project(M, p, randn(MersenneTwister(n + 10), n, n))
            Xf = vee(M, p, X)
            @test isa(Xf, AbstractVector)
            @test hat(M, p, Xf) ≈ X
        end
    end

    @testset "Distribution tests" begin
        M = Rotations(2)
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

    @testset "SO($n)" for n in (3, 4, 5)
        M = Rotations(n)
        rng = MersenneTwister(42)
        p1 = rand(rng, M)
        # points close enough to p1 for the polar and QR inverse retractions
        pts = [p1, [exp(M, p1, 0.5 .* rand(rng, M; vector_at = p1)) for _ in 1:2]...]
        X = 0.5 .* rand(rng, M; vector_at = pts[1])
        Y = rand(rng, M; vector_at = pts[1])
        Xs = [X, rand(rng, M; vector_at = pts[2]), rand(rng, M; vector_at = pts[3])]
        d = manifold_dimension(M)
        diag_basis_1 = DiagonalizingOrthonormalBasis(Y)
        diag_basis_2 = DiagonalizingOrthonormalBasis(hat(M, pts[1], [1.0, zeros(d - 1)...]))
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [collect(1.0:d) ./ d],
                :Covectors => [flat(M, pts[1], X)],
                :EmbeddedPoints => [pts[1] + 0.1 .* ones(n, n)],
                :EmbeddedVectors => [randn(rng, n, n)],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                    default_vector_transport_method, distance, embed, exp, flat, geodesic,
                    get_basis, get_coordinates, get_embedding, get_vector, get_vectors,
                    injectivity_radius, inner, inverse_retract, is_flat, is_point, is_vector,
                    log, manifold_dimension, manifold_volume, mid_point, norm,
                    parallel_transport_direction, parallel_transport_to, project, rand, repr,
                    representation_size, retract, sectional_curvature, sharp, shortest_geodesic,
                    vector_transport_direction, vector_transport_to, volume_density, zero_vector,
                ],
                :InvalidPoints => [2 .* pts[1], pts[1] * Diagonal([-1.0; ones(n - 1)])],
                :InvalidVectors => [ones(n, n)],
                :InverseRetractionMethods => [
                    LogarithmicInverseRetraction(), PolarInverseRetraction(), QRInverseRetraction(),
                ],
                :Points => pts,
                :RetractionMethods => [ExponentialRetraction(), PolarRetraction(), QRRetraction()],
                :SecondVector => Y,
                :Vectors => Xs,
                :VectorTransportMethods => [ParallelTransport()],
                :Seed => 42,
                :TestRandScaling => true,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => ExponentialRetraction(),
                default_vector_transport_method => ParallelTransport(),
                get_embedding => Euclidean(n, n),
                (get_vectors, DefaultOrthonormalBasis()) => :Orthonormal,
                injectivity_radius => π * sqrt(2),
                is_flat => false,
                manifold_dimension => d,
                repr => "Rotations($n)",
                representation_size => (n, n),
                sectional_curvature_min => 0.0,
                sectional_curvature_max => n == 3 ? 1 / 8 : 1 / 4,
                :IsPointErrors => [DomainError, DomainError],
                :IsVectorErrors => [DomainError],
                :IsVectorBasepointError => ManifoldDomainError,
            ),
        )

        @testset "bases only available through get_basis" begin
            p = pts[1]
            for B in [diag_basis_1, diag_basis_2]
                b = get_basis(M, p, B)
                @test b isa CachedBasis
                V = get_vectors(M, p, b)
                @test length(V) == d
                @test all(is_vector(M, p, v) for v in V)
                @test [inner(M, p, v, w) for v in V, w in V] ≈ I(d) atol = 1.0e-12
                c = get_coordinates(M, p, X, b)
                @test isapprox(M, p, get_vector(M, p, c, b), X; atol = 1.0e-12)
            end

            b = get_basis(M, p, ProjectedOrthonormalBasis(:svd))
            @test b isa CachedBasis
            V = get_vectors(M, p, b)
            @test length(V) == d
            @test [inner(M, p, v, w) for v in V, w in V] ≈ I(d) atol = 1.0e-12
            # ManifoldsBase builds this basis from the rows instead of the columns of the
            # projected ambient basis, which only agree for symmetric projections
            @test_broken all(is_vector(M, p, v) for v in V)
            @test_broken isapprox(M, p, get_vector(M, p, get_coordinates(M, p, X, b), b), X)
        end

        @testset "distributions" begin
            pt_d = Manifolds.normal_rotation_distribution(M, Matrix(1.0I, n, n), 1.0)
            tvd = Manifolds.normal_tvector_distribution(M, Matrix(1.0I, n, n), 1.0)
            q = rand(pt_d)
            @test is_point(M, q)
            rand!(pt_d, q)
            @test is_point(M, q)
            Z = rand(tvd)
            @test is_vector(M, Matrix(1.0I, n, n), Z)
            rand!(tvd, Z)
            @test is_vector(M, Matrix(1.0I, n, n), Z)
        end

        @testset "exp and log at distance π" begin
            Z = Matrix(hat(M, pts[1], π * normalize(randn(rng, d))))
            p = exp(M, pts[1], Z)
            Z2 = log(M, pts[1], p)
            @test distance(M, p, exp(M, pts[1], Z2)) < 100 * eps()
            p2 = ManifoldsBase.exp_fused(M, pts[1], Z, 1.0)
            @test isapprox(M, p2, p)
            Z3 = log(M, pts[1], p2)
            @test distance(M, p2, exp(M, pts[1], Z3)) < 100 * eps()
        end

        @testset "gradient and metric conversion" begin
            @test change_metric(M, EuclideanMetric(), pts[1], X) == X
            @test change_representer(M, EuclideanMetric(), pts[1], X) == X
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
            Rotations(2), [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 -1.0], PolarInverseRetraction()
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
        q = [0.0 -1.0; 1.0 0.0]
        @test Weingarten(M, q, X, q * [1.0 0.0; 0.0 2.0]) == [0.0 -4.5; 4.5 0.0]
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
        # a tangent vector whose angle squared underflows to zero
        Xs = 1.0e-170 .* X
        @test exp(M, p, Xs) ≈ p
        @test exp(M, Matrix(p), Matrix(Xs)) ≈ p
        @test parallel_transport_direction(M, p, X, Xs) ≈ X
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
        @test manifold_volume(Rotations(3)) ≈ 16 * π^2 * sqrt(2)
        @test manifold_volume(Rotations(4)) ≈ 8 * (2 * π)^4
        @test manifold_volume(Rotations(5)) ≈ 128 * (2 * π)^6 / 6
        @test volume_density(Rotations(2), [1.0 0.0; 0.0 1.0], [0.0 0.5; -0.5 0.0]) == 1.0
        @test volume_density(OrthogonalMatrices(2), [1.0 0.0; 0.0 1.0], [0.0 0.5; -0.5 0.0]) == 1.0

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
        @test volume_density(M, p, X) ≈ 0.9590216116974793
        @test volume_density(M, p, zero(X)) ≈ 1.0
        @test volume_density(M, p, 1.0e-12 * X) ≈ 1.0

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
        @test volume_density(M, p, X) ≈ 0.9195936931567809
        # at this vector some zero eigenvalues are computed as rounding errors
        p = Matrix{Float64}(I, 4, 4)
        X = get_vector(M, p, [-0.94, 0.45, 0.83, -0.66, -0.66, 0.83], DefaultOrthonormalBasis())
        @test volume_density(M, p, X) ≈ 0.7531906207139224
    end
end
