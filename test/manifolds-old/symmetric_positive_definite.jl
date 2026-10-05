include("../header.jl")

@testset "Symmetric Positive Definite Matrices" begin
    M1 = SymmetricPositiveDefinite(3)
    @test repr(M1) == "SymmetricPositiveDefinite(3)"
    M2 = MetricManifold(SymmetricPositiveDefinite(3), Manifolds.AffineInvariantMetric())
    M3 = MetricManifold(SymmetricPositiveDefinite(3), Manifolds.LogCholeskyMetric())
    M4 = MetricManifold(SymmetricPositiveDefinite(3), Manifolds.LogEuclideanMetric())
    M5 = MetricManifold(SymmetricPositiveDefinite(3), Manifolds.BuresWassersteinMetric())
    M6 = MetricManifold(
        SymmetricPositiveDefinite(3),
        Manifolds.GeneralizedBuresWassersteinMetric(
            [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0],
        ),
    )

    @test !is_flat(M4)
    @test injectivity_radius(M1) == Inf
    @test injectivity_radius(M1, one(zeros(3, 3))) == Inf
    @test injectivity_radius(M1, ExponentialRetraction()) == Inf
    @test injectivity_radius(M1, one(zeros(3, 3)), ExponentialRetraction()) == Inf
    @test zero_vector(M1, one(zeros(3, 3))) == zero_vector(M2, one(zeros(3, 3)))
    @test zero_vector(M1, one(zeros(3, 3))) == zero_vector(M3, one(zeros(3, 3)))
    metrics = [M1, M2, M3, M5, M6]
    types = [Matrix{Float64}, MMatrix{3, 3, Float64, 9}, SPDPoint]

    for M in metrics
        basis_types = if (M == M1 || M == M2 || M == M3)
            (DefaultOrthonormalBasis(),)
        else
            ()
        end
        @testset "$(typeof(M))" begin
            @test representation_size(M) == (3, 3)
            if M === M3
                @test is_flat(M)
            else
                @test !is_flat(M)
            end
            for T in types
                exp_log_atol_multiplier = 8.0
                if M == M6
                    # we have to raise this slightly for the nondiagonal case.
                    exp_log_atol_multiplier = 5.0e1
                end
                if T == SPDPoint && (M != M1 && M != M2)
                    # SPDPoint only meant for Affine metric
                    continue
                end
                A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
                ptsF = [#
                    [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1],
                    [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1],
                    A(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1] * transpose(A(π / 6)),
                ]
                pts = [convert(T, a) for a in ptsF]
                Manifolds.test_manifold(
                    M,
                    pts;
                    vector_transport_methods = M isa SymmetricPositiveDefinite ?
                        [ParallelTransport()] : [],
                    exp_log_atol_multiplier = exp_log_atol_multiplier,
                    basis_types_vecs = basis_types,
                    basis_types_to_from = basis_types,
                    is_tangent_atol_multiplier = 1,
                    test_inplace = true,
                    test_rand_point = M === M1,
                    test_rand_tvector = M === M1,
                    test_default_vector_transport = !(M === M5 || M === M6),
                )
            end
            @testset "Test Error cases in is_point and is_vector" begin
                pt1f = zeros(2, 3) # wrong size
                pt2f = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0] # not positive Definite
                pt3f = [2.0 0.0 1.0; 0.0 1.0 0.0; 0.0 0.0 4.0] # not symmetric
                pt4 = [2.0 1.0 0.0; 1.0 2.0 0.0; 0.0 0.0 4.0]
                @test !is_point(M, pt1f)
                @test !is_point(M, pt2f)
                @test !is_point(M, pt3f)
                @test is_point(M, pt4)
                @test !is_vector(M, pt4, pt1f)
                @test is_vector(M, pt4, pt2f)
                @test !is_vector(M, pt4, pt3f)
            end
        end
    end
    p = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1]
    q = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1]
    @testset "Convert Eucl/Emb to (Generalised) BW" begin
        Z = ones(3, 3)
        em = EuclideanMetric()
        x = 2 .* (Z * p + p * Z)
        X = change_representer(M5, em, p, Z)
        @test isapprox(x, X)
        y = 2 .* (Z * p * M6.metric.M + M6.metric.M * p' * Z)
        Y = change_representer(M6, em, p, Z)
        @test isapprox(y, Y)
    end
    @testset "Bures-Wasserstein logarithm of points whose product is not symmetric" begin
        p2 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]
        X2 = [0.5 0.0 0.2; 0.0 -1.0 0.3; 0.2 0.3 0.5]
        @test isapprox(M5, p2, log(M5, p2, exp(M5, p2, X2)), X2)
    end
    @testset "Bures-Wasserstein distances are real numbers" begin
        @test distance(M6, q, q) isa Real
        A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
        B(α) = [cos(α) sin(α) 0.0; -sin(α) cos(α) 0.0; 0.0 0.0 1.0]
        p1 = A(π / 4) * [1.0e-6 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e6] * transpose(A(π / 4))
        q1 = B(π / 4) * [1.0e6 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e-6] * transpose(B(π / 4))
        for M in (M5, M6)
            @test distance(M, p1, p1) isa Real
            @test distance(M, p1, q1) isa Real
        end
    end
    @testset "Convert SPD to Cholesky" begin
        v = log(M1, p, q)
        (l, w) = Manifolds.spd_to_cholesky(p, v)
        (xs, vs) = Manifolds.cholesky_to_spd(l, w)
        @test isapprox(xs, p)
        @test isapprox(vs, v)
    end
    @testset "Orthonormal basis of the Log-Cholesky metric away from the identity" begin
        pc = [2.0 1.0 0.0; 1.0 2.0 0.0; 0.0 0.0 4.0]
        X = [1.0 1.0 0.5; 1.0 1.0 0.0; 0.5 0.0 1.0]
        c = get_coordinates(M3, pc, X, DefaultOrthonormalBasis())
        @test get_vector(M3, pc, c, DefaultOrthonormalBasis()) ≈ X
        Vs = get_vectors(M3, pc, get_basis(M3, pc, DefaultOrthonormalBasis()))
        @test [inner(M3, pc, V, W) for V in Vs, W in Vs] ≈ Matrix{Float64}(I, 6, 6)
    end
    @testset "Preliminary tests for LogEuclidean" begin
        @test representation_size(M4) == (3, 3)
        @test isapprox(distance(M4, p, q), sqrt(2) * log(2))
        @test manifold_dimension(M4) == manifold_dimension(M1)
    end
    @testset "Test for tangent ONB on AffineInvariantMetric" begin
        v = log(M2, p, q)
        donb = get_basis(base_manifold(M2), p, DiagonalizingOrthonormalBasis(v))
        Xs = get_vectors(base_manifold(M2), p, donb)
        k = donb.data.eigenvalues
        @test isapprox(0.0, first(k))
        for i in 1:length(Xs)
            @test isapprox(1.0, norm(M2, p, Xs[i]))
            for j in (i + 1):length(Xs)
                @test isapprox(0.0, inner(M2, p, Xs[i], Xs[j]))
            end
        end
        d2onb = get_basis(M2, p, DiagonalizingOrthonormalBasis(v))
        @test donb.data.eigenvalues == d2onb.data.eigenvalues
        @test get_vectors(base_manifold(M2), p, donb) == get_vectors(M2, p, d2onb)
    end
    @testset "Vector transport with Schild and Pole ladder" begin
        A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
        M = SymmetricPositiveDefinite(3)
        p1 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1]
        p2 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1]
        p3 = A(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1] * transpose(A(π / 6))
        @test embed(M, p1) == p1
        X1 = log(M, p1, p3)
        Y1 = vector_transport_to(M, p1, X1, p2)
        @test is_vector(M, p2, Y1)
        Y2 = vector_transport_to(M, p1, X1, p2, PoleLadderTransport())
        @test is_vector(M, p2, Y2, atol = 10^-16)
        Y3 = vector_transport_to(M, p1, X1, p2, SchildsLadderTransport())
        @test is_vector(M, p2, Y3)
        @test isapprox(M, p2, Y1, Y2) # pole is exact on SPDs, i.e. identical to parallel transport
        @test norm(M, p1, X1) ≈ norm(M, p2, Y1) # parallel transport is length preserving
        # test isometry
        X2 = log(M, p1, p2)
        Y4 = vector_transport_to(M, p1, X2, p2)
        @test norm(M, p1, X2) ≈ norm(M, p2, Y4)
        @test is_vector(M, p2, Y4)
        Y5 = vector_transport_to(M, p1, X2, p2, PoleLadderTransport())
        @test inner(M, p1, X1, X2) ≈ inner(M, p2, Y1, Y4) # parallel transport isometric
        @test inner(M, p1, X1, X2) ≈ inner(M, p2, Y2, Y5) # pole ladder transport isometric
    end
    @testset "Points that are symmetric up to rounding" begin
        Arot(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
        p_rot = Arot(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1] * transpose(Arot(π / 6))
        @test is_point(M1, exp(M1, p_rot, [1.0 1.0 0.5; 1.0 1.0 0.0; 0.5 0.0 1.0]))
        q_near = [2.0e3 1.0e3 0.0; nextfloat(1.0e3) 2.0e3 0.0; 0.0 0.0 4.0e3]
        @test is_point(M1, q_near)
        @test is_point(SymmetricPositiveDefinite(3; parameter = :field), q_near)
        @test !is_point(M1, [2.0 0.0 1.0; 0.0 1.0 0.0; 0.0 0.0 4.0])
    end
    @testset "Metric change for Linear Affine Metric" begin
        X = log(M1, p, q)
        Y = change_metric(M1, EuclideanMetric(), p, X)
        @test Y == p * X
        Z = change_representer(M1, EuclideanMetric(), p, X)
        @test Z == p * X * p
    end
    @testset "Affine invariant distance of points far apart" begin
        pD = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e-30]
        qD = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e-20]
        @test distance(M1, pD, qD) ≈ log(1.0e10)
        E = Matrix{Float64}(I, 3, 3)
        @test distance(M1, 1.0e-9 * E, 1.0e9 * E) ≈ sqrt(3) * log(1.0e18)
        @test distance(M1, 1.0e-20 * E, 2.0e-20 * E) ≈ sqrt(3) * log(2)
    end
    @testset "Projection on Tangent space" begin
        p = Matrix{Float64}(I, 3, 3)
        X = [1.0 2.0 1.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        Y = project(M1, p, X)
        @test is_vector(M1, p, Y)
    end
    @testset "Affine invariant logarithm between points far apart" begin
        E = Matrix{Float64}(I, 3, 3)
        X = log(M1, 1.0e9 * E, 1.0e-9 * E)
        @test norm(M1, 1.0e9 * E, X) ≈ sqrt(3) * log(1.0e18)
        @test exp(M1, 1.0e9 * E, X) ≈ 1.0e-9 * E
    end
    @testset "Tangent ONB" begin
        q = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1]
        b = DefaultOrthonormalBasis()
        B = get_basis(M1, q, b)
        for i in 1:length(B.data)
            @test norm(M1, q, B.data[i]) ≈ 1
            for j in (i + 1):length(B.data)
                @test inner(M1, q, B.data[i], B.data[j]) ≈ 0
            end
        end
        X = [1.0 1.0 0.5; 1.0 1.0 0.0; 0.5 0.0 1.0]
        c = get_coordinates(M1, q, X, b)
        X2 = get_vector(M1, q, c, b)
        @test isapprox(M1, q, X, X2)
    end
    @testset "rand()" begin
        p = rand(M1)
        @test is_point(M1, p)
        @test is_vector(M1, p, rand(M1; vector_at = p, tangent_distr = :Rician))
        @test is_vector(
            M1,
            p,
            rand(MersenneTwister(123), M1; vector_at = p, tangent_distr = :Rician),
        )
        # coordinates of a Gaussian tangent vector in an orthonormal basis at the point
        q = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]
        B = get_basis(M1, q, DiagonalizingOrthonormalBasis(Matrix{Float64}(I, 3, 3)))
        z = randn(MersenneTwister(42), 6)
        X = rand(MersenneTwister(42), M1; vector_at = q, σ = 2.0)
        @test get_coordinates(M1, q, X, B) ≈ 2 * z
        Y = rand(MersenneTwister(42), M1; vector_at = q)
        @test get_coordinates(M1, q, Y, B) ≈ z / sqrt(3)
        # the Rician draw at an SPDPoint equals the one at its matrix
        qS = SPDPoint(q)
        XR = rand(MersenneTwister(42), M1; vector_at = q, tangent_distr = :Rician)
        @test rand(MersenneTwister(42), M1; vector_at = qS, tangent_distr = :Rician) ≈ XR
        # a random point drawn into an SPDPoint stores its matrix square roots
        pR = rand!(MersenneTwister(42), M1, SPDPoint(q))
        @test pR.sqrt ≈ SPDPoint(pR.p).sqrt
        @test pR.sqrt_inv ≈ SPDPoint(pR.p).sqrt_inv
        pN = SPDPoint(q; store_sqrt = false, store_sqrt_inv = false)
        @test rand!(MersenneTwister(42), M1, pN).p == pR.p
    end
    @testset "metric" begin
        p = [
            1.3996531703810995 -0.050757455076129054 0.012468338878281887
            -0.050757455076129054 1.4934536968479577 -0.1322157270710227
            0.012468338878281887 -0.1322157270710227 1.1406534894519493
        ]
        X1 = [
            -0.040154529852424355 0.0940975617218614 -0.14399681300759132
            0.0940975617218614 -0.07959493647453719 0.039521032486382335
            -0.14399681300759132 0.039521032486382335 -0.08077209831531029
        ]
        X2 = [
            0.22320202697960773 0.1294383854675894 -0.15402580371166136
            0.1294383854675894 0.23251725583184363 0.13919342694607098
            -0.15402580371166136 0.139193426946071 0.4493904218883048
        ]
        X3 = [
            0.14955817329734716 0.04874731541620811 0.16142208646317371
            0.048747315416208095 0.4135413163173373 0.0813738001379617
            0.16142208646317371 0.0813738001379617 0.5381987409318881
        ]
        X_rt = [
            0.0026845159609378074 0.0008609241226852796 0.003549384063443887
            0.0008609241226852785 -0.0010229969873821812 -0.0014137071858029112
            0.003549384063443888 -0.0014137071858029105 -0.0011418891263384548
        ]

        @test isapprox(M1, p, riemann_tensor(M1, p, X1, X2, X3), X_rt)
    end
    @testset "SPDPoint functions" begin
        p = SPDPoint(2 * Matrix{Float64}(I, 3, 3))
        p2 = copy(p)
        @test SPDPoint(p2) === p2
        @test p2.eigen == p.eigen
        pS = SPDPoint(
            2 * Matrix{Float64}(I, 3, 3);
            store_p = false,
            store_sqrt = false,
            store_sqrt_inv = false,
        )
        m = missing
        s = "$(typeof(pS))\np:\n $m\np^{1/2}:\n $m\np^{-1/2}:\n $m"
        @test sprint(show, "text/plain", pS) == s
        pF = SPDPoint(Matrix{Float64}(I, 3, 3))
        copyto!(pF, pS) # fill values in F
        @test pF == p
        pF2 = SPDPoint(Matrix{Float64}(I, 3, 3))
        copyto!(pF2, pF) # copy values in F
        @test !ismissing(pF2.p)
        @test !ismissing(pF2.sqrt)
        @test !ismissing(pF2.sqrt_inv)
        pF3 = SPDPoint(
            Matrix{Float64}(I, 3, 3);
            store_p = false,
            store_sqrt = false,
            store_sqrt_inv = false,
        )
        copyto!(pF3, pF2) # do not fill values from F
        @test ismissing(pF3.p)
        @test ismissing(pF3.sqrt)
        @test ismissing(pF3.sqrt_inv)
        @test isapprox(Manifolds.spd_sqrt(pS), pF.sqrt) # recreate
        @test isapprox(Manifolds.spd_sqrt_inv(pF), pF.sqrt_inv) #identity
        pF4 = SPDPoint(2 * Matrix{Float64}(I, 3, 3); store_sqrt_inv = false)
        pF5 = SPDPoint(2 * Matrix{Float64}(I, 3, 3); store_sqrt = false)
        ssi = (pF.sqrt, pF.sqrt_inv)
        @test Manifolds.spd_sqrt_and_sqrt_inv(pF4) == ssi # comp sqrt inv
        @test Manifolds.spd_sqrt_and_sqrt_inv(pF5) == ssi # comp sqrt
        @test Manifolds.spd_sqrt_and_sqrt_inv(pS) == ssi # com both
        M = SymmetricPositiveDefinite(3)
        @test isapprox(exp!(M, pS, p, zero_vector(M, p)), p)
        @test ismissing(pS.sqrt)
        @test ismissing(pS.sqrt_inv)
        qR = SPDPoint(Matrix{Float64}(I, 3, 3); store_p = false)
        exp!(M, qR, p, ones(3, 3))
        @test isapprox(M, qR, exp(M, p, ones(3, 3)))
        @test exp(M, p, ones(3, 3)) == qR
        @test allocate_result(M1, zero_vector, p) isa Matrix
        c1 = ManifoldsBase.allocate_coordinates(M, p, Float64, 6)
        c2 = ManifoldsBase.allocate_coordinates(M, embed(M, p), Float64, 6)
        @test typeof(c1) == typeof(c2)
        @test length(c1) == length(c2)
    end

    @testset "test BigFloat" begin
        M = SymmetricPositiveDefinite(2)
        p1 = BigFloat[
            1.6590891025248637458133771360735408961772918701171875 -2.708777790960681386422947980463504791259765625e-7
            -2.708777790960681386422947980463504791259765625e-7 1.6590893171834280028775765458703972399234771728515625
        ]
        @test is_point(M, p1)
    end

    @testset "Riemannian Hessian Conversion" begin
        M = SymmetricPositiveDefinite(2)
        p = [2.0 1.0; 1.0 1.0]
        G = [1.0 0.0; 1.0 0.1]
        H = [2.0 1.0; 0.0 0.0]
        X = [0.0 3.0; 3.0 0.0]
        Y = X * 1 / 2 * (G' + G) * p
        Y = 1 / 2 * p * (H' + H) * p + 1 / 2 * (Y' + Y)
        @test riemannian_Hessian(M, p, G, H, X) == Y
    end

    @testset "Volume density" begin
        M = SymmetricPositiveDefinite(3)
        @test manifold_volume(M) == Inf
        p = [
            1.680908185710701 -0.030208936760309613 -0.284402826783584
            -0.030208936760309613 1.7199740873691465 -0.0066638025832747305
            -0.284402826783584 -0.0066638025832747305 1.6842441059901379
        ]
        X = [
            -1.3447374559982306 0.2591587910302816 0.9739140395169474
            0.2591587910302816 0.3649975914025044 -0.2888865063584093
            0.9739140395169474 -0.2888865063584093 -0.9564259306801289
        ]
        @test volume_density(M, p, X) ≈ 1.1326952644501451
    end
    @testset "field parameter" begin
        M = SymmetricPositiveDefinite(3; parameter = :field)
        @test typeof(get_embedding(M)) === Euclidean{ℝ, Tuple{Int, Int}}
        @test repr(M) == "SymmetricPositiveDefinite(3; parameter=:field)"
        @test Manifolds.get_parameter_type(M) === :field
    end

    @testset "Curvature" begin
        @test sectional_curvature_min(SymmetricPositiveDefinite(1)) == 0.0
        @test sectional_curvature_min(SymmetricPositiveDefinite(2)) == -0.5
        @test sectional_curvature_min(SymmetricPositiveDefinite(3)) == -0.5
        @test sectional_curvature_max(SymmetricPositiveDefinite(3)) == 0.0
        M = SymmetricPositiveDefinite(3)
        p = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        X = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -1.0] / sqrt(2)
        Y = [0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0] / sqrt(2)
        @test sectional_curvature(M, p, X, Y) ≈ sectional_curvature_min(M)
    end
end
