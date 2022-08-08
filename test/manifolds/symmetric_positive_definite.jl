include("../utils.jl")

@testset "Symmetric Positive Definite Matrices" begin
    M1 = SymmetricPositiveDefinite(3)
    @test repr(M1) == "SymmetricPositiveDefinite(3)"
    M2 = MetricManifold(SymmetricPositiveDefinite(3), Manifolds.LinearAffineMetric())
    M3 = MetricManifold(SymmetricPositiveDefinite(3), Manifolds.LogCholeskyMetric())
    M4 = MetricManifold(SymmetricPositiveDefinite(3), Manifolds.LogEuclideanMetric())
    M5 = MetricManifold(SymmetricPositiveDefinite(3), Manifolds.BuresWassersteinMetric())
    M6 = MetricManifold(
        SymmetricPositiveDefinite(3),
        Manifolds.GeneralizedBuresWassersteinMetric(
            [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0],
        ),
    )

    @test injectivity_radius(M1) == Inf
    @test injectivity_radius(M1, one(zeros(3, 3))) == Inf
    @test injectivity_radius(M1, ExponentialRetraction()) == Inf
    @test injectivity_radius(M1, one(zeros(3, 3)), ExponentialRetraction()) == Inf
    @test zero_vector(M1, one(zeros(3, 3))) == zero_vector(M2, one(zeros(3, 3)))
    @test zero_vector(M1, one(zeros(3, 3))) == zero_vector(M3, one(zeros(3, 3)))
    metrics = [M1, M2, M3, M5, M6]
    types = [Matrix{Float64}, SPDPoint]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{3,3,Float64,9})

    for M in metrics
        basis_types = if (M == M1 || M == M2)
            (DefaultOrthonormalBasis(),)
        else
            ()
        end
        @testset "$(typeof(M))" begin
            @test representation_size(M) == (3, 3)
            for T in types
                exp_log_atol_multiplier = 8.0
                if T <: MMatrix{3,3,Float64}
                    # eigendecomposition of 3x3 SPD matrices from StaticArrays is not very accurate
                    exp_log_atol_multiplier = 5.0e7
                end
                if M == M6
                    # we have to raise this slightly for the nondiagonal case.
                    exp_log_atol_multiplier = 5e1
                end
                if T == SPDPoint && (M != M1 && M != M2)
                    # SPDPoint only meant for Affine metric
                    continue
                end
                if M == M3 && T <: MMatrix
                    # Cholesky or something does not work in vector_transport yet for MMatrix
                    continue
                end
                A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
                ptsF = [#
                    [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1],
                    [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1],
                    A(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1] * transpose(A(π / 6)),
                ]
                pts = [convert(T, a) for a in ptsF]
                test_manifold(
                    M,
                    pts;
                    vector_transport_methods=typeof(M) == SymmetricPositiveDefinite{3} ?
                                             [ParallelTransport()] : [],
                    test_vee_hat=M === M2,
                    exp_log_atol_multiplier=exp_log_atol_multiplier,
                    basis_types_vecs=basis_types,
                    basis_types_to_from=basis_types,
                    is_tangent_atol_multiplier=1,
                    test_inplace=true,
                    test_rand_point=M === M1,
                    test_rand_tvector=M === M1,
                    test_default_vector_transport=!(M === M5 || M === M6),
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
    @testset "Convert SPD to Cholesky" begin
        v = log(M1, p, q)
        (l, w) = Manifolds.spd_to_cholesky(p, v)
        (xs, vs) = Manifolds.cholesky_to_spd(l, w)
        @test isapprox(xs, p)
        @test isapprox(vs, v)
    end
    @testset "Preliminary tests for LogEuclidean" begin
        @test representation_size(M4) == (3, 3)
        @test isapprox(distance(M4, p, q), sqrt(2) * log(2))
        @test manifold_dimension(M4) == manifold_dimension(M1)
    end
    @testset "Test for tangent ONB on LinearAffineMetric" begin
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
    @testset "Vector transport and transport along with Schild and Pole ladder" begin
        A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
        M = SymmetricPositiveDefinite(3)
        p1 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1]
        p2 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1]
        p3 = A(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1] * transpose(A(π / 6))
        X1 = log(M, p1, p3)
        Y1 = vector_transport_to(M, p1, X1, p2)
        @test is_vector(M, p2, Y1)
        Y2 = vector_transport_to(M, p1, X1, p2, PoleLadderTransport())
        @test is_vector(M, p2, Y2, atol=10^-16)
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
    @testset "Metric change for Linear Affine Metric" begin
        X = log(M1, p, q)
        Y = change_metric(M1, EuclideanMetric(), p, X)
        @test Y == p * X
        Z = change_representer(M1, EuclideanMetric(), p, X)
        @test Z == p * X * p
    end
    @testset "Projection on Tangent space" begin
        p = Matrix{Float64}(I, 3, 3)
        X = [1.0 2.0 1.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        Y = project(M1, p, X)
        @test is_vector(M1, p, Y)
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
        @test is_vector(M1, p, rand(M1; vector_at=p, tangent_distr=:Rician))
        @test is_vector(
            M1,
            p,
            rand(MersenneTwister(123), M1; vector_at=p, tangent_distr=:Rician),
        )
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
            store_p=false,
            store_sqrt=false,
            store_sqrt_inv=false,
        )
        m = missing
        s = "$(typeof(pS))\np:\n $m\np^{1/2}:\n $m\np^{-1/2}:\n $m"
        @test sprint(show, "text/plain", pS) == s
        pF = SPDPoint(Matrix{Float64}(I, 3, 3))
        copyto!(pF, pS) # fill values in F
        @test pF == p
        pF2 = SPDPoint(Matrix{Float64}(I, 3, 3))
        copyto!(pF2, pF) # copy alues in F
        @test !ismissing(pF2.p)
        @test !ismissing(pF2.sqrt)
        @test !ismissing(pF2.sqrt_inv)
        pF3 = SPDPoint(
            Matrix{Float64}(I, 3, 3);
            store_p=false,
            store_sqrt=false,
            store_sqrt_inv=false,
        )
        copyto!(pF3, pF2) # do not fill values from F
        @test ismissing(pF3.p)
        @test ismissing(pF3.sqrt)
        @test ismissing(pF3.sqrt_inv)
        @test isapprox(Manifolds.spd_sqrt(pS), pF.sqrt) # recreate
        @test isapprox(Manifolds.spd_sqrt_inv(pF), pF.sqrt_inv) #identity
        pF4 = SPDPoint(2 * Matrix{Float64}(I, 3, 3); store_sqrt_inv=false)
        pF5 = SPDPoint(2 * Matrix{Float64}(I, 3, 3); store_sqrt=false)
        ssi = (pF.sqrt, pF.sqrt_inv)
        @test Manifolds.spd_sqrt_and_sqrt_inv(pF4) == ssi # comp sqrt inv
        @test Manifolds.spd_sqrt_and_sqrt_inv(pF5) == ssi # comp sqrt
        @test Manifolds.spd_sqrt_and_sqrt_inv(pS) == ssi # com both
        M = SymmetricPositiveDefinite(3)
        @test isapprox(exp!(M, pS, p, zero_vector(M, p)), p)
        @test ismissing(pS.sqrt)
        @test ismissing(pS.sqrt_inv)
    end
end
