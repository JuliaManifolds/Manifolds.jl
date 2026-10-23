using LinearAlgebra, Manifolds, ManifoldsBase, Random, StaticArrays, Test

@testset "The manifold of symmetric positive definite matrices" begin
    @test_throws DomainError SymmetricPositiveDefinite(-1)

    M = SymmetricPositiveDefinite(3)
    A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
    p1 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    p2 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]
    p3 = A(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0] * transpose(A(π / 6))
    X = [log(2) 0.0 0.0; 0.0 log(2) 0.0; 0.0 0.0 0.0] # log(M, p1, p2) for all metrics but Bures-Wasserstein
    X2 = [1.0 1.0 0.5; 1.0 1.0 0.0; 0.5 0.0 1.0]
    Y = [0.5 0.0 0.2; 0.0 -1.0 0.3; 0.2 0.3 0.5]
    # Invalids
    pf1 = zeros(2, 3) # wrong size
    pf2 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0] # not positive definite
    pf3 = [2.0 0.0 1.0; 0.0 1.0 0.0; 0.0 0.0 4.0] # not symmetric
    invalid_points = [pf1, pf2, pf3]
    invalid_vectors = [pf1, pf3]
    is_point_errors = [ManifoldDomainError, DomainError, DomainError]
    is_vector_errors = [ManifoldDomainError, DomainError]
    basis = DefaultOrthonormalBasis()

    @testset "Affine invariant metric with $T points" for T in
        [Matrix{Float64}, MMatrix{3, 3, Float64, 9}, SPDPoint]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [basis],
                :Coordinates => [[log(2), 0.0, 0.0, log(2), 0.0, 0.0]],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method,
                    default_retraction_method, default_vector_transport_method, distance,
                    exp, geodesic, get_basis, get_coordinates, get_embedding, get_vector,
                    get_vectors, injectivity_radius, inner, inverse_retract, is_flat,
                    is_point, is_vector, log, manifold_dimension, mid_point, norm,
                    parallel_transport_direction, parallel_transport_to, rand, repr,
                    representation_size, retract, shortest_geodesic,
                    vector_transport_direction, vector_transport_to, zero_vector,
                    # embedding and curvature are not yet available for SPDPoint
                    (T === SPDPoint ? () : (embed, sectional_curvature))...,
                ],
                :InvalidPoints => invalid_points,
                :InvalidVectors => invalid_vectors,
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Points => [convert(T, p) for p in [p1, p2, p3]],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => X2,
                :Vectors => [X, Y, X2],
                :VectorTransportMethods => [
                    ParallelTransport(), PoleLadderTransport(), SchildsLadderTransport(),
                ],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => ExponentialRetraction(),
                default_vector_transport_method => ParallelTransport(),
                distance => sqrt(2) * log(2),
                exp => convert(T, p2),
                (get_coordinates, basis) => [log(2), 0.0, 0.0, log(2), 0.0, 0.0],
                (get_vectors, basis) => :Orthonormal,
                get_embedding => Euclidean(3, 3),
                injectivity_radius => Inf,
                (injectivity_radius, ExponentialRetraction()) => Inf,
                inner => 2 * log(2),
                is_flat => false,
                log => X,
                manifold_dimension => 6,
                mid_point => convert(T, [sqrt(2) 0.0 0.0; 0.0 sqrt(2) 0.0; 0.0 0.0 1.0]),
                norm => sqrt(2) * log(2),
                # parallel transport from the identity to p2 is X ↦ p2^(1/2) X p2^(1/2)
                parallel_transport_to => 2 * X,
                parallel_transport_direction => 2 * X,
                (vector_transport_to, ParallelTransport()) => 2 * X,
                # the pole ladder is exact on SPD matrices
                (vector_transport_to, PoleLadderTransport()) => 2 * X,
                repr => "SymmetricPositiveDefinite(3)",
                representation_size => (3, 3),
                sectional_curvature_min => -0.5,
                sectional_curvature_max => 0.0,
                :IsPointErrors => is_point_errors,
                :IsVectorErrors => is_vector_errors,
                :IsVectorBasepointError => ManifoldDomainError,
            ),
        )
    end

    @testset "Affine invariant metric as a metric manifold with $T points" for T in
        [Matrix{Float64}, SPDPoint]
        M2 = MetricManifold(M, AffineInvariantMetric())
        Manifolds.Test.test_manifold(
            M2,
            Dict(
                :Bases => [basis],
                :Coordinates => [[log(2), 0.0, 0.0, log(2), 0.0, 0.0]],
                :Functions => [
                    default_vector_transport_method, distance, exp, get_basis,
                    get_coordinates, get_vector, get_vectors, injectivity_radius, inner,
                    is_default_metric, is_flat, is_point, is_vector, log, manifold_dimension,
                    norm, parallel_transport_to, representation_size, vector_transport_to,
                    zero_vector,
                ],
                :InvalidPoints => invalid_points,
                :InvalidVectors => invalid_vectors,
                :OtherMetrics => [LogCholeskyMetric(), BuresWassersteinMetric()],
                :Points => [convert(T, p) for p in [p1, p2, p3]],
                :SecondVector => X2,
                :Vectors => [X, Y, X2],
                :VectorTransportMethods => [ParallelTransport()],
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                default_vector_transport_method => ParallelTransport(),
                distance => sqrt(2) * log(2),
                exp => convert(T, p2),
                (get_vectors, basis) => :Orthonormal,
                injectivity_radius => Inf,
                inner => 2 * log(2),
                is_default_metric => AffineInvariantMetric(),
                is_flat => false,
                log => X,
                manifold_dimension => 6,
                norm => sqrt(2) * log(2),
                parallel_transport_to => 2 * X,
                (vector_transport_to, ParallelTransport()) => 2 * X,
                representation_size => (3, 3),
                :IsPointErrors => is_point_errors,
                :IsVectorErrors => is_vector_errors,
            ),
        )
    end

    @testset "Log-Cholesky metric" begin
        M3 = MetricManifold(M, LogCholeskyMetric())
        Manifolds.Test.test_manifold(
            M3,
            Dict(
                :Bases => [basis],
                :Coordinates => [get_coordinates(M3, p1, X, basis)],
                :Functions => [
                    default_vector_transport_method, distance, exp, get_basis,
                    get_coordinates, get_vector, get_vectors, injectivity_radius, inner,
                    is_default_metric, is_flat, is_point, is_vector, log, manifold_dimension,
                    mid_point, norm, parallel_transport_to, representation_size,
                    sectional_curvature, vector_transport_to, zero_vector,
                ],
                :InvalidPoints => invalid_points,
                :InvalidVectors => invalid_vectors,
                :OtherMetrics => [AffineInvariantMetric()],
                :Points => [p1, p2, p3],
                :SecondVector => X2,
                :Vectors => [X, Y, X2],
                :VectorTransportMethods => [ParallelTransport()],
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                default_vector_transport_method => ParallelTransport(),
                distance => log(2) / sqrt(2),
                exp => p2,
                (get_vectors, basis) => :Orthonormal,
                injectivity_radius => Inf,
                is_default_metric => LogCholeskyMetric(),
                is_flat => true,
                log => X,
                manifold_dimension => 6,
                mid_point => [sqrt(2) 0.0 0.0; 0.0 sqrt(2) 0.0; 0.0 0.0 1.0],
                norm => log(2) / sqrt(2),
                representation_size => (3, 3),
                sectional_curvature => 0.0,
                sectional_curvature_min => 0.0,
                sectional_curvature_max => 0.0,
                :IsPointErrors => is_point_errors,
                :IsVectorErrors => is_vector_errors,
            ),
        )
        @testset "curvature" begin
            p = [2.0 1.0 0.0; 1.0 2.0 0.0; 0.0 0.0 1.0]
            Z1 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
            Z2 = [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
            @test riemann_tensor(M3, p, Z1, Z2, Z1) == zero(Z1)
            @test sectional_curvature(M3, p, Z1, Z2) === 0.0
        end
        @testset "Orthonormal basis away from the identity" begin
            pc = [2.0 1.0 0.0; 1.0 2.0 0.0; 0.0 0.0 4.0]
            c = get_coordinates(M3, pc, X2, basis)
            @test get_vector(M3, pc, c, basis) ≈ X2
            Vs = get_vectors(M3, pc, get_basis(M3, pc, basis))
            @test [inner(M3, pc, V, W) for V in Vs, W in Vs] ≈ Matrix{Float64}(I, 6, 6)
        end
    end

    @testset "Log-Euclidean metric" begin
        M4 = MetricManifold(M, LogEuclideanMetric())
        Manifolds.Test.test_manifold(
            M4,
            Dict(
                :Functions => [
                    distance, injectivity_radius, is_flat, is_point, is_vector,
                    manifold_dimension, representation_size,
                ],
                :InvalidPoints => invalid_points,
                :InvalidVectors => invalid_vectors,
                :Points => [p1, p2, p3],
                :Vectors => [X, Y, X2],
                :TestVectorArithmetic => false,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14,
                distance => sqrt(2) * log(2),
                injectivity_radius => Inf,
                is_flat => true,
                manifold_dimension => 6,
                representation_size => (3, 3),
                :IsPointErrors => is_point_errors,
                :IsVectorErrors => is_vector_errors,
            ),
        )
        @testset "curvature" begin
            p = [2.0 1.0 0.0; 1.0 2.0 0.0; 0.0 0.0 1.0]
            Z1 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
            Z2 = [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
            @test riemann_tensor(M4, p, Z1, Z2, Z1) == zero(Z1)
            @test sectional_curvature(M4, p, Z1, Z2) === 0.0
            @test sectional_curvature_max(M4) === 0.0
            @test sectional_curvature_min(M4) === 0.0
        end
    end

    @testset "Bures-Wasserstein metric" begin
        M5 = MetricManifold(M, BuresWassersteinMetric())
        Manifolds.Test.test_manifold(
            M5,
            Dict(
                :Functions => [
                    distance, exp, geodesic, injectivity_radius, inner, is_default_metric,
                    is_flat, is_point, is_vector, log, manifold_dimension, mid_point, norm,
                    representation_size, shortest_geodesic, zero_vector,
                ],
                :InvalidPoints => invalid_points,
                :InvalidVectors => invalid_vectors,
                :OtherMetrics => [AffineInvariantMetric()],
                :Points => [p1, p2, p3],
                :SecondVector => X2,
                :Vectors => [2 * (sqrt(2) - 1) / log(2) * X, Y, X2],
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                :atols => Dict(geodesic => 1.0e-10, shortest_geodesic => 1.0e-10),
                # the eigenvalues of p1 and p2 commute: d = ‖√λ(p2) - √λ(p1)‖
                distance => 2 - sqrt(2),
                exp => p2,
                injectivity_radius => Inf,
                is_default_metric => BuresWassersteinMetric(),
                is_flat => false,
                log => 2 * (sqrt(2) - 1) / log(2) * X,
                manifold_dimension => 6,
                mid_point => [((1 + sqrt(2)) / 2)^2 0.0 0.0; 0.0 ((1 + sqrt(2)) / 2)^2 0.0; 0.0 0.0 1.0],
                norm => 2 - sqrt(2),
                representation_size => (3, 3),
                :IsPointErrors => is_point_errors,
                :IsVectorErrors => is_vector_errors,
            ),
        )
        @testset "element type of the inner product" begin
            Xf = Float32[0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]
            @test inner(M5, Float32.(p1), Xf, Xf) isa Float32
        end
        @testset "logarithm of points whose product is not symmetric" begin
            @test isapprox(M5, p2, log(M5, p2, exp(M5, p2, Y)), Y)
        end
        @testset "change of representer" begin
            Z = ones(3, 3)
            @test isapprox(change_representer(M5, EuclideanMetric(), p1, Z), 2 .* (Z * p1 + p1 * Z))
        end
    end

    @testset "Generalized Bures-Wasserstein metric" begin
        B = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
        g = GeneralizedBuresWassersteinMetric(B)
        M6 = MetricManifold(M, g)
        Manifolds.Test.test_manifold(
            M6,
            Dict(
                :Functions => [
                    distance, exp, geodesic, injectivity_radius, inner, is_default_metric,
                    is_flat, is_point, is_vector, log, manifold_dimension, mid_point, norm,
                    representation_size, shortest_geodesic, zero_vector,
                ],
                :InvalidPoints => invalid_points,
                :InvalidVectors => invalid_vectors,
                :OtherMetrics => [AffineInvariantMetric()],
                :Points => [p1, p2, p3],
                :SecondVector => X2,
                :Vectors => [X, Y, X2],
            ),
            # Expectations
            Dict(
                :atol => 1.0e-10,
                # the distance is the square root of a difference of traces, so d(p, p) ≈ 3e-8
                :atols => Dict(distance => 1.0e-7, geodesic => 1.0e-7, shortest_geodesic => 1.0e-7),
                injectivity_radius => Inf,
                is_default_metric => g,
                is_flat => false,
                manifold_dimension => 6,
                representation_size => (3, 3),
                :IsPointErrors => is_point_errors,
                :IsVectorErrors => is_vector_errors,
            ),
        )
        @testset "metric" begin
            @test repr(g) == "GeneralizedBuresWassersteinMetric([2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0])"
            @test g == GeneralizedBuresWassersteinMetric(copy(B))
            @test hash(g) == hash(GeneralizedBuresWassersteinMetric(copy(B)))
            g2 = GeneralizedBuresWassersteinMetric([2.0 1.0 -0.0; 1.0 2.0 1.0; 0.0 1.0 2.0])
            @test !isequal(g, g2)
            Z = ones(3, 3)
            @test isapprox(
                change_representer(M6, EuclideanMetric(), p1, Z),
                2 .* (Z * p1 * B + B * p1' * Z),
            )
        end
        @testset "Bures-Wasserstein distances are real numbers" begin
            M5 = MetricManifold(M, BuresWassersteinMetric())
            B2(α) = [cos(α) sin(α) 0.0; -sin(α) cos(α) 0.0; 0.0 0.0 1.0]
            q1 = A(π / 4) * [1.0e-6 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e6] * transpose(A(π / 4))
            q2 = B2(π / 4) * [1.0e6 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e-6] * transpose(B2(π / 4))
            @test distance(M6, p2, p2) isa Real
            for N in (M5, M6)
                @test distance(N, q1, q1) isa Real
                @test distance(N, q1, q2) isa Real
            end
        end
    end

    @testset "Affine invariant metric: further properties" begin
        @testset "Convert SPD to Cholesky" begin
            (l, w) = Manifolds.spd_to_cholesky(p2, Y)
            (xs, vs) = Manifolds.cholesky_to_spd(l, w)
            @test isapprox(xs, p2)
            @test isapprox(vs, Y)
        end
        @testset "Diagonalizing orthonormal basis" begin
            M2 = MetricManifold(M, AffineInvariantMetric())
            donb = get_basis(M, p1, DiagonalizingOrthonormalBasis(X))
            Xs = get_vectors(M, p1, donb)
            @test isapprox(0.0, first(donb.data.eigenvalues))
            @test [inner(M, p1, V, W) for V in Xs, W in Xs] ≈ Matrix{Float64}(I, 6, 6)
            d2onb = get_basis(M2, p1, DiagonalizingOrthonormalBasis(X))
            @test donb.data.eigenvalues == d2onb.data.eigenvalues
            @test get_vectors(M, p1, donb) == get_vectors(M2, p1, d2onb)
        end
        @testset "Orthonormal basis away from the identity" begin
            Bq = get_basis(M, p2, basis)
            @test [inner(M, p2, V, W) for V in Bq.data, W in Bq.data] ≈ Matrix{Float64}(I, 6, 6)
            @test isapprox(M, p2, get_vector(M, p2, get_coordinates(M, p2, X2, basis), basis), X2)
        end
        @testset "Vector transports are isometric" begin
            Z1 = log(M, p1, p3)
            Z2 = log(M, p1, p2)
            for m in [ParallelTransport(), PoleLadderTransport()]
                W1 = vector_transport_to(M, p1, Z1, p2, m)
                W2 = vector_transport_to(M, p1, Z2, p2, m)
                @test norm(M, p1, Z1) ≈ norm(M, p2, W1)
                @test inner(M, p1, Z1, Z2) ≈ inner(M, p2, W1, W2)
            end
            p4 = A(0.3) * [1.0e6 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e-6] * transpose(A(0.3))
            @test parallel_transport_to(M, p4, Z1, p4) == Z1 # transport to the same point
        end
        @testset "Points that are symmetric up to rounding" begin
            @test is_point(M, exp(M, p3, X2))
            q_near = [2.0e3 1.0e3 0.0; nextfloat(1.0e3) 2.0e3 0.0; 0.0 0.0 4.0e3]
            @test is_point(M, q_near)
            @test is_point(SymmetricPositiveDefinite(3; parameter = :field), q_near)
        end
        @testset "Metric change" begin
            @test change_metric(M, EuclideanMetric(), p1, X) == p1 * X
            @test change_representer(M, EuclideanMetric(), p1, X) == p1 * X * p1
        end
        @testset "Distance and logarithm of points far apart" begin
            pD = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e-30]
            qD = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e-20]
            @test distance(M, pD, qD) ≈ log(1.0e10)
            E = Matrix{Float64}(I, 3, 3)
            @test distance(M, 1.0e-9 * E, 1.0e9 * E) ≈ sqrt(3) * log(1.0e18)
            @test distance(M, 1.0e-20 * E, 2.0e-20 * E) ≈ sqrt(3) * log(2)
            Z = log(M, 1.0e9 * E, 1.0e-9 * E)
            @test norm(M, 1.0e9 * E, Z) ≈ sqrt(3) * log(1.0e18)
            @test exp(M, 1.0e9 * E, Z) ≈ 1.0e-9 * E
        end
        @testset "Projection onto the tangent space" begin
            Z = project(M, p1, [1.0 2.0 1.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
            @test is_vector(M, p1, Z)
            @test Z ≈ [1.0 1.0 0.5; 1.0 1.0 0.0; 0.5 0.0 1.0]
        end
        @testset "rand" begin
            q = rand(M)
            @test is_vector(M, q, rand(M; vector_at = q, tangent_distr = :Rician))
            @test is_vector(
                M, q, rand(MersenneTwister(123), M; vector_at = q, tangent_distr = :Rician)
            )
            # coordinates of a Gaussian tangent vector in an orthonormal basis at the point
            Bd = get_basis(M, p2, DiagonalizingOrthonormalBasis(Matrix{Float64}(I, 3, 3)))
            z = randn(MersenneTwister(42), 6)
            Z = rand(MersenneTwister(42), M; vector_at = p2, σ = 2.0)
            @test get_coordinates(M, p2, Z, Bd) ≈ 2 * z
            Z2 = rand(MersenneTwister(42), M; vector_at = p2)
            @test get_coordinates(M, p2, Z2, Bd) ≈ z / sqrt(3)
            # the Rician draw at an SPDPoint equals the one at its matrix
            qS = SPDPoint(p2)
            ZR = rand(MersenneTwister(42), M; vector_at = p2, tangent_distr = :Rician)
            @test rand(MersenneTwister(42), M; vector_at = qS, tangent_distr = :Rician) ≈ ZR
            # a random point drawn into an SPDPoint stores its matrix square roots
            pR = rand!(MersenneTwister(42), M, SPDPoint(p2))
            @test pR.sqrt ≈ SPDPoint(pR.p).sqrt
            @test pR.sqrt_inv ≈ SPDPoint(pR.p).sqrt_inv
            pN = SPDPoint(p2; store_sqrt = false, store_sqrt_inv = false)
            @test rand!(MersenneTwister(42), M, pN).p == pR.p
        end
        @testset "Riemann tensor" begin
            p = [
                1.3996531703810995 -0.050757455076129054 0.012468338878281887
                -0.050757455076129054 1.4934536968479577 -0.1322157270710227
                0.012468338878281887 -0.1322157270710227 1.1406534894519493
            ]
            Z1 = [
                -0.040154529852424355 0.0940975617218614 -0.14399681300759132
                0.0940975617218614 -0.07959493647453719 0.039521032486382335
                -0.14399681300759132 0.039521032486382335 -0.08077209831531029
            ]
            Z2 = [
                0.22320202697960773 0.1294383854675894 -0.15402580371166136
                0.1294383854675894 0.23251725583184363 0.13919342694607098
                -0.15402580371166136 0.139193426946071 0.4493904218883048
            ]
            Z3 = [
                0.14955817329734716 0.04874731541620811 0.16142208646317371
                0.048747315416208095 0.4135413163173373 0.0813738001379617
                0.16142208646317371 0.0813738001379617 0.5381987409318881
            ]
            Z_rt = [
                0.0026845159609378074 0.0008609241226852796 0.003549384063443887
                0.0008609241226852785 -0.0010229969873821812 -0.0014137071858029112
                0.003549384063443888 -0.0014137071858029105 -0.0011418891263384548
            ]
            @test isapprox(M, p, riemann_tensor(M, p, Z1, Z2, Z3), Z_rt)
        end
        @testset "Curvature" begin
            @test sectional_curvature_min(SymmetricPositiveDefinite(1)) == 0.0
            @test sectional_curvature_min(SymmetricPositiveDefinite(2)) == -0.5
            Z1 = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -1.0] / sqrt(2)
            Z2 = [0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0] / sqrt(2)
            @test sectional_curvature(M, p1, Z1, Z2) ≈ sectional_curvature_min(M)
        end
        @testset "Volume density" begin
            @test manifold_volume(M) == Inf
            p = [
                1.680908185710701 -0.030208936760309613 -0.284402826783584
                -0.030208936760309613 1.7199740873691465 -0.0066638025832747305
                -0.284402826783584 -0.0066638025832747305 1.6842441059901379
            ]
            Z = [
                -1.3447374559982306 0.2591587910302816 0.9739140395169474
                0.2591587910302816 0.3649975914025044 -0.2888865063584093
                0.9739140395169474 -0.2888865063584093 -0.9564259306801289
            ]
            @test volume_density(M, p, Z) ≈ 1.1326952644501451
        end
        @testset "Riemannian Hessian conversion" begin
            N = SymmetricPositiveDefinite(2)
            p = [2.0 1.0; 1.0 1.0]
            G = [1.0 0.0; 1.0 0.1]
            H = [2.0 1.0; 0.0 0.0]
            Z = [0.0 3.0; 3.0 0.0]
            W = Z * 1 / 2 * (G' + G) * p
            W = 1 / 2 * p * (H' + H) * p + 1 / 2 * (W' + W)
            @test riemannian_Hessian(N, p, G, H, Z) == W
        end
        @testset "BigFloat points" begin
            N = SymmetricPositiveDefinite(2)
            p = BigFloat[
                1.6590891025248637458133771360735408961772918701171875 -2.708777790960681386422947980463504791259765625e-7
                -2.708777790960681386422947980463504791259765625e-7 1.6590893171834280028775765458703972399234771728515625
            ]
            @test is_point(N, p)
        end
    end

    @testset "SPDPoint functions" begin
        p = SPDPoint(2 * Matrix{Float64}(I, 3, 3))
        p2 = copy(p)
        @test SPDPoint(p2) === p2
        @test p2.eigen == p.eigen
        @test ManifoldsBase.check_size(M, p; atol = 1.0e-8) === nothing
        @test ManifoldsBase.check_size(M, p, Matrix{Float64}(I, 3, 3); atol = 1.0e-8) === nothing
        pS = SPDPoint(
            2 * Matrix{Float64}(I, 3, 3);
            store_p = false, store_sqrt = false, store_sqrt_inv = false,
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
            store_p = false, store_sqrt = false, store_sqrt_inv = false,
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
        @test isapprox(exp!(M, pS, p, zero_vector(M, p)), p)
        @test ismissing(pS.sqrt)
        @test ismissing(pS.sqrt_inv)
        qR = SPDPoint(Matrix{Float64}(I, 3, 3); store_p = false)
        exp!(M, qR, p, ones(3, 3))
        @test isapprox(M, qR, exp(M, p, ones(3, 3)))
        @test exp(M, p, ones(3, 3)) == qR
        @test allocate_result(M, zero_vector, p) isa Matrix
        c1 = ManifoldsBase.allocate_coordinates(M, p, Float64, 6)
        c2 = ManifoldsBase.allocate_coordinates(M, embed(M, p), Float64, 6)
        @test typeof(c1) == typeof(c2)
        @test length(c1) == length(c2)
    end

    @testset "field parameter" begin
        Mf = SymmetricPositiveDefinite(3; parameter = :field)
        @test typeof(get_embedding(Mf)) === Euclidean{ℝ, Tuple{Int, Int}}
        @test repr(Mf) == "SymmetricPositiveDefinite(3; parameter=:field)"
        @test Manifolds.get_parameter_type(Mf) === :field
    end
end
