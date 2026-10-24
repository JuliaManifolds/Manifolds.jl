using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "The probability simplex" begin
    @test_throws DomainError ProbabilitySimplex(-1)
    @test_throws ArgumentError ProbabilitySimplex(2; boundary = :tomato)

    M = ProbabilitySimplex(2)
    p = [0.5, 0.3, 0.2]
    q = [0.4, 0.4, 0.2]
    r = [0.3, 0.5, 0.2]
    # the simplex is isometric to the positive orthant of the sphere of radius 2
    # via p ↦ 2√p, so for example the distance is 2 arccos(∑ᵢ √(pᵢqᵢ))
    s = sum(sqrt.(p .* q))
    d = 2 * acos(s)
    X = d / sqrt(1 - s^2) .* (sqrt.(p .* q) .- s .* p) # log(M, p, q)
    Y = [0.1, -0.2, 0.1]
    # Invalids
    pf1 = p .+ 1 # does not sum to 1
    pf2 = -ones(3) # negative entries
    pf3 = [0.0] # wrong size
    Xf1 = Y .+ 1 # does not sum to 0
    Xf2 = zeros(4) # wrong size

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Bases => [DefaultOrthonormalBasis()],
            :Coordinates => [[1.0, 0.0]],
            :Covectors => [flat(M, p, X)],
            :EmbeddedPoints => [2 .* p],
            :EmbeddedVectors => [X .+ 1],
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, distance, embed, embed_project, exp, flat,
                geodesic, get_basis, get_coordinates, get_embedding, get_vector, get_vectors,
                injectivity_radius, inner, inverse_retract, is_default_metric, is_flat,
                is_point, is_vector, log, manifold_dimension, manifold_volume, mid_point, norm,
                parallel_transport_direction, parallel_transport_to, project, rand, repr,
                representation_size, retract, sectional_curvature, sharp, shortest_geodesic,
                vector_transport_direction, vector_transport_to, volume_density, zero_vector,
            ],
            :InvalidPoints => AbstractArray[pf1, pf2, pf3],
            :InvalidVectors => AbstractArray[Xf1, Xf2],
            :InverseRetractionMethods => [LogarithmicInverseRetraction(), SoftmaxInverseRetraction()],
            :OtherMetrics => [EuclideanMetric()],
            :Points => [p, q, r],
            :RetractionMethods => [ExponentialRetraction(), SoftmaxRetraction()],
            :SecondVector => Y,
            :Vectors => [X, log(M, q, r), log(M, r, p)],
            :VectorTransportMethods => [ParallelTransport()],
            :Seed => 42,
            :TestRandScaling => true,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
            default_inverse_retraction_method => LogarithmicInverseRetraction(),
            default_retraction_method => ExponentialRetraction(),
            default_vector_transport_method => ParallelTransport(),
            distance => d,
            (embed, :Point) => p,
            (embed, :Vector) => X,
            exp => q,
            get_embedding => Euclidean(3),
            (get_vectors, DefaultOrthonormalBasis()) => :Orthonormal,
            injectivity_radius => 0,
            # the distance to the closest face
            (injectivity_radius, p) => 2 * acos(sqrt(0.8)),
            (injectivity_radius, SoftmaxRetraction()) => 0,
            (injectivity_radius, p, SoftmaxRetraction()) => 2 * acos(sqrt(0.8)),
            # the Fisher–Rao metric ∑ XY/p
            inner => sum(X .* Y ./ p),
            is_default_metric => Manifolds.FisherRaoMetric(),
            is_flat => false,
            log => X,
            manifold_dimension => 2,
            # a quarter of the area of the sphere of radius 2
            manifold_volume => 2 * π,
            mid_point => normalize(sqrt.(p) .+ sqrt.(q)) .^ 2,
            norm => d,
            (project, :Point) => p,
            (project, :Vector) => X,
            (retract, ExponentialRetraction()) => q,
            repr => "ProbabilitySimplex(2; boundary=:open)",
            representation_size => (3,),
            # the curvature of the sphere of radius 2
            sectional_curvature => 1 / 4,
            sectional_curvature_max => 1 / 4,
            sectional_curvature_min => 1 / 4,
            sharp => X,
            volume_density => sin(d / 2) / (d / 2),
            :IsPointErrors => [DomainError, DomainError, ManifoldDomainError],
            :IsVectorErrors => [DomainError, ManifoldDomainError],
            :IsVectorBasepointError => ManifoldDomainError,
        ),
    )

    @testset "Basics" begin
        # tangent vectors of any length are accepted, a clearly nonzero sum is not
        @test is_vector(M, p, project(M, p, [100.0, -50.0, 30.0]))
        @test !is_vector(M, p, [1.0, -1.0, 1.0e-6])
        @test distance(ProbabilitySimplex(3), [0.2, 0.4, 0.3, 0.1], [0.2, 0.4, 0.3, 0.1]) == 0.0
        @test M^4 == MultinomialMatrices(3, 4)
        @test mean(M, [p, q]) == shortest_geodesic(M, p, q)(0.5)
    end

    @testset "Euclidean metric" begin
        M_euc = MetricManifold(M, EuclideanMetric())
        Manifolds.Test.test_manifold(
            M_euc,
            Dict(
                :EmbeddedPoints => [2 .* p],
                :EmbeddedVectors => [(q - p) .+ 1],
                :Functions => [
                    distance, exp, geodesic, get_embedding, inner, log,
                    manifold_dimension, manifold_volume, mid_point, norm, project, rand,
                    representation_size, shortest_geodesic, volume_density, zero_vector,
                ],
                :Points => [p, q, r],
                :SecondVector => Y,
                :Vectors => [q - p, r - q, p - r],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14,
                distance => norm(q - p),
                exp => q,
                get_embedding => Euclidean(3),
                inner => dot(q - p, Y),
                log => q - p,
                manifold_dimension => 2,
                # the area of the triangle with vertices at the unit vectors
                manifold_volume => sqrt(3) / 2,
                mid_point => (p + q) / 2,
                norm => norm(q - p),
                (project, :Point) => p,
                (project, :Vector) => q - p,
                representation_size => (3,),
                volume_density => 1.0,
            ),
        )
        @test ManifoldsBase.exp_fused(M_euc, p, q - p, 0.5) ≈ (p + q) / 2

        @testset "random points and vectors with a given random number generator" begin
            pr = rand(MersenneTwister(42), M_euc)
            @test is_point(M_euc, pr)
            @test pr == rand!(MersenneTwister(42), M_euc, zeros(3))
            Xr = rand(MersenneTwister(44), M_euc; vector_at = p, σ = 0.5)
            @test is_vector(M_euc, p, Xr)
            @test Xr == rand!(MersenneTwister(44), M_euc, zeros(3); vector_at = p, σ = 0.5)
        end
    end

    @testset "Projection and metric conversion" begin
        p = [1 / 2, 1 / 3, 1 / 6]
        q = [0.2, 0.3, 0.5]
        X = log(M, p, q)
        @test isapprox(M, p, X, project(M, p, X .+ 10))

        # Check adaption of metric and representer
        Y1 = change_metric(M, EuclideanMetric(), p, X)
        @test Y1 ≈ [-0.17062114054478128, 0.04002429219016789, 0.13059684835461377]
        # p3 is a point close to the boundary of the simplex, so the metric is nearly singular
        p3 = [0.05218307154737151, 0.9471667513325462, 0.0006501771200825463]
        Y3 = change_metric(M, EuclideanMetric(), p3, [1.0, -0.5, -0.5])
        @test is_vector(M, p3, Y3)
        @test inner(M, p3, Y3, Y3) ≈ 1.5
        Y2 = change_representer(M, EuclideanMetric(), p, X)
        @test Y2 ≈ [-0.10040964054128285, 0.03818665287320871, 0.06222298766807415]

        X = log(M, q, p)
        @test is_vector(M, q, project(M, q, X + [1, 2, 3]); atol = 1.0e-15)

        @test_throws DomainError project(M, [1, -1, 2])
    end

    @testset "Riemannian gradient" begin
        M = ProbabilitySimplex(4) # n=5
        # For f(p) = \sum_i 1/n log(p_i) we know the Euclidean gradient
        grad_f_eucl(p) = [1 / (5 * pi) for pi in p]
        # but we can also easily derive the Riemannian one
        grad_f(M, p) = 1 / 5 .- p
        p = [0.1, 0.2, 0.4, 0.2, 0.1]
        Y = grad_f_eucl(p)
        X = grad_f(M, p)
        @test isapprox(M, p, X, riemannian_gradient(M, p, Y))
        Z = zero_vector(M, p)
        riemannian_gradient!(M, Z, p, Y)
        @test X == Z
    end

    @testset "Simplex with boundary" begin
        Mb = ProbabilitySimplex(2; boundary = :closed)
        pb = [0, 0.5, 0.5]
        Xb = [0, 1, -1]
        Yb = [0, 2, -2]
        @test is_point(Mb, pb)
        @test_throws DomainError is_point(Mb, pb .- 1; error = :error)
        @test inner(Mb, pb, Xb, Yb) == 8
        @test repr(Mb) == "ProbabilitySimplex(2; boundary=:closed)"
        Mb3 = ProbabilitySimplex(3; boundary = :closed)
        pb = [0.0, 0.2, 0.3, 0.5]
        qb = [0.0, 0.5, 0.25, 0.25]
        # a vanishing entry stays zero, so exp inverts log on that face
        @test isapprox(Mb3, exp(Mb3, pb, log(Mb3, pb, qb)), qb)
        # the same holds for the softmax retraction
        qb = retract(Mb3, pb, [0.0, 0.1, 0.05, -0.15], SoftmaxRetraction())
        @test is_point(Mb3, qb)
        @test qb[1] == 0
        # an exponent of 1000 does not overflow
        pc = [1.0e-6, 0.5, 0.499999]
        @test retract(Mb, pc, [0.001, -0.0005, -0.0005], SoftmaxRetraction()) == [1.0, 0.0, 0.0]
    end

    @testset "Probability amplitudes" begin
        p = [0.1, 0.7, 0.2]
        Y = [-0.1, 0.05, 0.05]
        @test Manifolds.simplex_to_amplitude(M, p) ≈
            [0.31622776601683794, 0.8366600265340756, 0.4472135954999579]
        @test Manifolds.amplitude_to_simplex(
            M, [0.31622776601683794, 0.8366600265340756, 0.4472135954999579]
        ) ≈ p
        @test Manifolds.simplex_to_amplitude_diff(M, p, Y) ≈
            [-0.31622776601683794, 0.05976143046671968, 0.1118033988749895]
        @test Manifolds.amplitude_to_simplex_diff(M, p, Y) ≈ [-0.01, 0.035, 0.01]
    end

    @testset "Riemann tensor" begin
        p = [0.1, 0.7, 0.2]
        X = [-0.1, 0.05, 0.05]
        Y = [0.05, 0.05, -0.1]
        Z = [-0.1, 0.15, -0.05]
        @test riemann_tensor(M, p, X, Y, Z) ≈
            [-0.0008705357142857144, -0.0014062500000000002, 0.0022767857142857143]
        @test sectional_curvature_max(ProbabilitySimplex(1)) == 0.0
        @test sectional_curvature_min(ProbabilitySimplex(1)) == 0.0
    end

    @testset "Softmax retraction agrees with exp to first order" begin
        p = [0.1, 0.7, 0.2]
        X = [0.05, 0.05, -0.1]
        dist(t) = distance(M, exp(M, p, t * X), retract(M, p, t * X, SoftmaxRetraction()))
        @test dist(0.2) / dist(0.1) ≈ 4 atol = 0.05
    end

    Manifolds.Test.test_manifold(
        ProbabilitySimplex(2; parameter = :field),
        Dict(:Functions => [get_embedding, is_point, manifold_dimension, repr], :Points => [p]),
        # Expectations
        Dict(
            get_embedding => Euclidean(3; parameter = :field),
            manifold_dimension => 2,
            repr => "ProbabilitySimplex(2; boundary=:open, parameter=:field)",
        ),
    )
end
