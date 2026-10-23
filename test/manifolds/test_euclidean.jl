using Distributions, FiniteDifferences, LinearAlgebra, ManifoldDiff, Manifolds, ManifoldsBase
using Random, RecursiveArrayTools, StaticArrays, Test
using Manifolds: induced_basis

# a test suite run on a flat space; the expected values follow from the points and vectors
function test_euclidean(
        M, pts, X, Y;
        bases = [DefaultOrthonormalBasis()], coordinates = [], mutating = true,
        extra_functions = [], extra_expectations = Dict(), invalid_points = [],
        invalid_vectors = [], errors_points = [], errors_vectors = [],
    )
    p1, p2 = pts[1], pts[2]
    functions = [
        copy, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, exp, flat, geodesic, get_basis,
        get_coordinates, get_vector, get_vectors, injectivity_radius, inner, inverse_retract,
        is_flat, is_point, is_vector, log, manifold_dimension, mid_point, norm,
        parallel_transport_direction, parallel_transport_to, project, rand, repr,
        representation_size, retract, sharp, shortest_geodesic, vector_transport_direction,
        vector_transport_to, zero_vector, extra_functions...,
    ]
    mutating && push!(functions, copyto!)
    properties = Dict(
        :Bases => bases,
        :Coordinates => coordinates,
        :Covectors => [flat(M, p1, X)],
        :EmbeddedPoints => [p1],
        :EmbeddedVectors => [X],
        :Functions => functions,
        :InvalidPoints => invalid_points,
        :InvalidVectors => invalid_vectors,
        :InverseRetractionMethods => [LogarithmicInverseRetraction()],
        :Mutating => mutating,
        :Points => pts,
        :RetractionMethods => [ExponentialRetraction()],
        :SecondVector => Y,
        :Vectors => [X, X, X],
        :VectorTransportMethods => [
            ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport(),
        ],
        :Seed => 42,
    )
    expectations = Dict(
        :atol => 1.0e-12,
        default_inverse_retraction_method => LogarithmicInverseRetraction(),
        default_retraction_method => ExponentialRetraction(),
        default_vector_transport_method => ParallelTransport(),
        distance => norm(p2 - p1),
        exp => p1 + X,
        injectivity_radius => Inf,
        (injectivity_radius, ExponentialRetraction()) => Inf,
        inner => dot(X, Y),
        is_flat => true,
        log => p2 - p1,
        mid_point => (p1 + p2) / 2,
        norm => norm(X),
        parallel_transport_direction => X,
        parallel_transport_to => X,
        (project, :Point) => p1,
        (project, :Vector) => X,
        representation_size => size(p1),
        sharp => X,
        (vector_transport_to, ParallelTransport()) => X,
        :IsPointErrors => errors_points,
        :IsVectorErrors => errors_vectors,
    )
    for B in bases
        expectations[(get_vectors, B)] = :Orthonormal
    end
    return Manifolds.Test.test_manifold(M, properties, merge(expectations, extra_expectations))
end

@testset "Euclidean" begin
    @test_throws DomainError Euclidean(2, -1)
    @testset "parameter = $param" for param in [:field, :type]
        E = Euclidean(3, parameter = param)
        Ec = Euclidean(3; field = ℂ, parameter = param)
        EM = Manifolds.MetricManifold(E, Manifolds.EuclideanMetric())
        EH = Euclidean(2, 3; field = ℍ, parameter = param)
        if param === :type
            @test repr(E) == "Euclidean(3; field=ℝ)"
            @test repr(Ec) == "Euclidean(3; field=ℂ)"
            @test repr(EH) == "Euclidean(2, 3; field=ℍ)"
        else
            @test repr(E) == "Euclidean(3; field=ℝ, parameter=:field)"
            @test repr(Ec) == "Euclidean(3; field=ℂ, parameter=:field)"
            @test repr(EH) == "Euclidean(2, 3; field=ℍ, parameter=:field)"
        end

        @test Manifolds.allocation_promotion_function(Ec, get_vector, ()) === complex
        @test is_flat(E)
        @test is_flat(Ec)
        p = zeros(3)
        A = Manifolds.RetractionAtlas()
        B = induced_basis(EM, A, p, TangentSpaceType())
        @test det_local_metric(EM, p, B) == one(eltype(p))
        for M in [E, Ec, EH]
            @test has_components(M)
            p = [1.0, 2.0, 3.0]
            q = [1.0, 2.0, 3.0]
            X = [4.0, 5.0, 6.0]
            for r in [1, 2, Inf]
                @test norm(M, p, X, r) ≈ norm(X, r)
                @test distance(M, p, q, r) ≈ norm(p - q, r)
            end
        end
        @test log_local_metric_density(EM, p, B) == zero(eltype(p))
        @test project!(E, p, p) == p
        @test embed!(E, p, p) == p
        @test manifold_dimension(Ec) == 2 * manifold_dimension(E)
        X = zeros(3)
        X[1] = 1.0
        Y = similar(X)
        project!(E, Y, p, X)
        @test Y == X
        @test embed(E, p, X) == X

        # temp: explicit test for induced basis
        B = induced_basis(E, RetractionAtlas(), 0, ManifoldsBase.TangentSpaceType())
        @test get_coordinates(E, p, X, B) == X
        get_coordinates!(E, Y, p, X, B)
        @test Y == X
        @test get_vector(E, p, Y, B) == X
        Y2 = similar(X)
        get_vector!(E, Y2, p, Y, B)
        @test Y2 == X

        # real manifold does not allow complex values
        @test_throws DomainError is_point(Ec, [:a, :b, :b]; error = :error)
        @test_throws DomainError is_point(E, [1.0, 1.0im, 0.0], error = :error)
        @test_throws DomainError is_point(E, [1]; error = :error)
        @test_throws DomainError is_vector(Ec, [:a, :b, :b], [1.0, 1.0, 0.0]; error = :error)
        @test_throws DomainError is_vector(
            E, [1.0, 1.0im, 0.0], [1.0, 1.0, 0.0];
            error = :error,
        ) # real manifold does not allow complex values
        @test_throws DomainError is_vector(E, [1], [1.0, 1.0, 0.0]; error = :error)
        @test_throws DomainError is_vector(E, [0.0, 0.0, 0.0], [1.0]; error = :error)
        @test_throws DomainError is_vector(
            E, [0.0, 0.0, 0.0], [1.0, 0.0, 1.0im];
            error = :error,
        )
        @test_throws DomainError is_vector(Ec, [0.0, 0.0, 0.0], [:a, :b, :c]; error = :error)

        @test E^2 === Euclidean(3, 2, parameter = param)
        @test ^(E, 2) === Euclidean(3, 2, parameter = param)
        @test E^(2,) === Euclidean(3, 2, parameter = param)
        @test Ec^(4, 5) === Euclidean(3, 4, 5; field = ℂ, parameter = param)

        pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        X = [-1.0, 1.0, 0.5]
        Y = [0.0, 2.0, 1.0]
        D = DiagonalizingOrthonormalBasis([1.0, 2.0, 3.0])
        invalid_points = [[1.0, 1.0im, 0.0], [1.0]]
        invalid_vectors = [[1.0], [1.0, 0.0, 1.0im]]
        flat_space_expectations = Dict(
            manifold_volume => Inf,
            sectional_curvature => 0.0,
            sectional_curvature_max => 0.0,
            sectional_curvature_min => 0.0,
            volume_density => 1.0,
        )
        @testset "real Euclidean space $M" for M in [E, EM]
            test_euclidean(
                M, pts, X, Y;
                bases = [DefaultOrthonormalBasis(), D],
                coordinates = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                extra_functions = M === E ?
                    [manifold_volume, sectional_curvature, volume_density] : [],
                extra_expectations = merge(
                    Dict(
                        (get_coordinates, DefaultOrthonormalBasis()) => X,
                        (get_vector, [1.0, 2.0, 3.0], DefaultOrthonormalBasis()) => [1.0, 2.0, 3.0],
                        manifold_dimension => 3,
                        repr => repr(M),
                    ),
                    M === E ? flat_space_expectations : Dict(),
                ),
                invalid_points = invalid_points,
                invalid_vectors = invalid_vectors,
                errors_points = [DomainError, DomainError],
                errors_vectors = [DomainError, DomainError],
            )
        end
        bases_c = [
            DefaultOrthonormalBasis(), DefaultOrthonormalBasis(ℂ),
            D, DiagonalizingOrthonormalBasis([1.0, 2.0, 3.0], ℂ),
        ]
        coordinates_c = [
            collect(1.0:6.0), [1.0, 2.0im, 3.0], collect(1.0:6.0), [1.0, 2.0im, 3.0],
        ]
        complex_expectations = merge(
            flat_space_expectations,
            Dict(
                (get_coordinates, DefaultOrthonormalBasis(ℂ)) => X,
                manifold_dimension => 6,
                repr => repr(Ec),
            ),
        )
        @testset "complex Euclidean space with real points" begin
            test_euclidean(
                Ec, pts, X, Y;
                bases = bases_c, coordinates = coordinates_c,
                extra_functions = [manifold_volume, sectional_curvature, volume_density],
                extra_expectations = complex_expectations,
                invalid_points = [[:a, :b, :b], [1.0]],
                invalid_vectors = [[:a, :b, :c], [1.0]],
                errors_points = [DomainError, DomainError],
                errors_vectors = [DomainError, DomainError],
            )
        end
        @testset "complex Euclidean space with complex points" begin
            pts_c = [[1.0im, -1.0im, 1.0], [0.0, 1.0, 1.0im], [0.0, 0.0, 1.0]]
            test_euclidean(
                Ec, pts_c, [1.0im, 0.5, -1.0], [2.0, 1.0im, 0.0];
                bases = bases_c, coordinates = coordinates_c,
                extra_functions = [manifold_volume, sectional_curvature, volume_density],
                extra_expectations = merge(
                    delete!(copy(complex_expectations), (get_coordinates, DefaultOrthonormalBasis(ℂ))),
                    Dict((number_eltype, rand) => ComplexF64),
                ),
                invalid_points = [[:a, :b, :b], [1.0]],
                invalid_vectors = [[:a, :b, :c], [1.0]],
                errors_points = [DomainError, DomainError],
                errors_vectors = [DomainError, DomainError],
            )
        end

        @testset "ProjectedOrthonormalBasis" begin
            p = pts[1]
            b = get_basis(E, p, ProjectedOrthonormalBasis(:svd))
            V = get_vectors(E, p, b)
            @test length(V) == 3
            @test [inner(E, p, v, w) for v in V, w in V] ≈ I(3)
            @test get_vector(E, p, get_coordinates(E, p, X, b), b) ≈ X
        end

        @testset "distributions" begin
            for M in [E, EM, Ec]
                pd = Manifolds.projected_distribution(M, Distributions.MvNormal(zeros(3), 1.0 * I))
                q = rand(pd)
                @test is_point(M, q)
                rand!(pd, q)
                @test is_point(M, q)
                tvd = Manifolds.normal_tvector_distribution(M, pts[1], 1.0 * I)
                Z = rand(tvd)
                @test is_vector(M, pts[1], Z)
                rand!(tvd, Z)
                @test is_vector(M, pts[1], Z)
            end
        end
    end
    E = Euclidean(3)
    Ec = Euclidean(3; field = ℂ)

    @testset "(Nonmutating) Real and Complex Numbers, type $T" for T in [Float64, ComplexF64]
        M = (T <: Complex) ? Euclidean(; field = ℂ) : Euclidean()
        pts = convert.(Ref(T), [1.0, 4.0, 2.0])
        @test embed(M, pts[1]) == pts[1]
        @test project(M, pts[1]) == pts[1]
        @test retract(M, pts[1], pts[2]) == exp(M, pts[1], pts[2])
        test_euclidean(
            M, pts, convert(T, 2.0), convert(T, -0.5);
            # coordinates of numbers are tested separately below
            bases = [],
            mutating = false,
            invalid_points = [[1.0, 2.0]],
            invalid_vectors = [[1.0, 2.0]],
            errors_points = [DomainError],
            errors_vectors = [DomainError],
            extra_functions = [manifold_volume, sectional_curvature, volume_density],
            extra_expectations = merge(
                Dict(
                    manifold_volume => Inf, volume_density => 1.0,
                    sectional_curvature => 0.0, sectional_curvature_max => 0.0,
                    sectional_curvature_min => 0.0,
                ),
                Dict(
                    manifold_dimension => (T <: Complex ? 2 : 1),
                    (number_eltype, rand) => T,
                    repr => repr(M),
                ),
            ),
        )
        @test rand(MersenneTwister(4711), M; vector_at = pts[1]) isa T
    end

    @testset "Coordinates of the number valued Euclidean space" begin
        B = DefaultOrthonormalBasis()
        Bc = DefaultOrthonormalBasis(ℂ)
        for param in [:type, :field]
            M = Euclidean(; parameter = param)
            @test get_coordinates(M, 0.3, 1.2, B) == 1.2
            Mc = Euclidean(; field = ℂ, parameter = param)
            @test get_coordinates(Mc, 0.3im, 1.2 + 2.0im, B) == [1.2, 2.0]
            @test get_coordinates(Mc, 0.3im, 1.2 + 2.0im, Bc) == 1.2 + 2.0im
        end
    end

    @testset "hat/vee" begin
        E = Euclidean(3, 2)
        p = collect(reshape(1.0:6.0, (3, 2)))
        X = collect(reshape(7.0:12.0, (3, 2)))
        @test hat(E, p, vec(X)) ≈ X
        Y = allocate(X)
        @test hat!(E, Y, p, vec(X)) === Y
        @test Y ≈ X
        @test vee(E, p, X) ≈ vec(X)
        Y = allocate(vec(X))
        @test vee!(E, Y, p, X) === Y
        @test Y ≈ vec(X)
    end

    @testset "Number systems power" begin
        @test ℝ^2 === Euclidean(2)
        @test ℝ^(2, 3) === Euclidean(2, 3)

        @test ℂ^2 === Euclidean(2; field = ℂ)
        @test ℂ^(2, 3) === Euclidean(2, 3; field = ℂ)

        @test ℍ^2 === Euclidean(2; field = ℍ)
        @test ℍ^(2, 3) === Euclidean(2, 3; field = ℍ)
    end

    @testset "Embeddings into larger Euclidean Manifolds" begin
        M = Euclidean(3, 3)
        N = Euclidean(4, 4)
        O = EmbeddedManifold(M, N)
        # first test with same length of sizes
        p = ones(3, 3)
        q = zeros(4, 4)
        qT = zeros(4, 4)
        qT[1:3, 1:3] .= 1.0
        embed!(O, q, p)
        @test norm(qT - q) == 0
        qM = embed(O, p)
        @test norm(project(O, qM) - p) == 0
        @test norm(qT - qM) == 0
        # test with different sizes, check that it only fills first element
        q2 = zeros(4, 4, 3)
        q2T = zeros(4, 4, 3)
        q2T[1:3, 1:3, 1] .= 1.0
        embed!(O, q2, p)
        @test norm(q2T - q2) == 0
        O2 = EmbeddedManifold(M, Euclidean(4, 4, 3))
        q2M = embed(O2, p)
        @test norm(q2T - q2M) == 0
        @test norm(project(O2, q2M) - p) == 0
        # wrong size error checks
        @test_throws DomainError embed!(O, zeros(3, 3), zeros(3, 3, 5))
        @test_throws DomainError embed!(O, zeros(3, 3), zeros(4, 4))
        @test_throws DomainError project!(O, zeros(3, 3, 5), zeros(3, 3))
        @test_throws DomainError project!(O, zeros(4, 4), zeros(3, 3))
        @test_throws "shorter than the Euclidean" project!(O, zeros(3, 3, 5), zeros(3, 3))
        @test_throws "entry smaller than the Euclidean" project!(O, zeros(4, 4), zeros(3, 3))
    end

    @testset "Embedding Real into Complex" begin
        M = Euclidean(3, 3)
        N = Euclidean(3, 4; field = ℂ)
        O = EmbeddedManifold(M, N)
        p = ones(3, 3)
        qT = zeros(ComplexF64, 3, 4)
        qT[1:3, 1:3] .= 1.0
        q = embed(O, p)
        @test is_point(N, q)
        @test q == qT
        q2 = zeros(ComplexF64, 3, 4)
        embed!(O, q2, p)
        @test q2 == qT
    end

    @testset "Euclidean metric tests" begin
        M = Euclidean(2)
        p = zeros(2)
        A = Manifolds.get_default_atlas(M)
        i = Manifolds.get_chart_index(M, A, p)
        B = Manifolds.induced_basis(M, A, i, TangentSpaceType())
        C1 = christoffel_symbols_first(M, p, B)
        @test size(C1) == (2, 2, 2)
        @test norm(C1) ≈ 0.0 atol = 1.0e-13
        C2 = christoffel_symbols_second(M, p, B)
        @test size(C2) == (2, 2, 2)
        @test norm(C2) ≈ 0.0 atol = 1.0e-13
        C2j = christoffel_symbols_second_jacobian(M, p, B)
        @test size(C2j) == (2, 2, 2, 2)
        @test norm(C2j) ≈ 0.0 atol = 1.0e-16
        @test norm(einstein_tensor(M, p, B)) < eps()^2
        @test ricci_curvature(M, p, B) ≈ 0 atol = 1.0e-16
        RC = ricci_tensor(M, p, B)
        @test size(RC) == (2, 2)
        @test norm(RC) ≈ 0.0 atol = 1.0e-16
        @test local_metric(M, p, B) == Diagonal(ones(2))
        @test inverse_local_metric(M, p, B) == Diagonal(ones(2))
        @test det_local_metric(M, p, B) == 1
        RT = riemann_tensor(M, p, B)
        @test size(RT) == (2, 2, 2, 2)
        @test norm(RT) ≈ 0.0 atol = 1.0e-16

        @test !Manifolds.check_chart_switch(M, A, i, p)

        @test riemann_tensor(M, p, [1, 2], [1, 3], [1, 4]) == [0, 0]
    end
    @testset "Induced Basis and local metric for EuclideanMetric" begin
        struct DefaultManifold <: AbstractManifold{ℝ} end
        p = zeros(3)
        M = DefaultManifold()
        TpM = TangentSpace(M, p)
        @test base_manifold(TpM) === M
        B = induced_basis(M, Manifolds.get_default_atlas(M), p, TangentSpaceType())
        MM = MetricManifold(M, EuclideanMetric())
        @test local_metric(MM, p, B) == Diagonal(ones(3))
        @test inverse_local_metric(MM, p, B) == Diagonal(ones(3))
        @test det_local_metric(MM, p, B) == 1.0
        DB1 = dual_basis(MM, p, B)
        @test DB1 isa InducedBasis
        @test DB1.vs isa ManifoldsBase.CotangentSpaceType
        DB2 = induced_basis(M, Manifolds.get_default_atlas(M), p, CotangentSpaceType())
        @test DB2 isa InducedBasis
        @test DB2.vs isa ManifoldsBase.CotangentSpaceType
        DDB = dual_basis(MM, p, DB2)
        @test DDB isa InducedBasis
        @test DDB.vs isa ManifoldsBase.TangentSpaceType
    end
    @testset "RNG point with σ" begin
        Random.seed!(42)
        @test is_point(E, rand(E; σ = 10.0))
        @test is_point(E, rand(MersenneTwister(123), E; σ = 10.0))
        pc = rand(Ec)
        @test is_point(Ec, pc)
        @test norm(imag.(pc)) != 0
    end

    @testset "StaticArrays specializations" begin
        M1 = Euclidean(3)
        @test get_vector(
            M1, SA[1.0, 2.0, 3.0], SA[-1.0, -2.0, -3.0], DefaultOrthonormalBasis()
        ) === SA[-1.0, -2.0, -3.0]

        c_sv = SizedVector{3}([-1.0, -2.0, -3.0])
        @test get_vector(M1, SA[1.0, 2.0, 3.0], c_sv, DefaultOrthonormalBasis()) === c_sv

        M2 = Euclidean(2, 2)
        @test get_vector(
            M2, SA[1.0 2.0; 3.0 4.0], SA[-1.0, -2.0, -3.0, -4.0], DefaultOrthonormalBasis()
        ) === SA[-1.0 -3.0; -2.0 -4.0]

        @test get_vector(
            M2,
            SizedMatrix{2, 2}([1.0 2.0; 3.0 4.0]),
            SizedMatrix{2, 2}([-1.0, -2.0, -3.0, -4.0]),
            DefaultOrthonormalBasis(),
        ) == SA[-1.0 -3.0; -2.0 -4.0]

        M1c = Euclidean(3, field = ℂ)
        @test get_vector(
            M1c,
            SizedVector{3}([1.0im, 2.0, 4.0im]),
            SizedVector{3}([-1.0, -3.0, -4.0im]),
            DefaultOrthonormalBasis(ℂ),
        ) == SA[-1.0, -3.0, -4.0im]
    end

    @testset "Euclidean(1)" begin
        M = Euclidean(1)
        @test distance(M, 2.0, 4.0) == 2.0
    end

    @testset "the distance on a complex space is a real number" begin
        M = Euclidean(2; field = ℂ)
        d = distance(M, [1.0 + 0im, 2.0 + 0im], [0.0im, 0.0im])
        @test d isa Float64
        @test d ≈ sqrt(5)
    end

    @testset "errors" begin
        M = Euclidean(4)
        @test_throws DimensionMismatch distance(M, [1, 2, 3, 4], [1 2; 3 4])
    end

    @testset "ManifoldDiff" begin
        # ManifoldDiff
        M0 = Euclidean()
        @test ManifoldDiff.adjoint_Jacobi_field(
            M0, 0.0, 1.0, 0.5, 2.0, ManifoldDiff.βdifferential_shortest_geodesic_startpoint
        ) === 1.0
        @test ManifoldDiff.diagonalizing_projectors(M0, 0.0, 2.0) ==
            ((0.0, ManifoldDiff.IdentityProjector()),)
        @test ManifoldDiff.jacobi_field(
            M0, 0.0, 1.0, 0.5, 2.0, ManifoldDiff.βdifferential_shortest_geodesic_startpoint
        ) === 1.0
    end

    @testset "Weingarten & Hessian" begin
        M = Euclidean(2)
        p = [1.0, 2.0]
        G = [3.0, 4.0]
        H = [5.0, 6.0]
        X = [7.0, 8.0]
        rH = riemannian_Hessian(M, p, G, H, X)
        @test rH == H
    end
    @testset "field parameter" begin
        Ms = Euclidean(1; parameter = :field)
        M0s = Euclidean(; parameter = :field)

        @test distance(Ms, 2.0, 4.0) == 2.0
        @test distance(M0s, 2.0, 4.0) == 2.0
        @test log(M0s, 2.0, 4.0) == 2.0
        @test manifold_dimension(M0s) == 1
        @test project(M0s, 4.0) == 4.0
        @test project(M0s, 2.0, 4.0) == 4.0
        @test zero_vector(M0s, 2.0) == 0.0
        @test retract(M0s, 2.0, 4.0) == 6.0
        @test retract(M0s, 2.0, 4.0, ExponentialRetraction()) == 6.0

        @test ManifoldDiff.adjoint_Jacobi_field(
            M0s, 0.0, 1.0, 0.5, 2.0, ManifoldDiff.βdifferential_shortest_geodesic_startpoint
        ) === 1.0
        @test ManifoldDiff.diagonalizing_projectors(M0s, 0.0, 2.0) ==
            ((0.0, ManifoldDiff.IdentityProjector()),)
        @test ManifoldDiff.jacobi_field(
            M0s, 0.0, 1.0, 0.5, 2.0, ManifoldDiff.βdifferential_shortest_geodesic_startpoint
        ) === 1.0
    end

    @testset "Mixed array dimensions for exp and PT" begin
        # this is an issue on Julia 1.6 but not later releases
        for M in [Euclidean(), Euclidean(; parameter = :field)]
            p = fill(0.0)
            Manifolds.exp_fused!(M, p, p, [1.0], 2.0)
            @test p ≈ fill(2.0)
            parallel_transport_to!(M, p, p, [4.0], p)
            @test p ≈ fill(4.0)
        end
    end

    @testset "get_vector on the number valued Euclidean space" begin
        B = DefaultOrthonormalBasis()
        Bc = DefaultOrthonormalBasis(ℂ)
        for param in [:type, :field]
            @test get_vector(Euclidean(; parameter = param), 0.3, [1.2], B) === 1.2
            Mc = Euclidean(; field = ℂ, parameter = param)
            @test get_vector(Mc, 0.3im, [1.2, 2.0], B) === 1.2 + 2.0im
            @test get_vector(Mc, 0.3im, [1.2 + 2.0im], Bc) === 1.2 + 2.0im
        end
    end
end
