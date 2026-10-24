using LinearAlgebra, Manifolds, ManifoldsBase, Random, StaticArrays, Test
using Distributions, ManifoldDiff
using RecursiveArrayTools: ArrayPartition

@testset "Product manifold" begin
    @test_throws MethodError ProductManifold()
    M1 = Sphere(2)
    M2 = Euclidean(2)
    @test (@inferred ProductManifold(M1, M2)) isa ProductManifold
    M = ProductManifold(M1, M2)
    # the sphere component moves along a quarter of a great circle
    p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 0.0])
    q = ArrayPartition([0.0, 1.0, 0.0], [1.0, 0.0])
    r = ArrayPartition([0.0, 0.0, 1.0], [0.0, 0.1])
    X = ArrayPartition([0.0, π / 2, 0.0], [1.0, 0.0]) # log(M, p, q)
    Y = ArrayPartition([0.0, 1.0, 1.0], [1.0, 2.0])
    # multiples of p on the sphere component are normal vectors
    V = ArrayPartition([2.0, 0.0, 0.0], [0.0, 0.0])
    B = DefaultOrthonormalBasis()
    # the sphere has no coordinates in a diagonalizing basis, only a precomputed one
    Bd = get_basis(
        M, p, DiagonalizingOrthonormalBasis(ArrayPartition([0.0, 0.0, 1.0], [1.0, 0.0]))
    )
    retraction_methods = [
        ProductRetraction(ExponentialRetraction(), ExponentialRetraction()),
        ExponentialRetraction(),
    ]
    inverse_retraction_methods = [
        InverseProductRetraction(LogarithmicInverseRetraction(), LogarithmicInverseRetraction()),
        LogarithmicInverseRetraction(),
    ]
    vector_transport_methods = [
        ProductVectorTransport(ParallelTransport(), ParallelTransport()),
        ParallelTransport(),
        ProductVectorTransport(SchildsLadderTransport(), SchildsLadderTransport()),
        ProductVectorTransport(PoleLadderTransport(), PoleLadderTransport()),
    ]
    # the velocity of the geodesic from p to q, i.e. X, is transported to its velocity at q
    X_at_q = ArrayPartition([-π / 2, 0.0, 0.0], [1.0, 0.0])

    functions = [
        copy, copyto!, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, exp, flat, geodesic, get_basis,
        get_coordinates, get_vector, get_vectors, injectivity_radius, inner,
        inverse_retract, is_default_metric, is_flat, is_point, is_vector, log,
        manifold_dimension, manifold_volume, mid_point, norm, parallel_transport_direction,
        parallel_transport_to, project, rand, repr, retract, sectional_curvature, sharp,
        shortest_geodesic, vector_transport_direction, vector_transport_to, volume_density,
        Weingarten, zero_vector,
    ]
    expectations = Dict(
        :atol => 1.0e-14,
        :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
        default_inverse_retraction_method => InverseProductRetraction(
            LogarithmicInverseRetraction(), LogarithmicInverseRetraction()
        ),
        default_retraction_method => ProductRetraction(
            StabilizedRetraction(), ExponentialRetraction()
        ),
        default_vector_transport_method => ProductVectorTransport(
            ParallelTransport(), ParallelTransport()
        ),
        distance => sqrt(π^2 / 4 + 1),
        exp => q,
        (get_coordinates, B) => [π / 2, 0.0, 1.0, 0.0],
        (get_vector, [1.0, 2.0, 3.0, 4.0], B) =>
            ArrayPartition([0.0, 1.0, 2.0], [3.0, 4.0]),
        # the diagonalizing basis at p starts with the direction e₃ on the sphere
        (get_coordinates, Bd) => [0.0, -π / 2, 1.0, 0.0],
        (get_vector, [1.0, 2.0, 3.0, 4.0], Bd) =>
            ArrayPartition([0.0, -2.0, 1.0], [3.0, 4.0]),
        (get_vectors, B) => :Orthonormal,
        (get_vectors, Bd) => :Orthonormal,
        injectivity_radius => π,
        (injectivity_radius, retraction_methods[1]) => π,
        (injectivity_radius, ExponentialRetraction()) => π,
        inner => π / 2 + 1,
        (inverse_retract, inverse_retraction_methods[1]) => X,
        (inverse_retract, LogarithmicInverseRetraction()) => X,
        is_default_metric => ProductMetric(),
        is_flat => false,
        log => X,
        manifold_dimension => 4,
        manifold_volume => Inf,
        mid_point => ArrayPartition([1 / sqrt(2), 1 / sqrt(2), 0.0], [0.5, 0.0]),
        norm => sqrt(π^2 / 4 + 1),
        parallel_transport_direction => X_at_q,
        parallel_transport_to => X_at_q,
        (project, :Point) => p,
        (project, :Vector) => X,
        repr => "ProductManifold(Sphere(2), Euclidean(2; field=ℝ))",
        (retract, retraction_methods[1]) => q,
        (retract, ExponentialRetraction()) => q,
        # only the sphere components of X and Y contribute to the curvature tensor
        sectional_curvature => (π^2 / 4) / ((π^2 / 4 + 1) * 7 - (π / 2 + 1)^2),
        sectional_curvature_max => 1.0,
        sectional_curvature_min => 0.0,
        sharp => X,
        (vector_transport_to, vector_transport_methods[1]) => X_at_q,
        (vector_transport_to, ParallelTransport()) => X_at_q,
        # sin(‖Xₛ‖)/‖Xₛ‖ on the sphere and one on the Euclidean space
        volume_density => 2 / π,
        Weingarten => ArrayPartition([0.0, -π, 0.0], [0.0, 0.0]),
        :IsPointErrors => [ComponentManifoldError, DomainError],
        :IsVectorErrors => [ComponentManifoldError],
        :IsVectorBasepointError => ComponentManifoldError,
    )

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Bases => [B, Bd],
            :Coordinates => [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            :Covectors => [flat(M, p, X)],
            :EmbeddedPoints => [ArrayPartition([2.0, 0.0, 0.0], [0.0, 0.0])],
            :EmbeddedVectors => [ArrayPartition([1.0, π / 2, 0.0], [1.0, 0.0])],
            :Functions => functions,
            # arrays are not points on a product manifold
            :InvalidPoints => [ArrayPartition([1.0, 1.0, 0.0], [0.0, 0.0]), [1.0, 2.0]],
            :InvalidVectors => [ArrayPartition([1.0, 1.0, 0.0], [0.0, 0.0])],
            :InverseRetractionMethods => inverse_retraction_methods,
            :NormalVectors => [V],
            :Points => [p, q, r],
            :RetractionMethods => retraction_methods,
            :SecondVector => Y,
            :Vectors => [X, log(M, q, r), log(M, r, p)],
            :VectorTransportMethods => vector_transport_methods,
            :Seed => 42,
        ),
        expectations,
    )

    @testset "Basics" begin
        @test M == M1 × M2
        @test M == ProductManifold(M1) × M2
        @test M == ProductManifold(M1) × ProductManifold(M2)
        @test M == M1 × ProductManifold(M2)
        @test M[1] == M1
        @test M[2] == M2
        @test Manifolds.number_of_components(M) == 2
        @test injectivity_radius(M, p, retraction_methods[1]) ≈ π
        @test check_point(M, [1, 2]) isa DomainError
        @test_throws DomainError is_vector(M, 1, [1, 2]; error = :error, check_base_point = false)
        @test check_vector(M, 1, [1, 2]; check_base_point = false) isa DomainError
        # the default fallbacks for check_size do not work with arrays
        @test Manifolds.check_size(M, zeros(2)) isa DomainError
        @test Manifolds.check_size(M, zeros(2), zeros(3)) isa DomainError
        @test number_eltype(p) === Float64
        @test (@inferred ManifoldsBase._get_vector_cache_broadcast(p)) === Val(false)
        @test ManifoldsBase.allocate_coordinates(
            M, p, Float64, number_of_coordinates(M, DefaultOrthogonalBasis())
        ) isa Vector{Float64}
        @test ManifoldsBase.allocate_coordinates(M, 1, Float64, 2) == [0.0, 0.0]
    end

    @testset "Cotangent vectors" begin
        ξ = flat(M, p, X)
        @test ξ(Y) ≈ π / 2 + 1
        @test allocate(ξ) isa typeof(ξ)
        Bc = DefaultOrthonormalBasis(ℝ, ManifoldsBase.CotangentSpaceType())
        # in orthonormal bases the musical isomorphisms keep the coordinates
        c = get_coordinates(M, p, ξ, Bc)
        @test c ≈ [π / 2, 0.0, 1.0, 0.0]
        @test sharp(M, p, get_vector(M, p, c, Bc)) ≈ X
        Xt = TFVector(c, B)
        ξt = flat(M, p, Xt)
        @test ξt isa CoTFVector
        @test ξt.data == c
        @test sharp(M, p, ξt).data == c
    end

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

    @testset "Three factors" begin
        M3 = Rotations(2)
        Mser = ProductManifold(M1, M2, M3)
        @test submanifold(Mser, 2) == M2
        @test (@inferred submanifold(Mser, Val((1, 3)))) == M1 × M3
        @test submanifold(Mser, 2:3) == M2 × M3
        @test submanifold(Mser, [1, 3]) == M1 × M3

        rot(ϕ) = [cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]
        p3 = ArrayPartition(p.x..., rot(0.0))
        q3 = ArrayPartition(q.x..., rot(π / 2))
        r3 = ArrayPartition(r.x..., rot(2π / 3))
        # tangent vectors on the rotations are represented in the Lie algebra
        X3 = ArrayPartition(X.x..., [0.0 π / 2; -π / 2 0.0])
        Manifolds.Test.test_manifold(
            Mser,
            Dict(
                :Functions => [
                    copy, copyto!, distance, exp, geodesic, inner, is_point, is_vector, log,
                    manifold_dimension, mid_point, norm, rand, retract, inverse_retract,
                    shortest_geodesic, zero_vector,
                ],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Points => [p3, q3, r3],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => ArrayPartition(Y.x..., [0.0 1.0; -1.0 0.0]),
                :Vectors => [X3, log(Mser, q3, r3), log(Mser, r3, p3)],
                :Seed => 42,
            ),
            Dict(
                :atol => 1.0e-14,
                :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
                # the norm on the rotations is the Frobenius norm
                distance => sqrt(π^2 / 4 + 1 + π^2 / 2),
                exp => q3,
                inner => π / 2 + 1 + π,
                (inverse_retract, LogarithmicInverseRetraction()) => X3,
                log => X3,
                manifold_dimension => 5,
                mid_point => ArrayPartition(mid_point(M, p, q).x..., rot(π / 4)),
                norm => sqrt(π^2 / 4 + 1 + π^2 / 2),
                (retract, ExponentialRetraction()) => q3,
            ),
        )
    end

    @testset "Statically sized components" begin
        ps = ArrayPartition(SizedVector{3}(p.x[1]), SizedVector{2}(p.x[2]))
        qs = ArrayPartition(SizedVector{3}(q.x[1]), SizedVector{2}(q.x[2]))
        rs = ArrayPartition(SizedVector{3}(r.x[1]), SizedVector{2}(r.x[2]))
        Xs = ArrayPartition(SizedVector{3}(X.x[1]), SizedVector{2}(X.x[2]))
        Bs = get_basis(M, ps, ProjectedOrthonormalBasis(:svd))
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [B, Bs],
                :Coordinates => [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                :Functions => [
                    copy, copyto!, distance, exp, get_basis, get_coordinates, get_vector, get_vectors,
                    injectivity_radius, inner, inverse_retract, is_point, is_vector, log,
                    mid_point, norm, parallel_transport_to, project, rand, retract,
                    vector_transport_to, zero_vector,
                ],
                :EmbeddedPoints => [2 .* ps],
                :EmbeddedVectors => [ArrayPartition(SizedVector{3}([1.0, π / 2, 0.0]), Xs.x[2])],
                :InverseRetractionMethods => inverse_retraction_methods,
                :Points => [ps, qs, rs],
                :RetractionMethods => retraction_methods,
                :SecondVector => ArrayPartition(SizedVector{3}(Y.x[1]), SizedVector{2}(Y.x[2])),
                :Vectors => [Xs, log(M, qs, rs), log(M, rs, ps)],
                :VectorTransportMethods => vector_transport_methods,
            ),
            merge(expectations, Dict((get_vectors, Bs) => :Orthonormal)),
        )
    end

    @testset "vee/hat" begin
        Mre = Rotations(3) × Euclidean(3)
        e = Matrix{Float64}(I, 3, 3)
        pre = ArrayPartition(exp(Rotations(3), e, hat(Rotations(3), e, [1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
        c = [0.1, 0.2, 0.3, -1.0, 2.0, -3.0]
        @test vee(Mre, pre, hat(Mre, pre, c)) ≈ c
    end

    @testset "Distributions" begin
        pa = allocate_result(M, Manifolds.uniform_distribution)
        @test pa isa ArrayPartition
        @test size(pa[M, 1]) == (3,)
        @test size(pa[M, 2]) == (2,)

        Mss = ProductManifold(Sphere(2), Sphere(2))
        pss = rand(Manifolds.uniform_distribution(Mss))
        @test is_point(Mss, pss)
        @test is_point(Mss, rand(Manifolds.uniform_distribution(Mss, pss)))

        ext = Base.get_extension(Manifolds, :ManifoldsDistributionsExt)
        pd = ext.ProductPointDistribution(
            Manifolds.uniform_distribution(M1, p.x[1]),
            Manifolds.projected_distribution(M2, Distributions.MvNormal(zeros(2), 1.0 * I)),
        )
        @test Distributions.support(pd) isa ext.MPointSupport
        @test is_point(M, rand(MersenneTwister(42), pd))
        pr = allocate(p)
        rand!(MersenneTwister(42), pd, pr)
        @test is_point(M, pr)
        vd = ext.ProductFVectorDistribution(
            Manifolds.normal_tvector_distribution(M1, p.x[1], 1.0),
            Manifolds.normal_tvector_distribution(M2, p.x[2], 1.0),
        )
        @test Distributions.support(vd) isa ext.FVectorSupport
        @test is_vector(M, p, rand(MersenneTwister(42), vd))
        Xr = allocate(p)
        rand!(MersenneTwister(42), vd, Xr)
        @test is_vector(M, p, Xr)
    end

    @testset "Atlases and induced bases" begin
        Mee = ProductManifold(Euclidean(2), Euclidean(2))
        pe = ArrayPartition(zeros(2), ones(2))
        RA = RetractionAtlas()
        @test get_point(Mee, RA, pe, get_parameters(Mee, RA, pe, pe)) == pe

        Mtt = ProductManifold(Manifolds.EmbeddedTorus(2.0, 1.0), Manifolds.EmbeddedTorus(2.0, 1.0))
        pt = ArrayPartition(([Manifolds._torus_param(Mi, 0.5, -1.2)...] for Mi in Mtt.manifolds)...)
        @test length(allocate_result(Mtt, get_parameters, pt)) == 4
        @test allocate_result(Mtt, get_point, zeros(4)) isa typeof(pt)

        A = get_default_atlas(M)
        @test A isa ProductAtlas
        @test A.atlases == (get_default_atlas(M1), get_default_atlas(M2))
        pa = ArrayPartition([1.0, 0.0, 0.0], [2.0, 3.0])
        i = get_chart_index(M, A, pa)
        @test i == (pa.x[1], pa.x[2])
        a = get_parameters(M, A, i, pa)
        @test length(a) == manifold_dimension(M)
        @test isapprox(M, get_point(M, A, i, a), pa)
        @test isapprox(
            M, get_point(M, A, i, ArrayPartition([0.1, 0.2], [0.3, 0.4])),
            get_point(M, A, i, [0.1, 0.2, 0.3, 0.4]),
        )
        A2 = ProductAtlas(StereographicAtlas(), get_default_atlas(M2))
        i2 = get_chart_index(M, A2, pa)
        @test i2 == (:north, pa.x[2])
        @test get_chart_index(M, A2, i2, [3.0, 3.0, 3.0, 3.0]) == (:south, [5.0, 6.0])

        Mss = ProductManifold(Sphere(2), Sphere(2))
        As = ProductAtlas(StereographicAtlas(), StereographicAtlas())
        pss = ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        Xss = ArrayPartition([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
        Bi = induced_basis(Mss, As, get_chart_index(Mss, As, pss))
        c = get_coordinates(Mss, pss, Xss, Bi)
        @test get_vector(Mss, pss, c, Bi) ≈ Xss
        Z = similar(Xss)
        get_vector!(Mss, Z, pss, c, Bi)
        @test isapprox(Mss, pss, Xss, Z)
        c2 = similar(c)
        get_coordinates!(Mss, c2, pss, Xss, Bi)
        @test c2 ≈ c
    end

    @testset "Componentwise conversions" begin
        Mspd = SymmetricPositiveDefinite(3)
        N = ProductManifold(Mspd, Mspd)
        e = EuclideanMetric()
        P = ArrayPartition(Matrix(1.0I, 3, 3), Matrix(Diagonal([2.0, 2.0, 1.0])))
        Xp = ArrayPartition(log(Mspd, P.x[1], P.x[2]), log(Mspd, P.x[2], P.x[1]))
        @test change_metric(N, e, P, Xp) ≈
            ArrayPartition(map((pᵢ, Xᵢ) -> change_metric(Mspd, e, pᵢ, Xᵢ), P.x, Xp.x)...)
        @test change_representer(N, e, P, Xp) ≈
            ArrayPartition(map((pᵢ, Xᵢ) -> change_representer(Mspd, e, pᵢ, Xᵢ), P.x, Xp.x)...)

        Mps = ProbabilitySimplex(3)
        pp = [0.1, 0.2, 0.3, 0.4]
        Xe = [0.5, -1.0, 2.0, 0.0]
        Np = Mps × Mps
        Xr = riemannian_gradient(Np, ArrayPartition(pp, pp), ArrayPartition(Xe, Xe))
        @test Xr ≈ ArrayPartition(riemannian_gradient(Mps, pp, Xe), riemannian_gradient(Mps, pp, Xe))
        Zr = similar(Xr)
        riemannian_gradient!(Np, Zr, ArrayPartition(pp, pp), ArrayPartition(Xe, Xe))
        @test Zr ≈ Xr

        Mss = M1 × M1
        pss = ArrayPartition([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        Xss = ArrayPartition([0.0, 1.0, 1.0], [1.0, 1.0, 0.0])
        Yss = ArrayPartition([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
        Zss = ArrayPartition([0.5, 2.0, 0.0], [0.0, 1.0, 3.0])
        @test riemannian_Hessian(Mss, pss, Yss, Zss, Xss) == ArrayPartition(
            [riemannian_Hessian(M1, pss.x[i], Yss.x[i], Zss.x[i], Xss.x[i]) for i in 1:2]...
        )
        Vss = ArrayPartition([0.2, 0.0, 0.0], [0.0, 0.0, 0.3])
        @test Weingarten(Mss, pss, Xss, Vss) ==
            ArrayPartition([Weingarten(M1, pss.x[i], Xss.x[i], Vss.x[i]) for i in 1:2]...)
    end

    @testset "Volume" begin
        PM = ProductManifold(Sphere(2), Sphere(3))
        @test manifold_volume(PM) ≈ 4π * 2π^2
        # the densities are sin(‖X‖)/‖X‖ on S² and (sin(‖X‖)/‖X‖)² on S³
        pv = ArrayPartition([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        Xv = ArrayPartition([0.0, π / 2, 0.0], [0.0, 0.0, 0.0, π / 2])
        @test volume_density(PM, pv, Xv) ≈ (2 / π)^3
    end

    @testset "default retraction, inverse retraction and vector transport" begin
        Mstb = ProductManifold(M1, TangentBundle(M1))
        T_p_ap = ArrayPartition{
            Float64,
            Tuple{Matrix{Float64}, ArrayPartition{Float64, Tuple{Matrix{Float64}, Matrix{Float64}}}},
        }
        rm = ProductRetraction(StabilizedRetraction(), Manifolds.FiberBundleProductRetraction())
        irm = InverseProductRetraction(
            LogarithmicInverseRetraction(), Manifolds.FiberBundleInverseProductRetraction()
        )
        vtm = ProductVectorTransport(
            ParallelTransport(),
            Manifolds.FiberBundleProductVectorTransport(ParallelTransport(), ParallelTransport()),
        )
        @test default_retraction_method(Mstb) === rm
        @test default_retraction_method(Mstb, T_p_ap) === rm
        @test default_inverse_retraction_method(Mstb) === irm
        @test default_inverse_retraction_method(Mstb, T_p_ap) === irm
        @test default_vector_transport_method(Mstb) === vtm
        @test default_vector_transport_method(Mstb, T_p_ap) === vtm
    end

    @testset "ManifoldDiff" begin
        pd = ArrayPartition([0.0, 1.0, 0.0], [2.0, 3.0])
        qd = ArrayPartition([1.0, 0.0, 0.0], [-2.0, 3.0])
        Xd = ArrayPartition([1.0, 0.0, 0.0], [2.0, 3.0])
        β = ManifoldDiff.βdifferential_shortest_geodesic_startpoint
        Jd = ArrayPartition([sqrt(2) / 4, -sqrt(2) / 4, 0.0], [1.0, 1.5])
        aJd = ArrayPartition([0.5, 0.0, 0.0], [1.0, 1.5])
        @test ManifoldDiff.adjoint_Jacobi_field(M, pd, qd, 0.5, Xd, β) ≈ aJd
        Zd = allocate(Xd)
        ManifoldDiff.adjoint_Jacobi_field!(M, Zd, pd, qd, 0.5, Xd, β)
        @test Zd ≈ aJd
        @test ManifoldDiff.jacobi_field(M, pd, qd, 0.5, Xd, β) ≈ Jd
        ManifoldDiff.jacobi_field!(M, Zd, pd, qd, 0.5, Xd, β)
        @test Zd ≈ Jd
    end
end
