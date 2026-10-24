using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test
using RecursiveArrayTools: ArrayPartition

struct TestVectorSpaceType <: VectorSpaceType end

# a fiber element holds two tangent vectors, each moved by the transport of the base manifold
function Manifolds.bundle_transport_tangent_direction!(
        B::FiberBundle{<:Any, TestVectorSpaceType}, Y, p, pf, X, d, m
    )
    for i in 1:2
        vector_transport_direction!(B.manifold, Y.x[i], p, X.x[i], d, m)
    end
    return Y
end

@testset "Vector bundle" begin
    M = Sphere(2)
    TB = TangentBundle(M)
    m_prod_retr = Manifolds.FiberBundleProductRetraction()
    m_prod_invretr = Manifolds.FiberBundleInverseProductRetraction()
    m_prod_vt = Manifolds.FiberBundleProductVectorTransport()
    B = DefaultOrthonormalBasis()

    @testset "Tangent bundle of the sphere" begin
        # the base points move along a quarter of a great circle, whose parallel transport
        # maps e₂ to -e₁ and fixes e₃
        p = ArrayPartition([1.0, 0.0, 0.0], [0.0, -1.0, -1.0])
        q = ArrayPartition([0.0, 1.0, 0.0], [2.0, 0.0, 1.0])
        r = ArrayPartition([1.0, 0.0, 0.0], [0.0, 2.0, -1.0])
        # the logarithmic map of the base points and the difference of the fiber part of p
        # and the fiber part of q transported back to e₁, (0, -2, 1)
        X = ArrayPartition([0.0, π / 2, 0.0], [0.0, -1.0, 2.0])
        Y = ArrayPartition([0.0, 1.0, 2.0], [0.0, 3.0, 4.0])
        Bc = get_basis(TB, p, B)
        @test get_basis(TB, p, Bc) === Bc
        Manifolds.Test.test_manifold(
            TB,
            Dict(
                :Bases => [B, Bc],
                :Coordinates => [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                :Covectors => [flat(TB, p, X)],
                :EmbeddedPoints => [ArrayPartition([2.0, 0.0, 0.0], [1.0, -1.0, -1.0])],
                :EmbeddedVectors => [ArrayPartition([1.0, π / 2, 0.0], [1.0, -1.0, 2.0])],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method,
                    default_retraction_method, default_vector_transport_method, flat,
                    get_basis, get_coordinates, get_vector, get_vectors, injectivity_radius,
                    inner, inverse_retract, is_flat, is_point, is_vector, manifold_dimension,
                    norm, project, rand, repr, representation_size, retract, sharp,
                    vector_transport_to, zero_vector,
                ],
                :InverseRetractionMethods => [m_prod_invretr, missing],
                :InvalidPoints => [ArrayPartition([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
                :InvalidVectors => [ArrayPartition([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])],
                :Points => [p, q, r],
                :RetractionMethods => [m_prod_retr, SasakiRetraction(5)],
                :SecondVector => Y,
                :Vectors => [
                    X,
                    ArrayPartition([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                    ArrayPartition([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
                ],
                :VectorTransportMethods => [m_prod_vt],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14,
                default_inverse_retraction_method => m_prod_invretr,
                default_retraction_method => m_prod_retr,
                default_vector_transport_method => m_prod_vt,
                # the coordinates at e₁ are the second and third entries of both parts
                (get_coordinates, B) => [π / 2, 0.0, -1.0, 2.0],
                (get_vector, [1.0, 2.0, 3.0, 4.0], B) => Y,
                (get_coordinates, Bc) => [π / 2, 0.0, -1.0, 2.0],
                (get_vector, [1.0, 2.0, 3.0, 4.0], Bc) => Y,
                (get_vectors, B) => :Orthonormal,
                (get_vectors, Bc) => :Orthonormal,
                injectivity_radius => 0.0,
                # the metric is the sum of the metrics of the base and the fiber
                inner => π / 2 + 5,
                (inverse_retract, m_prod_invretr) => X,
                is_flat => false,
                manifold_dimension => 4,
                norm => sqrt(π^2 / 4 + 5),
                (project, :Point) => p,
                (project, :Vector) => X,
                repr => "TangentBundle(Sphere(2))",
                representation_size => nothing,
                # the base point moves to e₂ and the fiber part (0, -2, 1) is transported
                (retract, m_prod_retr) => q,
                sharp => X,
                (vector_transport_to, m_prod_vt) =>
                    ArrayPartition([-π / 2, 0.0, 0.0], [1.0, 0.0, 2.0]),
                :IsPointErrors => [ComponentManifoldError],
                :IsVectorErrors => [ComponentManifoldError],
            ),
        )
        for pt in (p, q, r)
            @test bundle_projection(TB, pt) == pt.x[1]
        end
        @test distance(Fiber(M, p[TB, :point], TangentSpaceType()), p[TB, :vector], [0.0, 2.0, 3.0]) == 5.0
        @test Manifolds.bundle_transport_to(TB, p.x[1], p.x[2], q.x[1]) ≈ [1.0, 0.0, -1.0]
        # the mutating variants without a vector transport method use the default one
        Z = zero_vector(TB, q)
        vector_transport_to!(TB, Z, p, X, q)
        @test Z ≈ ArrayPartition([-π / 2, 0.0, 0.0], [1.0, 0.0, 2.0])
        Z = zero_vector(TB, p)
        get_vector!(TB, Z, p, [1.0, 2.0, 3.0, 4.0], B)
        @test Z ≈ Y
    end

    @testset "Tangent space" begin
        p = [1.0, 0.0, 0.0]
        TpM = TangentSpace(M, p)
        # points and tangent vectors of the tangent space are tangent vectors of the sphere
        a, b, c = [0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, -1.0, 1.0]
        Bc = get_basis(TpM, a, B)
        @test get_basis(TpM, a, Bc) === Bc
        Manifolds.Test.test_manifold(
            TpM,
            Dict(
                :Bases => [B, Bc],
                :Coordinates => [[1.0, 2.0], [1.0, 2.0]],
                :EmbeddedPoints => [[1.0, 0.0, 1.0]],
                :EmbeddedVectors => [[3.0, 2.0, -1.0]],
                :Functions => [
                    copy, copyto!, distance, embed, exp, geodesic, get_basis, get_coordinates,
                    get_embedding, get_vector, get_vectors, injectivity_radius, inner,
                    inverse_retract, is_flat,
                    is_point, is_vector, log, manifold_dimension, mid_point, norm,
                    parallel_transport_to, project, rand, retract, shortest_geodesic,
                    vector_transport_to, zero_vector,
                ],
                :InvalidPoints => [[1.0, 0.0, 0.0]],
                :InvalidVectors => [[1.0, 0.0, 0.0]],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Points => [a, b, c],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => [0.0, 1.0, 1.0],
                :Vectors => [b - a, c - b, a - c],
                :VectorTransportMethods => [ParallelTransport()],
                :Seed => 42,
            ),
            Dict(
                :atol => 1.0e-14,
                # the tangent space is a Euclidean space
                distance => sqrt(5),
                (embed, :Point) => a,
                (embed, :Vector) => b - a,
                exp => b,
                get_embedding => TangentSpace(Euclidean(3), p),
                (get_coordinates, B) => [2.0, -1.0],
                (get_vector, [1.0, 2.0], B) => [0.0, 1.0, 2.0],
                (get_vectors, B) => :Orthonormal,
                injectivity_radius => Inf,
                inner => 1.0,
                (inverse_retract, LogarithmicInverseRetraction()) => b - a,
                is_flat => true,
                log => b - a,
                manifold_dimension => 2,
                mid_point => [0.0, 1.0, 0.5],
                norm => sqrt(5),
                parallel_transport_to => b - a,
                (project, :Point) => a,
                (project, :Vector) => [0.0, 2.0, -1.0],
                (retract, ExponentialRetraction()) => b,
                (vector_transport_to, ParallelTransport()) => b - a,
                :IsPointErrors => [DomainError],
                :IsVectorErrors => [DomainError],
            ),
        )
    end

    @testset "Nice access to vector bundle components" begin
        p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 2.0, 4.0])
        @test p[TB, :point] === p.x[1]
        @test p[TB, :vector] === p.x[2]
        p[TB, :vector] = [0.0, 3.0, 1.0]
        @test p.x[2] == [0.0, 3.0, 1.0]
        p[TB, :point] = [0.0, 1.0, 0.0]
        @test p.x[1] == [0.0, 1.0, 0.0]
        @test_throws DomainError p[TB, :error]
        @test_throws DomainError p[TB, :error] = [1, 2, 3]
        @test view(p, TB, :point) === p.x[1]
        @test view(p, TB, :vector) === p.x[2]
        view(p, TB, :point) .= [2.0, 3.0, 5.0]
        @test p.x[1] == [2.0, 3.0, 5.0]
        view(p, TB, :vector) .= [-2.0, -3.0, -5.0]
        @test p.x[2] == [-2.0, -3.0, -5.0]
        @test_throws DomainError view(p, TB, :error)
    end

    @testset "Basics" begin
        @test base_manifold(TB) == M
        CTB = CotangentBundle(M)
        @test sprint(show, CTB) == "CotangentBundle(Sphere(2))"
        @test sprint(show, FiberBundle(TestVectorSpaceType(), M)) ==
            "FiberBundle(TestVectorSpaceType(), Sphere(2), Manifolds.FiberBundleProductVectorTransport{ParallelTransport, ParallelTransport}(ParallelTransport(), ParallelTransport()))"
        @test Manifolds.fiber_dimension(M, ManifoldsBase.CotangentSpaceType()) == 2
        @test is_flat(TangentSpace(M, [1.0, 0.0, 0.0]))
        @test TangentBundle{ℝ, Sphere{ℝ, 2}} ==
            VectorBundle{ℝ, Manifolds.TangentSpaceType, Sphere{ℝ, 2}}
        @test CotangentBundle{ℝ, Sphere{ℝ, 2}} ==
            VectorBundle{ℝ, Manifolds.CotangentSpaceType, Sphere{ℝ, 2}}
        TE = TangentBundle(Euclidean(3))
        @test is_flat(TE)
        @test injectivity_radius(TE) == Inf
    end

    @testset "tensor product" begin
        TT = Manifolds.TensorProductType(TangentSpaceType(), TangentSpaceType())
        @test sprint(show, TT) == "TensorProductType(TangentSpaceType(), TangentSpaceType())"
        @test vector_space_dimension(Sphere(2), TT) == 4
        @test vector_space_dimension(Sphere(3), TT) == 9
        @test base_manifold(Fiber(Sphere(2), [1.0, 0.0, 0.0], TT)) == M
        @test sprint(show, Fiber(Sphere(2), [1.0, 0.0, 0.0], TT)) ==
            "VectorSpaceFiber{ℝ, $(typeof(Sphere(2))), Manifolds.TensorProductType{Tuple{TangentSpaceType, TangentSpaceType}}, Vector{Float64}}(Sphere(2), [1.0, 0.0, 0.0], TensorProductType(TangentSpaceType(), TangentSpaceType()))"
    end

    @testset "Error messages" begin
        vbf = Fiber(Euclidean(3), [1.0, 0.0, 0.0], TestVectorSpaceType())
        @test_throws MethodError inner(vbf, [1, 2, 3], [1, 2, 3], [1, 2, 3])
        @test_throws MethodError project!(vbf, [1, 2, 3], [1, 2, 3], [1, 2, 3])
        @test_throws MethodError zero_vector!(vbf, [1, 2, 3], [1, 2, 3])
        @test_throws MethodError vector_space_dimension(vbf)
    end

    @testset "in-place vector transport in a direction uses the fiber transport" begin
        FB = FiberBundle(TestVectorSpaceType(), Sphere(2))
        p = ArrayPartition([1.0, 0.0, 0.0], ArrayPartition([0.0, -1.0, -1.0], [0.0, -1.0, -1.0]))
        X = ArrayPartition([0.0, 1.0, 0.5], ArrayPartition([0.0, 0.0, 1.0], [0.0, 2.0, 0.0]))
        d = ArrayPartition([0.0, π / 2, 0.0], ArrayPartition(zeros(3), zeros(3)))
        Z = ArrayPartition([-1.0, 0.0, 0.5], ArrayPartition([0.0, 0.0, 1.0], [-2.0, 0.0, 0.0]))
        @test isapprox(vector_transport_direction(FB, p, X, d), Z)
        Y = zero(X)
        @test vector_transport_direction!(FB, Y, p, X, d) === Y
        @test isapprox(Y, Z)
    end

    @testset "product retraction and inverse retraction on tangent bundles of power and product manifolds" begin
        MP = PowerManifold(Circle(ℝ), 2)
        N = TangentBundle(MP)
        p1 = ArrayPartition([0.0, 0.0], [0.0, 0.0])
        p2 = ArrayPartition([-1.047, -1.047], [0.0, 0.0])
        X1 = inverse_retract(N, p1, p2, m_prod_invretr)
        @test X1 ≈ ArrayPartition([-1.047, -1.047], [0.0, 0.0])
        @test isapprox(N, p2, retract(N, p1, X1, m_prod_retr))
        @test isapprox(N, p2, ManifoldsBase.retract_fused(N, p1, X1, 1.0, m_prod_retr))
        p3 = deepcopy(p1)
        retract!(N, p3, p1, X1, m_prod_retr)
        @test isapprox(N, p2, p3)
        ManifoldsBase.retract_fused!(N, p3, p1, X1, 1.0, m_prod_retr)
        @test isapprox(N, p2, p3)
        @test is_vector(N, p2, vector_transport_to(N, p1, X1, p2))

        N2 = TangentBundle(ProductManifold(Circle(ℝ), Euclidean(2)))
        p1_2 = ArrayPartition(ArrayPartition([0.0], [0.0, 0.0]), ArrayPartition([0.0], [0.0, 0.0]))
        p2_2 = ArrayPartition(ArrayPartition([-1.047], [1.0, 0.0]), ArrayPartition([-1.047], [0.0, 1.0]))
        @test isapprox(
            N2, p2_2,
            retract(N2, p1_2, inverse_retract(N2, p1_2, p2_2, m_prod_invretr), m_prod_retr),
        )

        ppt = ParallelTransport()
        tbvt = Manifolds.FiberBundleProductVectorTransport(ppt, ppt)
        @test TangentBundle(MP, tbvt).vector_transport === tbvt
        @test CotangentBundle(MP, tbvt).vector_transport === tbvt
        @test TangentBundle(MP, tbvt) isa TangentBundle
        @test CotangentBundle(MP, tbvt) isa CotangentBundle
    end

    @testset "product retraction transports the fiber part with the vertical method" begin
        ppt, prt = ParallelTransport(), ProjectionTransport()
        B1 = TangentBundle(M, Manifolds.FiberBundleProductVectorTransport(ppt, prt))
        B2 = TangentBundle(M, Manifolds.FiberBundleProductVectorTransport(prt, ppt))
        p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        X = ArrayPartition([0.0, 0.3, 0.0], [0.0, 0.0, 0.5])
        xq = exp(M, p.x[1], X.x[1])
        q1 = retract(B1, p, X, m_prod_retr)
        @test isapprox(M, xq, q1.x[2], project(M, xq, [0.0, 1.0, 0.5]))
        q2 = retract(B2, p, X, m_prod_retr)
        @test isapprox(B2, p, inverse_retract(B2, p, q2, m_prod_invretr), X)
    end

    @testset "cotangent basis on a tangent bundle" begin
        p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        c = [1.0, 2.0, 3.0, 4.0]
        Bc = DefaultOrthonormalBasis(ℝ, CotangentSpaceType())
        ξ = get_vector(TB, p, c, Bc)
        @test ξ isa Manifolds.RieszRepresenterCotangentVector
        @test ξ.X ≈ get_vector(TB, p, c, DefaultOrthonormalBasis())
        @test get_coordinates(TB, p, ξ, Bc) ≈ c
    end
end
