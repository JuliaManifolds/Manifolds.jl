using LinearAlgebra, Manifolds, ManifoldsBase, Test
using RecursiveArrayTools: ArrayPartition

# two objects created by f with separately allocated arrays are equal and have equal hashes,
# while an object created by g is different
function test_value_equality(f, g)
    a, b = f(), f()
    @test a == b
    @test hash(a) == hash(b)
    @test isequal(a, b)
    return @test a != g()
end

# allocations of == and hash after compilation, measured within a function to avoid globals
eq_allocations(a, b) = (a == b; @allocated a == b)
hash_allocations(a) = (hash(a); @allocated hash(a))

@testset "Equality by value" begin
    B = [2.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    p = [1.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0]
    S = Sphere(2)
    # a manifold storing an array
    GS(B) = GeneralizedStiefel(3, 2, copy(B))
    # a point on the Tucker manifold, whose core is scaled by c
    tp(c) = TuckerPoint(c .* [1.0 0.0; 0.0 2.0], [1.0 0.0; 0.0 1.0; 0.0 0.0], [1.0 0.0; 0.0 1.0; 0.0 0.0])

    @testset "manifolds and methods" begin
        test_value_equality(() -> TangentBundle(GS(B)), () -> TangentBundle(GS(2 * B)))
        # wrapped manifolds from ManifoldsBase.jl are also compared by value
        test_value_equality(() -> ProductManifold(GS(B), S), () -> ProductManifold(GS(2 * B), S))
        test_value_equality(() -> PowerManifold(GS(B), 2), () -> PowerManifold(GS(2 * B), 2))
        vt(a) = VectorTransportWithKeywords(ParallelTransport(); a = copy(a))
        test_value_equality(
            () -> Manifolds.FiberBundleProductVectorTransport(vt(p), ParallelTransport()),
            () -> Manifolds.FiberBundleProductVectorTransport(vt(q), ParallelTransport()),
        )
        f!(q, p) = (q .= p)
        J!(J, p) = (J .= 1)
        ps(lb) = ParametricSurface(Hyperrectangle(copy(lb), [1.0, 1.0]), Euclidean(2), f!, f!, J!)
        test_value_equality(() -> ps([0.0, 0.0]), () -> ps([-1.0, 0.0]))
    end

    @testset "atlases, charts and bases" begin
        Bc(p) = get_basis(S, copy(p), DefaultOrthonormalBasis())
        ra(p) = RetractionAtlas{ℝ, ExponentialRetraction, LogarithmicInverseRetraction, typeof(Bc(p))}(
            ExponentialRetraction(), LogarithmicInverseRetraction(), Bc(p)
        )
        test_value_equality(() -> ra(p), () -> ra(q))
        @test RetractionAtlas() != RetractionAtlas{ℂ, ExponentialRetraction, LogarithmicInverseRetraction, typeof(DefaultOrthonormalBasis())}(
            ExponentialRetraction(), LogarithmicInverseRetraction(), DefaultOrthonormalBasis()
        )
        test_value_equality(() -> PowerAtlas(ra(p)), () -> PowerAtlas(ra(q)))
        test_value_equality(() -> ProductAtlas(ra(p), ra(q)), () -> ProductAtlas(ra(q), ra(p)))
        test_value_equality(() -> Manifolds.ParametricSurfaceAtlas(ra(p)), () -> Manifolds.ParametricSurfaceAtlas(ra(q)))
        St = Stiefel(3, 2)
        A = get_default_atlas(St)
        ch(p) = get_chart_index(St, A, copy(p))
        test_value_equality(() -> ch([1.0 0.0; 0.0 1.0; 0.0 0.0]), () -> ch([0.0 0.0; 0.0 1.0; 1.0 0.0]))
        test_value_equality(
            () -> induced_basis(S, RetractionAtlas(), copy(p), TangentSpaceType()),
            () -> induced_basis(S, RetractionAtlas(), q, TangentSpaceType()),
        )
        TB = TangentBundle(S)
        test_value_equality(
            () -> get_basis(TB, ArrayPartition(copy(p), copy(q)), DefaultOrthonormalBasis()),
            () -> get_basis(TB, ArrayPartition(q, p), DefaultOrthonormalBasis()),
        )
        M = Tucker((3, 3), (2, 2))
        test_value_equality(
            () -> get_basis(M, tp(1.0), DefaultOrthonormalBasis()),
            () -> get_basis(M, tp(2.0), DefaultOrthonormalBasis()),
        )
    end

    @testset "points and vectors" begin
        test_value_equality(() -> flat(S, copy(p), copy(q)), () -> flat(S, p, [0.0, 0.0, 1.0]))
        sf(U) = Manifolds.StiefelFactorization(copy(U), [1.0 0.0; 0.0 1.0])
        test_value_equality(() -> sf([1.0 0.0; 0.0 1.0; 0.0 0.0]), () -> sf([0.0 0.0; 0.0 1.0; 1.0 0.0]))
        P = [2.0 0.5; 0.5 1.0]
        test_value_equality(() -> SPDPoint(copy(P)), () -> SPDPoint(2 * P))
        U = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        test_value_equality(() -> SVDMPoint(copy(U), [2.0, 1.0], [1.0 0.0; 0.0 1.0]), () -> SVDMPoint(U, [3.0, 1.0], [1.0 0.0; 0.0 1.0]))
        test_value_equality(() -> UMVTangentVector(copy(U), [1.0 0.0; 0.0 1.0], zeros(2, 2)), () -> UMVTangentVector(U, [2.0 0.0; 0.0 1.0], zeros(2, 2)))
        test_value_equality(() -> VeronesePoint(copy(p), [1.0]), () -> VeronesePoint(q, [1.0]))
        test_value_equality(() -> VeroneseTangentVector(copy(p), [1.0]), () -> VeroneseTangentVector(q, [1.0]))
        test_value_equality(() -> tp(1.0), () -> tp(2.0))
        test_value_equality(() -> tp(1.0).hosvd, () -> tp(2.0).hosvd)
        X = TuckerTangentVector([1.0 0.0; 0.0 1.0], ([0.0 0.0; 0.0 0.0; 1.0 0.0], [0.0 0.0; 0.0 0.0; 0.0 1.0]))
        test_value_equality(() -> deepcopy(X), () -> 2 * X)
        # point types with a forwarded field, from ManifoldsBase.jl
        test_value_equality(() -> StiefelPoint(copy(U)), () -> StiefelPoint(2 * U))
        test_value_equality(() -> HyperboloidPoint([0.0, 0.0, 1.0]), () -> HyperboloidPoint([1.0, 0.0, sqrt(2)]))
    end

    @testset "type stability of the types with given fields" begin
        P = [2.0 0.5; 0.5 1.0]
        U = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        X = TuckerTangentVector([1.0 0.0; 0.0 1.0], ([0.0 0.0; 0.0 0.0; 1.0 0.0], [0.0 0.0; 0.0 0.0; 0.0 1.0]))
        for a in (
                TangentBundle(GS(B)),
                induced_basis(S, RetractionAtlas(), copy(p), TangentSpaceType()),
                RetractionAtlas(),
                SPDPoint(P),
                SVDMPoint(U, [2.0, 1.0], [1.0 0.0; 0.0 1.0]),
                UMVTangentVector(U, [1.0 0.0; 0.0 1.0], zeros(2, 2)),
                VeronesePoint(p, [1.0]),
                VeroneseTangentVector(p, [1.0]),
                tp(1.0),
                tp(1.0).hosvd,
                X,
            )
            b = deepcopy(a)
            @test (@inferred a == b)
            @test (@inferred hash(a)) == hash(b)
            @test eq_allocations(a, b) == 0
            @test hash_allocations(a) == 0
        end
    end
end
