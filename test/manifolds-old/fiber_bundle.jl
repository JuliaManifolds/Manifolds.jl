include("../header.jl")

using RecursiveArrayTools

@testset "fiber bundle" begin
    M = Stiefel(3, 2)
    vm = default_vector_transport_method(M)
    @test Manifolds.FiberBundleProductVectorTransport(M) ==
        Manifolds.FiberBundleProductVectorTransport(vm, vm)

    TB = TangentBundle(Sphere(2))
    p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    @test is_point(TB, p)
    @test is_vector(TB, p, ArrayPartition([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))
    @test is_vector(TB, p, zero_vector(TB, p))
    # a wrong base part, a wrong fiber part, and both wrong
    q1 = ArrayPartition([2.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    q2 = ArrayPartition([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    q3 = ArrayPartition([2.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    @test_throws ComponentManifoldError is_point(TB, q1; error = :error)
    @test_throws ComponentManifoldError is_point(TB, q2; error = :error)
    @test_throws CompositeManifoldError is_point(TB, q3; error = :error)
    X1 = ArrayPartition([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    X2 = ArrayPartition([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    X3 = ArrayPartition([1.0, 1.0, 1.0], [1.0, 0.0, 0.0])
    @test_throws ComponentManifoldError is_vector(TB, p, X1; error = :error)
    @test_throws ComponentManifoldError is_vector(TB, p, X2; error = :error)
    @test_throws CompositeManifoldError is_vector(TB, p, X3; error = :error)

    # random tangent vectors at a fixed point lie in the fiber over its base point
    B = TangentBundle(Sphere(2))
    pb = ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    @test is_vector(B, pb, rand(MersenneTwister(2), B; vector_at = pb))
    BS = TangentBundle(SymmetricPositiveDefinite(3))
    pS = ArrayPartition(Matrix(1.0I, 3, 3), zeros(3, 3))
    @test is_vector(BS, pS, rand(MersenneTwister(2), BS; vector_at = pS))
end
