using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test
using RecursiveArrayTools: ArrayPartition

@testset "Fiber bundle" begin
    @testset "Invalid points and vectors" begin
        TB = TangentBundle(Sphere(2))
        p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        X = ArrayPartition([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        Manifolds.Test.test_manifold(
            TB,
            Dict(
                :Functions => [is_point, is_vector],
                # a wrong base part, a wrong fiber part, and both wrong
                :InvalidPoints => [
                    ArrayPartition([2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                    ArrayPartition([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                    ArrayPartition([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                ],
                :InvalidVectors => [
                    ArrayPartition([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
                    ArrayPartition([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
                    ArrayPartition([1.0, 1.0, 1.0], [1.0, 0.0, 0.0]),
                ],
                :Points => [p],
                :Vectors => [X],
            ),
            # Expectations
            Dict(
                :IsPointErrors => [ComponentManifoldError, ComponentManifoldError, CompositeManifoldError],
                :IsVectorErrors => [ComponentManifoldError, ComponentManifoldError, CompositeManifoldError],
                :IsVectorBasepointError => ComponentManifoldError,
            ),
        )
    end

    @testset "Tangent bundle of the projective space" begin
        TB = TangentBundle(ProjectiveSpace(2))
        p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 0.5, 0.0])
        X = ArrayPartition([0.0, 0.5, 0.0], [0.0, 0.0, 0.5])
        Y = ArrayPartition([0.0, 2.0, 1.0], [0.0, 1.0, -1.0])
        Manifolds.Test.test_manifold(
            TB,
            Dict(
                :Functions => [
                    copy, copyto!, inner, is_point, is_vector, manifold_dimension, norm, rand,
                    repr, zero_vector,
                ],
                :Points => [p],
                :SecondVector => Y,
                :Vectors => [X, Y],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14,
                # the metric is the sum of the metrics of the base and the fiber
                inner => 0.5,
                manifold_dimension => 4,
                norm => sqrt(0.5),
                repr => "TangentBundle(ProjectiveSpace(2, ℝ))",
            ),
        )
        # tangent vectors are compared as tangent vectors at the base point, not as points
        # of the projective space, where X.x[1] and -X.x[1] would be equal
        @test isapprox(TB, p, X, X)
        @test isapprox(TB, p, zero_vector(TB, p), zero_vector(TB, p))
        @test !isapprox(TB, p, X, ArrayPartition([0.0, 0.7, 0.0], [0.0, 0.0, 0.5]))
        @test !isapprox(TB, p, X, ArrayPartition([0.0, -0.5, 0.0], [0.0, 0.0, 0.5]))
    end

    @testset "Tangent bundle of the SPD matrices" begin
        TB = TangentBundle(SymmetricPositiveDefinite(3))
        p = ArrayPartition(Matrix(1.0I, 3, 3), zeros(3, 3))
        X = ArrayPartition([1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0])
        Manifolds.Test.test_manifold(
            TB,
            Dict(
                :Functions => [inner, is_point, is_vector, manifold_dimension, norm, rand],
                :InvalidVectors => [ArrayPartition(zeros(3, 3), [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])],
                :Points => [p],
                :SecondVector => X,
                :Vectors => [X],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                # the affine invariant metric at the identity is the Frobenius one
                inner => 3.0,
                manifold_dimension => 12,
                norm => sqrt(3.0),
                :IsVectorErrors => [ComponentManifoldError],
            ),
        )
    end

    @testset "Product vector transport" begin
        M = Stiefel(3, 2)
        vm = default_vector_transport_method(M)
        @test Manifolds.FiberBundleProductVectorTransport(M) ==
            Manifolds.FiberBundleProductVectorTransport(vm, vm)
    end
end
