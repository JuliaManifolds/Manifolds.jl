include("utils.jl")

@testset "Positive Numbers" begin
    M = PositiveNumbers()
    @testset "Positive Numbers Basics" begin
        @test repr(M) == "PositiveNumbers()"
        @test repr(PositiveVectors(2)) == "PositiveVectors(2)"
        @test repr(PositiveMatrices(2, 3)) == "PositiveMatrices(2, 3)"
        @test repr(PositiveArrays(2, 3, 4)) == "PositiveArrays(2, 3, 4)"
        @test representation_size(M) == ()
        @test manifold_dimension(M) == 1
        @test !is_point(M, -1.0)
        @test_throws DomainError is_point(M, -1.0, true)
        @test is_vector(M, 1.0, 0.0)
        @test vector_transport_to(M, 1.0, 3.0, 2.0, ParallelTransport()) == 6.0
        @test retract(M, 1.0, 1.0) == exp(M, 1.0, 1.0)
        @test isinf(injectivity_radius(M))
        @test isinf(injectivity_radius(M, -2.0))
        @test isinf(injectivity_radius(M, -2.0, ExponentialRetraction()))
        @test isinf(injectivity_radius(M, ExponentialRetraction()))
        @test project(M, 1.5, 1.0) == 1.0
        @test zero_vector(M, 1.0) == 0.0
        X = similar([1.0])
        zero_vector!(M, X, 1.0)
        @test X == [0.0]
    end
    types = [Float64]
    TEST_FLOAT32 && push!(types, Float32)
    for T in types
        @testset "Type $T" begin
            pts = convert.(Ref(T), [1.0, 4.0, 2.0])
            test_manifold(
                M,
                pts,
                test_forward_diff=false,
                test_reverse_diff=false,
                test_vector_spaces=false,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                test_vee_hat=false,
                is_mutating=false,
            )
        end
    end
    @testset "Power of Positive Numbers" begin
        M2 = PositiveVectors(2)
        for T in types
            pts2 = [convert.(Ref(T), v) for v in [[1.0, 1.1], [3.0, 3.3], [2.0, 2.2]]]
            test_manifold(
                M2,
                pts2,
                test_forward_diff=false,
                test_reverse_diff=false,
                test_vector_spaces=false,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                test_vee_hat=false,
            )
        end
    end
end
