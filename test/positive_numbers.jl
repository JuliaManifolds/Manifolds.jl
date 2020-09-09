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
        @test !is_manifold_point(M, -1.0)
        @test_throws DomainError is_manifold_point(M, -1.0, true)
        @test is_tangent_vector(M, 1.0, 0.0; check_base_point = false)
        @test flat(M, 0.0, FVector(TangentSpace, 1.0)) == FVector(CotangentSpace, 1.0)
        @test sharp(M, 0.0, FVector(CotangentSpace, 1.0)) == FVector(TangentSpace, 1.0)
        @test vector_transport_to(M, 1.0, 3.0, 2.0, ParallelTransport()) == 6.0
        @test retract(M, 1.0, 1.0) == exp(M, 1.0, 1.0)
        @test isinf(injectivity_radius(M))
        @test isinf(injectivity_radius(M, Ref(-2.0)))
        @test isinf(injectivity_radius(M, Ref(-2.0), ExponentialRetraction()))
        @test isinf(injectivity_radius(M, ExponentialRetraction()))
        @test project(M, 1.5, 1.0) == 1.0
    end
    types = [Float64]
    TEST_FLOAT32 && push!(types, Float32)
    for T in types
        @testset "Type $T" begin
            pts = convert.(Ref(T), [1.0, 4.0, 2.0])
            test_manifold(
                M,
                pts,
                test_forward_diff = false,
                test_reverse_diff = false,
                test_vector_spaces = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_default_vector_transport = true,
                test_vee_hat = false,
                is_mutating = false,
            )
        end
    end
end
