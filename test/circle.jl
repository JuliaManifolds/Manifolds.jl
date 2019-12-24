include("utils.jl")

@testset "Circle" begin
    M = Circle()
    @testset "Real Circle Basics" begin
        @test representation_size(M) == (1,)
        @test manifold_dimension(M) == 1
        @test !is_manifold_point(M, 9.)
        @test_throws DomainError is_manifold_point(M, 9., true)
        @test !is_tangent_vector(M, 9., 0.)
        @test_throws DomainError is_tangent_vector(M, 9., 0., true)
    end
    types = [Float64, Float32]
    for T in types
        @testset "Type $T" begin
            pts = convert.(Ref(T), [-π/4,0.,π/4])
            test_manifold(
                M,
                pts,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
            )
        end
    end
    Mc = Circle(ℂ)
    @testset "Complex Circle Basics" begin
        @test representation_size(Mc) == (1,)
        @test manifold_dimension(Mc) == 1
        @test is_tangent_vector(Mc, 1im, 0.)
        @test is_manifold_point(Mc, 1im)
        @test !is_manifold_point(Mc, 1+1im)
        @test_throws DomainError is_manifold_point(Mc, 1+1im, true)
        @test !is_tangent_vector(Mc, 1+1im, 0.)
        @test_throws DomainError is_tangent_vector(Mc, 1+1im, 0., true)
        @test !is_tangent_vector(Mc, 1im, 2im)
        @test_throws DomainError is_tangent_vector(Mc, 1im, 2im, true)
    end
    types = [Complex{Float64}, Complex{Float32}]
    for T in types
        @testset "Type $T" begin
            pts = convert.(Ref(T), [sqrt(2.0)-sqrt(2.0)im, 0+0im, sqrt(2.0)+sqrt(2.0)im])
            test_manifold(
                Mc,
                pts,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
            )
        end
    end
end
