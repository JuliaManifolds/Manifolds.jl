include("../utils.jl")
include("group_utils.jl")

using Manifolds: connection

@testset "Cartan-Schouten connections" begin
    SO3 = SpecialOrthogonal(3)
    SO3minus = ConnectionManifold(SO3, CartanSchoutenMinus())
    SO3plus = ConnectionManifold(SO3, CartanSchoutenPlus())
    SO3zero = ConnectionManifold(SO3, CartanSchoutenZero())

    e = Matrix{Float64}(I, 3, 3)
    p = exp(hat(SO3, Identity(SO3), [1.0, 2.0, 3.0]))
    q = exp(hat(SO3, Identity(SO3), [3.0, 4.0, 1.0]))
    X = hat(SO3, Identity(SO3), [2.0, 3.0, 4.0])
    SO3e = Identity(SO3)

    @testset "connection" begin
        @test connection(SO3minus) === CartanSchoutenMinus()
        @test connection(SO3plus) === CartanSchoutenPlus()
        @test connection(SO3zero) === CartanSchoutenZero()
    end

    @testset "log/exp" begin
        for CSO3 in [SO3minus, SO3plus, SO3zero]
            @test isapprox(SO3, exp(CSO3, p, X), exp(SO3, p, X))
            @test isapprox(SO3, p, log(CSO3, p, q), log(SO3, p, q); atol=1e-6)
        end
    end

    @testset "Parallel transport" begin
        Y = similar(X)
        @test isapprox(SO3, q, X, vector_transport_to(SO3minus, SO3e, X, q))
        @test isapprox(SO3, q, X, vector_transport_to!(SO3minus, Y, SO3e, X, q))
        @test isapprox(SO3, q, q * X / q, vector_transport_to(SO3plus, SO3e, X, q))
        @test isapprox(SO3, q, q * X / q, vector_transport_to!(SO3plus, Y, SO3e, X, q))
        @test isapprox(
            SO3,
            q,
            vector_transport_to(SO3, e, X, q),
            vector_transport_to(SO3zero, SO3e, X, q),
        )
        @test isapprox(
            SO3,
            q,
            vector_transport_to(SO3, e, X, q),
            vector_transport_to!(SO3zero, Y, SO3e, X, q),
        )
    end
end
