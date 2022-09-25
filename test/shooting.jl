using Manifolds, Test

@testset "shooting methods" begin
    @testset "shooting" begin
        retractions = [ExponentialRetraction()]
        initial_inverse_retractions = [ProjectionInverseRetraction()]
        vector_transports = [ProjectionTransport()]
        num_transport_points = [2, 4]
        @testset for M in [Sphere(2), Sphere(3, ℂ), Stiefel(5, 3), Stiefel(5, 3, ℂ)]
            T = number_system(M) === ℝ ? Float64 : ComplexF64
            p = project(M, randn(T, representation_size(M)))
            X = project(M, p, randn(T, representation_size(M)))
            X .*= inv(norm(M, p, X))
            q = exp(M, p, X)
            @testset for retraction in retractions
                # check defaults
                if M isa Sphere
                    Y = shooting(M, p, q, retraction)
                    @test isapprox(M, p, Y, X)
                end
                @testset for initial_inverse_retraction in initial_inverse_retractions,
                    vector_transport in vector_transports,
                    ntp in num_transport_points

                    Y = shooting(
                        M,
                        p,
                        q,
                        retraction;
                        initial_inverse_retraction=initial_inverse_retraction,
                        vector_transport=vector_transport,
                        num_transport_points=ntp,
                    )
                    @test isapprox(M, p, Y, X)
                end
            end
        end
    end

    @testset "ShootingInverseRetraction" begin
        retractions = [ExponentialRetraction()]
        initial_inverse_retractions = [ProjectionInverseRetraction()]
        vector_transports = [ProjectionTransport()]
        num_transport_points = [2, 4]
        @testset for M in [Sphere(2), Sphere(3, ℂ), Stiefel(5, 3), Stiefel(5, 3, ℂ)]
            T = number_system(M) === ℝ ? Float64 : ComplexF64
            p = project(M, randn(T, representation_size(M)))
            X = project(M, p, randn(T, representation_size(M)))
            X .*= inv(norm(M, p, X))
            q = exp(M, p, X)
            @testset for retraction in retractions,
                initial_inverse_retraction in initial_inverse_retractions,
                vector_transport in vector_transports,
                ntp in num_transport_points

                inverse_retraction = ShootingInverseRetraction(
                    retraction,
                    initial_inverse_retraction,
                    vector_transport,
                    ntp,
                    1e-10,
                    10_000,
                )
                Y = inverse_retract(M, p, q, inverse_retraction)
                @test isapprox(M, p, Y, X)
            end
        end
    end
end
