using Manifolds
using Test
using ForwardDiff
using RecursiveArrayTools

@testset "Parametric surface" begin
    R, r = 3.0, 2.0
    M_torus = Manifolds.EmbeddedTorus(R, r)
    M = ParametricSurface(
        Torus(2),
        Euclidean(3),
        (q, p) -> (q .= Manifolds._torus_param(M_torus, p...)),
        (p, q) -> (p .= Manifolds._torus_theta_phi(M_torus, q)),
        function torus_jac(J, p)
            θ, φ = p
            sinθ, cosθ = sincos(θ)
            sinφ, cosφ = sincos(φ)
            J .= [
                -r * sinθ * cosφ -(R + r * cosθ) * sinφ
                -r * sinθ * sinφ (R + r * cosθ) * cosφ
                r * cosθ 0
            ]
        end,
    )
    parameters = [0.5, -1.2]
    p = [Manifolds._torus_param(M_torus, parameters...)...]
    X_parameters = [-1.2, 0.4]
    Y_parameters = [0.3, -0.5]
    A = Manifolds.DefaultTorusAtlas()
    J = zeros(3, 2)
    M.jacobian_f!(J, parameters)
    X = J * X_parameters
    Y = J * Y_parameters

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [manifold_dimension, get_embedding, representation_size, is_point, is_vector],
            :Points => [p], :Vectors => [X, Y],
            :InvalidPoints => [[0.0, 0.0, 0.0]],
            :InvalidVectors => [[1.0, 2.0, 3.0]],
        ),
        # Expectations
        Dict(
            :atol => 1.0e-7,
            manifold_dimension => manifold_dimension(M_torus),
            get_embedding => Euclidean(3),
            representation_size => representation_size(M_torus),
        ),
    )

    torus_inner = inner(M_torus, A, (0.0, 0.0), parameters, X_parameters, Y_parameters)
    @test inner(M, p, X, Y) ≈ torus_inner
    @test norm(M, p, X) ≈ sqrt(inner(M, p, X, X))
    @test project(M, p) ≈ p
    @test project(M, p, [1.0, 2.0, 3.0]) ≈ J * (J \ [1.0, 2.0, 3.0])

    @testset "forwarded atlas" begin
        A_parametric = get_default_atlas(M)
        i = get_chart_index(M, A_parametric, p)
        a = get_parameters(M, A_parametric, i, p)
        Y_parameters = [-0.2, -0.3]
        B = induced_basis(M, A_parametric, i)

        @test get_chart_index(M, A_parametric, i, a) ==
            get_chart_index(M.M_param, A_parametric.A, get_point(M.M_param, A_parametric.A, i, a))
        @test get_chart_index(M, A_parametric, p) ==
            get_chart_index(M.M_param, A_parametric.A, parameters)
        @test get_parameters(M, A_parametric, i, p) ≈ a
        @test get_point(M, A_parametric, i, a) ≈ p
        @test check_chart_switch(M, A_parametric, i, a) ==
            check_chart_switch(M.M_param, A_parametric.A, i, a)

        Y = get_vector(M, p, Y_parameters, B)
        @test get_vector(M, p, X_parameters, B) ≈ X
        @test get_coordinates(M, p, X, B) ≈ X_parameters
        @test inner(M, A_parametric, i, a, X_parameters, Y_parameters) ≈
            inner(M_torus, A, (0.0, 0.0), parameters, X_parameters, Y_parameters)
        @test affine_connection(M, A_parametric, i, a, X_parameters, Y_parameters) ≈
            affine_connection(M_torus, A, (0.0, 0.0), parameters, X_parameters, Y_parameters)
        @test gaussian_curvature(M, p) ≈ gaussian_curvature(M_torus, p)
    end
end
