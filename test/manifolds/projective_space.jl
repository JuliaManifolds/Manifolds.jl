include("../utils.jl")

@testset "ProjectiveSpace" begin
    @testset "Real" begin
        M = ProjectiveSpace(2)
        @testset "Basics" begin
            @test repr(M) == "ProjectiveSpace(2, ℝ)"
            @test representation_size(M) == (3,)
            @test manifold_dimension(M) == 2
            @test !is_flat(M)
            @test is_flat(ProjectiveSpace(1))
            @test !is_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_vector(M, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            @test_throws DomainError is_point(M, [2.0, 0.0, 0.0], true)
            @test !is_point(M, [2.0, 0.0, 0.0])
            @test !is_vector(M, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            @test_throws DomainError is_vector(M, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], true)
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, [1.0, 0.0, 0.0]) == π / 2
            @test injectivity_radius(M, [1.0, 0.0, 0.0], ExponentialRetraction()) == π / 2
        end
        types = [Vector{Float64}]
        TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

        TEST_FLOAT32 && push!(types, Vector{Float32})
        basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
        @testset "Type $T" for T in types
            x = [1.0, 0.0, 0.0]
            v = [0.0, 1.0, 0.0]
            y = exp(M, x, v)
            w = [0.0, 1.0, -1.0]
            z = exp(M, x, w)
            pts = convert.(T, [x, y, z])
            test_manifold(
                M,
                pts,
                test_injectivity_radius=false,
                test_project_point=true,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                vector_transport_methods=[
                    ParallelTransport(),
                    ProjectionTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                point_distributions=[Manifolds.uniform_distribution(M, pts[1])],
                tvector_distributions=[
                    Manifolds.normal_tvector_distribution(M, pts[1], 1.0),
                ],
                basis_types_vecs=(
                    DiagonalizingOrthonormalBasis([0.0, 1.0, 2.0]),
                    basis_types...,
                ),
                basis_types_to_from=basis_types,
                test_vee_hat=false,
                retraction_methods=[
                    ProjectionRetraction(),
                    PolarRetraction(),
                    QRRetraction(),
                ],
                inverse_retraction_methods=[
                    ProjectionInverseRetraction(),
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                ],
                is_tangent_atol_multiplier=2,
                test_inplace=true,
            )
        end

        @testset "retract/inverse_retract" begin
            x = [1.0, 0.0, 0.0]
            v = [0.0, 1.0, 0.0]
            y = retract(M, x, v, ProjectionRetraction())
            v2 = inverse_retract(M, x, y, ProjectionInverseRetraction())
            @test v ≈ v2
        end

        @testset "equivalence" begin
            x = [1.0, 0.0, 0.0]
            v = [0.0, 1.0, 0.0]
            @test isapprox(M, x, -x)
            @test isapprox(M, x, exp(M, x, π * v))
            @test log(M, x, -x) ≈ zero(v)
            @test isapprox(M, -x, vector_transport_to(M, x, v, -x), -v)
        end

        @testset "Distribution MVector tests" begin
            upd_mvector = Manifolds.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
            @test isa(rand(upd_mvector), MVector)

            gtpd_mvector =
                Manifolds.normal_tvector_distribution(M, (@MVector [1.0, 0.0, 0.0]), 1.0)
            @test isa(rand(gtpd_mvector), MVector)
        end
    end

    @testset "Complex" begin
        M = ProjectiveSpace(2, ℂ)
        @testset "Basics" begin
            @test repr(M) == "ProjectiveSpace(2, ℂ)"
            @test representation_size(M) == (3,)
            @test manifold_dimension(M) == 4
            @test !is_flat(M)
            @test Manifolds.allocation_promotion_function(M, exp!, (1,)) == complex
            @test !is_point(M, [1.0 + 0im, 0.0, 0.0, 0.0])
            @test !is_vector(M, [1.0 + 0im, 0.0, 0.0, 0.0], [0.0 + 0im, 1.0, 0.0])
            @test_throws DomainError is_point(M, [1.0, im, 0.0], true)
            @test !is_point(M, [1.0, im, 0.0])
            @test !is_vector(M, [1.0 + 0im, 0.0, 0.0], [1.0 + 0im, 0.0, 0.0])
            @test !is_vector(M, [1.0 + 0im, 0.0, 0.0], [-0.5im, 0.0, 0.0])
            @test_throws DomainError is_vector(
                M,
                [1.0 + 0im, 0.0, 0.0],
                [1.0 + 0im, 0.0, 0.0],
                true,
            )
            @test_throws DomainError is_vector(
                M,
                [1.0 + 0im, 0.0, 0.0],
                [-0.5im, 0.0, 0.0],
                true,
            )
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, [1.0 + 0im, 0.0, 0.0]) == π / 2
            @test injectivity_radius(M, [1.0 + 0im, 0.0, 0.0], ExponentialRetraction()) ==
                  π / 2
        end
        types = [Vector{ComplexF64}]
        @testset "Type $T" for T in types
            x = [0.5 + 0.5im, 0.5 + 0.5im, 0]
            v = [0.0, 0.0, 1.0 - im]
            y = im * exp(M, x, v)
            w = [0.5, -0.5, 0.5im]
            z = (sqrt(0.5) - sqrt(0.5) * im) * exp(M, x, w)
            pts = convert.(T, [x, y, z])
            test_manifold(
                M,
                pts,
                test_injectivity_radius=false,
                test_project_point=true,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                vector_transport_methods=[
                    ParallelTransport(),
                    ProjectionTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                basis_types_to_from=(DefaultOrthonormalBasis(),),
                test_vee_hat=false,
                retraction_methods=[
                    ProjectionRetraction(),
                    PolarRetraction(),
                    QRRetraction(),
                ],
                inverse_retraction_methods=[
                    ProjectionInverseRetraction(),
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                ],
                is_tangent_atol_multiplier=2,
                exp_log_atol_multiplier=10.0^3,
                retraction_atol_multiplier=10.0,
            )
        end

        @testset "retract/inverse_retract" begin
            x = [0.5 + 0.5im, 0.5 + 0.5im, 0]
            v = [0.0, 0.0, 1.0 - im]
            y = retract(M, x, v, ProjectionRetraction())
            v2 = inverse_retract(M, x, y, ProjectionInverseRetraction())
            @test v ≈ v2
        end

        @testset "equivalence" begin
            x = [1.0 + 0im, 0.0, 0.0]
            v = [0.0, im, 0.0]
            s = sqrt(0.5) - sqrt(0.5) * im
            @test isapprox(M, x, s * x)
            @test isapprox(M, x, exp(M, x, π * v))
            @test log(M, x, s * x) ≈ zero(v)
            @test isapprox(M, s * x, vector_transport_to(M, x, v, s * x), s * v)
        end
    end

    @testset "Right Quaternion" begin
        M = ProjectiveSpace(2, ℍ)
        @testset "Basics" begin
            @test repr(M) == "ProjectiveSpace(2, ℍ)"
            @test representation_size(M) == (3,)
            @test manifold_dimension(M) == 8
            @test !is_flat(M)
            @test !is_point(M, quat([1.0, 0.0, 0.0, 0.0]))
            @test !is_vector(M, quat([1.0, 0.0, 0.0, 0.0]), quat([0.0, 1.0, 0.0]))
            @test_throws DomainError is_point(M, [1.0, quat(0, 1, 0, 0), 0.0], true)
            @test !is_point(M, [1.0, quat(0, 1, 0, 0), 0.0])
            @test !is_vector(M, quat([1.0, 0.0, 0.0]), quat([1.0, 0.0, 0.0]))
            @test !is_vector(M, quat([1.0, 0.0, 0.0]), [quat(0, -0.5, 0, 0), 0.0, 0.0])
            @test_throws DomainError is_vector(
                M,
                Quaternion[1.0, 0.0, 0.0],
                Quaternion[1.0, 0.0, 0.0],
                true,
            )
            @test_throws DomainError is_vector(
                M,
                quat([1.0, 0.0, 0.0]),
                quat([-0.5, 0.0, 0.0]),
                true,
            )
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, Quaternion[1.0, 0.0, 0.0]) == π / 2
            @test injectivity_radius(
                M,
                Quaternion[1.0, 0.0, 0.0],
                ExponentialRetraction(),
            ) == π / 2
        end
        types = [Vector{Quaternion{Float64}}]
        @testset "Type $T" for T in types
            x = [Quaternion(0.5, 0, 0, 0.5), Quaternion(0, 0, 0.5, 0.5), 0]
            v = [Quaternion(0), Quaternion(0), Quaternion(0.0, -0.5, -0.5, 0.0)]
            y = Quaternion(0, 0, 0, 1) * exp(M, x, v)
            w = [
                Quaternion(0.25, -0.25, 0.25, 0.25),
                Quaternion(0.25, 0.25, -0.25, -0.25),
                1,
            ]
            z = Quaternion(0.5, -0.5, 0.5, -0.5) * exp(M, x, w)
            pts = convert.(T, [x, y, z])
            test_manifold(
                M,
                pts,
                test_injectivity_radius=false,
                test_project_point=true,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                vector_transport_methods=[
                    ParallelTransport(),
                    ProjectionTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                basis_types_to_from=(DefaultOrthonormalBasis(),),
                test_vee_hat=false,
                retraction_methods=[
                    ProjectionRetraction(),
                    PolarRetraction(),
                    QRRetraction(),
                ],
                inverse_retraction_methods=[
                    ProjectionInverseRetraction(),
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                ],
                is_tangent_atol_multiplier=10,
                exp_log_atol_multiplier=10.0^3,
                retraction_atol_multiplier=10.0,
            )
        end

        @testset "retract/inverse_retract" begin
            x = [Quaternion(0.5, 0, 0, 0.5), Quaternion(0, 0, 0.5, 0.5), 0]
            v = [Quaternion(0), Quaternion(0), Quaternion(0.0, -0.5, -0.5, 0.0)]
            y = retract(M, x, v, ProjectionRetraction())
            v2 = inverse_retract(M, x, y, ProjectionInverseRetraction())
            @test v ≈ v2
        end

        @testset "equivalence" begin
            x = Quaternion[1.0, 0.0, 0.0]
            v = [0.0, quat(0, 1, 0, 0), 0.0]
            s = Quaternion(0.5, -0.5, 0.5, -0.5)
            @test isapprox(M, x, x * s)
            @test isapprox(M, x, exp(M, x, π * v))
            @test log(M, x, x * s) ≈ zero(v)
            @test isapprox(M, x * s, vector_transport_to(M, x, v, x * s), v * s)
        end
    end

    @testset "ArrayProjectiveSpace" begin
        M = ArrayProjectiveSpace(2, 2; field=ℝ)
        @test manifold_dimension(M) == 3
        @test repr(M) == "ArrayProjectiveSpace(2, 2; field = ℝ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{2,2},ℝ}
        @test representation_size(M) == (2, 2)
        p = ones(2, 2)
        q = project(M, p)
        @test is_point(M, q)
        Y = [1.0 0.0; 0.0 1.1]
        X = project(M, q, Y)
        @test is_vector(M, q, X)

        M = ArrayProjectiveSpace(2, 2; field=ℂ)
        @test manifold_dimension(M) == 6
        @test repr(M) == "ArrayProjectiveSpace(2, 2; field = ℂ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{2,2},ℂ}
        @test representation_size(M) == (2, 2)
    end

    @testset "small distance tests" begin
        @testset for fT in (Float32, Float64), T in (fT, Complex{fT}, Quaternion{fT})
            𝔽 = T isa Complex ? ℂ : (T isa Quaternion ? ℍ : ℝ)
            M = ProjectiveSpace(2, 𝔽)
            rT = real(T)
            atol = rtol = sqrt(eps(rT))
            @testset for t in (zero(rT), eps(rT)^(1 // 4) / 8, eps(rT)^(1 // 4)),
                λ in (one(T), (T <: Real ? -one(T) : sign(randn(T))))

                p = project(M, randn(T, representation_size(M)))
                X = project(M, p, randn(T, representation_size(M)))
                X ./= norm(M, p, X)
                project!(M, X, p, X)
                @test distance(M, p, exp(M, p, t * X) * λ) ≈ t atol = atol rtol = rtol
                @test distance(M, p, exp(M, p, (π - t) * X) * λ) ≈ t atol = atol rtol = rtol
            end
        end
    end

    @testset "Volume" begin
        @test manifold_volume(ProjectiveSpace(0)) ≈ 1
        @test manifold_volume(ProjectiveSpace(1)) ≈ π
        @test manifold_volume(ProjectiveSpace(2)) ≈ 2 * π
        @test manifold_volume(ProjectiveSpace(3)) ≈ π * π
    end
end
