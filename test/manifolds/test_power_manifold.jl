using LinearAlgebra, Manifolds, ManifoldsBase, Random, StaticArrays, Test
using Distributions, HybridArrays, RecursiveArrayTools
using StaticArrays: Dynamic
using FillArrays: Fill
using NamedDims: NamedDimsArray

struct TestExponentialAtlas <: Manifolds.AbstractAtlas{ℝ} end

function Manifolds.get_point!(M::AbstractManifold, p, ::TestExponentialAtlas, i, a)
    return exp!(M, p, i, get_vector(M, i, a, DefaultOrthonormalBasis()))
end

function Manifolds.get_parameters!(M::AbstractManifold, a, ::TestExponentialAtlas, i, p)
    return get_coordinates!(M, a, i, log(M, i, p), DefaultOrthonormalBasis())
end

@testset "Power manifold" begin
    Ms = Sphere(2)
    Mr = Rotations(3)

    @testset "Constructors and dimensions" begin
        Ms1 = PowerManifold(Ms, 5)
        @test power_dimensions(Ms1) == (5,)
        @test manifold_dimension(Ms1) == 10
        Ms2 = PowerManifold(Ms, 5, 7)
        @test power_dimensions(Ms2) == (5, 7)
        @test manifold_dimension(Ms2) == 70
        Ms2n = PowerManifold(Ms1, NestedPowerRepresentation(), 7)
        @test power_dimensions(Ms2n) == (7,)
        @test manifold_dimension(Ms2n) == 70

        Mrn1 = PowerManifold(Mr, NestedPowerRepresentation(), 5)
        @test PowerManifold(PowerManifold(Mr, 2), 3) == PowerManifold(Mr, 2, 3)
        @test PowerManifold(Torus(2), 3) isa PowerManifold{ℝ, Torus{2}}
        @test PowerManifold(Mrn1, 3) == PowerManifold(Mr, NestedPowerRepresentation(), 5, 3)
        @test PowerManifold(Mrn1, ArrayPowerRepresentation(), 3) ==
            PowerManifold(Mr, ArrayPowerRepresentation(), 5, 3)
        @test manifold_dimension(PowerManifold(Mr, 5)) == 15
        @test manifold_dimension(Mrn1) == 15
        @test manifold_dimension(PowerManifold(Mr, 5, 7)) == 105
        @test manifold_dimension(PowerManifold(Mr, NestedPowerRepresentation(), 5, 7)) == 105

        @test repr(Mrn1) == "PowerManifold(Rotations(3), NestedPowerRepresentation(), 5)"
        @test sprint(show, "text/plain", PowerManifold(Ms, 5; parameter = :type)) ==
            "PowerManifold(Sphere(2), 5, parameter=:type)"
        @test Manifolds.allocation_promotion_function(Ms, exp, ([1],)) ==
            Manifolds.allocation_promotion_function(Ms1, exp, (1,))
        @test Ms^5 === Oblique(3, 5)
        @test Ms^(5,) === Ms1
        @test Mr^(5, 7) === PowerManifold(Mr, 5, 7)
    end

    # each component of q is e₁ rotated by θᵢ towards e₂
    θ = [0.0, π / 6, π / 4, π / 3, 1.0]
    p = repeat([1.0, 0.0, 0.0], 1, 5)
    q = [cos.(θ'); sin.(θ'); zeros(1, 5)]
    r = repeat([0.0, 0.0, 1.0], 1, 5)
    X = [zeros(1, 5); θ'; zeros(1, 5)] # log(M, p, q)
    c = [1.0, 2.0, 3.0, 4.0, 5.0]
    Y = [zeros(1, 5); ones(1, 5); c']
    B = DefaultOrthonormalBasis()

    @testset "Power of spheres" begin
        M = PowerManifold(Ms, 5)
        @test p[M, 2] == p[:, 2]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [B],
                :Coordinates => [collect(1.0:10.0)],
                :Covectors => [flat(M, p, X)],
                :EmbeddedPoints => [2 .* p],
                :EmbeddedVectors => [X .+ 3 .* p],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method,
                    default_retraction_method, default_vector_transport_method, distance,
                    exp, flat, geodesic, get_basis, get_coordinates, get_vector, get_vectors,
                    injectivity_radius, inner, inverse_retract, is_flat, is_point,
                    is_vector, log, manifold_dimension, manifold_volume, mid_point, norm,
                    parallel_transport_direction, parallel_transport_to, project, rand,
                    repr, representation_size, retract, sectional_curvature, sharp,
                    shortest_geodesic, vector_transport_direction, vector_transport_to,
                    volume_density, Weingarten, zero_vector,
                ],
                # with one invalid component, and of a wrong size
                :InvalidPoints => [[ones(3) p[:, 2:5]], ones(3, 4)],
                :InvalidVectors => [[ones(3) X[:, 2:5]]],
                :InverseRetractionMethods => [
                    LogarithmicInverseRetraction(), ProjectionInverseRetraction(),
                ],
                # multiples of p are normal to each sphere
                :NormalVectors => [2 .* p],
                :Points => [p, q, r],
                :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
                :SecondVector => Y,
                :Vectors => [X, log(M, q, r), log(M, r, p)],
                :VectorTransportMethods => [ParallelTransport(), ProjectionTransport()],
                :Seed => 42,
            ),
            Dict(
                :atol => 1.0e-14,
                :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => StabilizedRetraction(),
                default_vector_transport_method => ParallelTransport(),
                distance => norm(θ),
                exp => q,
                # the coordinates at e₁ are the second and third entries of each component
                (get_coordinates, B) => vec([θ'; zeros(1, 5)]),
                (get_vector, collect(1.0:10.0), B) => [zeros(1, 5); reshape(1.0:10.0, 2, 5)],
                (get_vectors, B) => :Orthonormal,
                injectivity_radius => π,
                (injectivity_radius, ExponentialRetraction()) => π,
                inner => sum(θ),
                (inverse_retract, LogarithmicInverseRetraction()) => X,
                (inverse_retract, ProjectionInverseRetraction()) =>
                    [zeros(1, 5); tan.(θ'); zeros(1, 5)],
                is_flat => false,
                log => X,
                manifold_dimension => 10,
                manifold_volume => (4π)^5,
                mid_point => [cos.(θ' ./ 2); sin.(θ' ./ 2); zeros(1, 5)],
                norm => norm(θ),
                # the velocity of the geodesic in each component is transported to q
                parallel_transport_direction => [-θ' .* sin.(θ'); θ' .* cos.(θ'); zeros(1, 5)],
                parallel_transport_to => [-θ' .* sin.(θ'); θ' .* cos.(θ'); zeros(1, 5)],
                (project, :Point) => p,
                (project, :Vector) => X,
                repr => "PowerManifold(Sphere(2), 5)",
                representation_size => (3, 5),
                (retract, ExponentialRetraction()) => q,
                (retract, ProjectionRetraction()) => [ones(1, 5); θ'; zeros(1, 5)] ./ sqrt.(1 .+ θ' .^ 2),
                # the components of X and Y span planes of curvature one
                sectional_curvature => sum(θ .^ 2 .* c .^ 2) /
                    (sum(θ .^ 2) * sum(1 .+ c .^ 2) - sum(θ)^2),
                sectional_curvature_max => 1.0,
                sectional_curvature_min => 0.0,
                sharp => X,
                (vector_transport_to, ParallelTransport()) =>
                    [-θ' .* sin.(θ'); θ' .* cos.(θ'); zeros(1, 5)],
                (vector_transport_to, ProjectionTransport()) => X .- q .* (θ .* sin.(θ))',
                volume_density => prod(sinc.(θ ./ π)),
                Weingarten => -2 .* X,
                :IsPointErrors => [ComponentManifoldError, DomainError],
                :IsVectorErrors => [ComponentManifoldError],
                :IsVectorBasepointError => ComponentManifoldError,
            ),
        )
    end

    @testset "Power of spheres with two indices, $(T)" for T in (
            Array{Float64, 3}, HybridArray{Tuple{3, Dynamic(), Dynamic()}, Float64, 3},
        )
        M = PowerManifold(Ms, 2, 2)
        to3(A) = convert(T, reshape(A[:, 1:4], 3, 2, 2))
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Functions => [
                    copy, copyto!, distance, exp, inner, inverse_retract, is_point, is_vector,
                    log, manifold_dimension, mid_point, norm, parallel_transport_to, rand,
                    representation_size, retract, zero_vector,
                ],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Points => [to3(p), to3(q), to3(r)],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => to3(Y),
                :Vectors => [to3(X), log(M, to3(q), to3(r)), log(M, to3(r), to3(p))],
            ),
            Dict(
                :atol => 1.0e-14,
                distance => norm(θ[1:4]),
                exp => to3(q),
                inner => sum(θ[1:4]),
                (inverse_retract, LogarithmicInverseRetraction()) => to3(X),
                log => to3(X),
                manifold_dimension => 8,
                mid_point => to3([cos.(θ' ./ 2); sin.(θ' ./ 2); zeros(1, 5)]),
                norm => norm(θ[1:4]),
                parallel_transport_to => to3([-θ' .* sin.(θ'); θ' .* cos.(θ'); zeros(1, 5)]),
                representation_size => (3, 2, 2),
                (retract, ExponentialRetraction()) => to3(q),
            ),
        )
    end

    # each component of qr is a rotation about the z-axis by the angle φᵢ
    φ = [π / 6, π / 4, π / 3]
    Rz(φ) = [cos(φ) -sin(φ) 0.0; sin(φ) cos(φ) 0.0; 0.0 0.0 1.0]
    Rx(φ) = [1.0 0.0 0.0; 0.0 cos(φ) -sin(φ); 0.0 sin(φ) cos(φ)]
    Ωz = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
    Ωx = [0.0 0.0 0.0; 0.0 0.0 -1.0; 0.0 1.0 0.0]
    pr = [Matrix(1.0I, 3, 3) for _ in 1:3]
    qr = Rz.(φ)
    rr = Rx.(φ)
    # tangent vectors are represented in the Lie algebra
    Xr = [φᵢ * Ωz for φᵢ in φ]
    Yr = [Ωz + Ωx for _ in 1:3]
    rotation_expectations = Dict(
        :atol => 1.0e-14,
        :atols => Dict(geodesic => 1.0e-9),
        default_inverse_retraction_method => LogarithmicInverseRetraction(),
        default_retraction_method => ExponentialRetraction(),
        default_vector_transport_method => ParallelTransport(),
        distance => sqrt(2) * norm(φ),
        exp => qr,
        injectivity_radius => π * sqrt(2),
        inner => 2 * sum(φ),
        (inverse_retract, LogarithmicInverseRetraction()) => Xr,
        is_flat => false,
        log => Xr,
        manifold_dimension => 9,
        manifold_volume => manifold_volume(Mr)^3,
        mid_point => Rz.(φ ./ 2),
        norm => sqrt(2) * norm(φ),
        # the velocity of a geodesic in the Lie algebra is constant
        parallel_transport_to => Xr,
        (retract, ExponentialRetraction()) => qr,
        sectional_curvature_max => 1 / 8,
        sectional_curvature_min => 0.0,
        (vector_transport_to, ParallelTransport()) => Xr,
        :IsPointErrors => [ComponentManifoldError],
        :IsVectorErrors => [ComponentManifoldError],
    )
    rotation_functions = [
        copy, copyto!, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, exp, flat, geodesic, injectivity_radius,
        inner, inverse_retract, is_flat, is_point, is_vector, log, manifold_dimension,
        manifold_volume, mid_point, norm, parallel_transport_to, rand, retract, sharp,
        vector_transport_to, zero_vector,
    ]

    @testset "Power of rotations, $(repr(R))" for (R, P, V) in (
            (NestedPowerRepresentation(), identity, identity),
            # the rotations are stacked along the third dimension
            (ArrayPowerRepresentation(), A -> cat(A...; dims = 3), A -> cat(A...; dims = 3)),
            (NestedReplacingPowerRepresentation(), A -> SMatrix{3, 3}.(A), A -> SMatrix{3, 3}.(A)),
        )
        M = PowerManifold(Mr, R, 3)
        Bsvd = ProjectedOrthonormalBasis(:svd)
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [Bsvd],
                :Covectors => [flat(M, P(pr), V(Xr))],
                :Functions => [rotation_functions; get_basis; get_vectors],
                :InvalidPoints => [P([2 .* pr[1], pr[2], pr[3]])],
                :InvalidVectors => [V([Matrix(1.0I, 3, 3), Xr[2], Xr[3]])],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Mutating => !(R isa NestedReplacingPowerRepresentation),
                # rand! replaces the components of the vector of static matrices
                :TestRandReproducibility => true,
                :Points => [P(pr), P(qr), P(rr)],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => V(Yr),
                :Vectors => [V(Xr), log(M, P(qr), P(rr)), log(M, P(rr), P(pr))],
                :VectorTransportMethods => [ParallelTransport()],
                :Seed => 42,
            ),
            merge(
                rotation_expectations,
                Dict(
                    exp => P(qr),
                    (inverse_retract, LogarithmicInverseRetraction()) => V(Xr),
                    log => V(Xr),
                    mid_point => P(Rz.(φ ./ 2)),
                    parallel_transport_to => V(Xr),
                    (retract, ExponentialRetraction()) => P(qr),
                    (get_vectors, Bsvd) => :Orthonormal,
                    sharp => V(Xr),
                    (vector_transport_to, ParallelTransport()) => V(Xr),
                ),
            ),
        )
    end

    @testset "Power of circles" begin
        M = PowerManifold(Circle(), 3)
        pc = [0.0, 1.0, 2.0]
        qc = [1.0, 1.0, 2.4]
        rc = [0.0, 2.0, 1.0]
        Xc = qc - pc
        Yc = [1.0, 2.0, 3.0]
        @test qc[M, 3] == 2.4
        @test Manifolds._read(M, (), HybridVector{3}(pc), 3) == 2.0
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [B],
                :Coordinates => [[1.0, 2.0, 3.0]],
                :Covectors => [flat(M, pc, Xc)],
                :Functions => [
                    copy, copyto!, distance, exp, flat, geodesic, get_coordinates,
                    get_vector, injectivity_radius, inner, inverse_retract, is_flat,
                    is_point, is_vector, log, manifold_dimension, manifold_volume,
                    mid_point, norm, parallel_transport_to, rand, repr, representation_size,
                    retract, sharp, shortest_geodesic, zero_vector,
                ],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Points => [pc, qc, rc],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => Yc,
                :Vectors => [Xc, log(M, qc, rc), log(M, rc, pc)],
                :Seed => 42,
            ),
            Dict(
                :atol => 1.0e-14,
                :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
                distance => norm(Xc),
                exp => qc,
                (get_coordinates, B) => Xc,
                (get_vector, [1.0, 2.0, 3.0], B) => [1.0, 2.0, 3.0],
                injectivity_radius => π,
                inner => dot(Xc, Yc),
                (inverse_retract, LogarithmicInverseRetraction()) => Xc,
                is_flat => true,
                log => Xc,
                manifold_dimension => 3,
                manifold_volume => (2π)^3,
                mid_point => (pc + qc) / 2,
                norm => norm(Xc),
                parallel_transport_to => Xc,
                repr => "PowerManifold(Circle(ℝ), 3)",
                representation_size => (3,),
                (retract, ExponentialRetraction()) => qc,
                sharp => Xc,
            ),
        )
    end

    @testset "get_component, set_component!, getindex and setindex!" begin
        Ms1 = PowerManifold(Ms, 5)
        p1 = reshape(collect(1.0:15.0), 3, 5)
        @test get_component(Ms1, p1, 1) == [1.0, 2.0, 3.0]
        @test p1[Ms1, 1] == [1.0, 2.0, 3.0]
        @test p1[Ms1, 1] isa Vector
        p2 = [10.0, 11.0, 12.0]
        set_component!(Ms1, p1, p2, 2)
        @test get_component(Ms1, p1, 2) == p2
        p1[Ms1, 2] = 2 * p2
        @test p1[Ms1, 2] == 2 * p2
        p1[Ms1, 2] += p2
        @test p1[Ms1, 2] == 3 * p2
        p1[Ms1, 2] .+= p2
        @test p1[Ms1, 2] == 4 * p2
        @test view(p1, Ms1, 1) == p1[Ms1, 1]
        @test view(p1, Ms1, 1) isa SubArray
        pv = VectorOfArray([q[:, i] for i in 1:5])
        @test pv[Ms1, 5] == q[:, 5]
        @test view(pv, Ms1, 5) == q[:, 5]
        Me5 = PowerManifold(Euclidean(), 5)
        pvs = VectorOfArray([1.0, 2.0, 3.0, 4.0, 5.0])
        @test pvs[Me5, 2] == 2.0
        @test view(pvs, Me5, 2)[] == 2.0

        Msn1 = PowerManifold(Ms, NestedPowerRepresentation(), 5)
        p_n1 = [collect(1.0:3.0) .+ 3k for k in 0:4]
        @test get_component(Msn1, p_n1, 1) == [1.0, 2.0, 3.0]
        @test p_n1[Msn1, 1] == [1.0, 2.0, 3.0]
        @test p_n1[Msn1, 1] isa Vector
        set_component!(Msn1, p_n1, p2, 2)
        @test get_component(Msn1, p_n1, 2) == p2
        p_n1[Msn1, 2] = 2 * p2
        @test p_n1[Msn1, 2] == 2 * p2
        p_n1[Msn1, 2] += p2
        @test p_n1[Msn1, 2] == 3 * p2
        p_n1[Msn1, 2] .+= p2
        @test p_n1[Msn1, 2] == 4 * p2
        @test view(p_n1, Msn1, 1) == p_n1[Msn1, 1]
        @test view(p_n1, Msn1, 1) isa SubArray
    end

    @testset "getindex and view of arrays from package extensions" begin
        Ms1 = PowerManifold(Ms, 5)
        p1 = reshape(collect(1.0:15.0), 3, 5)
        v1 = p1[1, :]
        Me5 = PowerManifold(Euclidean(), 5)
        @test NamedDimsArray{(:a, :b)}(p1)[Ms1, 1] == p1[:, 1]
        @test NamedDimsArray{(:a,)}(v1)[Me5, 2] == v1[2]
        @test view(NamedDimsArray{(:a, :b)}(p1), Ms1, 1) == p1[:, 1]
        @test view(Fill(1.0, 3, 5), Ms1, 1) == ones(3)
    end

    @testset "Errors in a single component" begin
        M = PowerManifold(Ms, NestedPowerRepresentation(), 2)
        p_n = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        Xn = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        @test_throws ComponentManifoldError is_point(M, Xn; error = :error)
        @test_throws ComponentManifoldError is_vector(M, p_n, Xn; error = :error)
    end

    @testset "Basis printing" begin
        Ms1 = PowerManifold(Ms, 5)
        Bc = get_basis(Ms1, p, DefaultOrthonormalBasis())
        @test sprint(show, "text/plain", Bc) == """
            DefaultOrthonormalBasis{ℝ, TangentSpaceType} for a power manifold
            Basis for component (1,):
            $(sprint(show, "text/plain", Bc.data.bases[1]))
            Basis for component (2,):
            $(sprint(show, "text/plain", Bc.data.bases[2]))
            Basis for component (3,):
            $(sprint(show, "text/plain", Bc.data.bases[3]))
            Basis for component (4,):
            $(sprint(show, "text/plain", Bc.data.bases[4]))
            Basis for component (5,):
            $(sprint(show, "text/plain", Bc.data.bases[5]))
            """
    end

    @testset "Atlases" begin
        M = PowerManifold(Euclidean(2), NestedPowerRepresentation(), 2)
        pe = [zeros(2), ones(2)]
        A = RetractionAtlas()
        a = get_parameters(M, A, pe, pe)
        @test get_point(M, A, pe, a) == pe
        @test get_parameters(M, TestExponentialAtlas(), pe, pe) ≈ a

        Ms1 = PowerManifold(Ms, 5)
        A = get_default_atlas(Ms1)
        @test A isa PowerAtlas
        @test A.atlas == get_default_atlas(Ms)
        pa = [1.0 0.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0 0.0; 0.0 0.0 1.0 1.0 1.0]
        i = get_chart_index(Ms1, A, pa)
        @test i == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        a = get_parameters(Ms1, A, i, pa)
        @test length(a) == manifold_dimension(Ms1)
        @test isapprox(Ms1, get_point(Ms1, A, i, a), pa)

        Mn = PowerManifold(Ms, NestedReplacingPowerRepresentation(), 2)
        An = get_default_atlas(Mn)
        p_n = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        i_n = get_chart_index(Mn, An, p_n)
        @test isapprox(Mn, get_point(Mn, An, i_n, get_parameters(Mn, An, i_n, p_n)), p_n)
    end

    @testset "Induced bases, $(repr(R))" for (R, P) in (
            (ArrayPowerRepresentation(), identity),
            (NestedPowerRepresentation(), A -> [A[:, i] for i in 1:2]),
            (NestedReplacingPowerRepresentation(), A -> [SVector{3}(A[:, i]) for i in 1:2]),
        )
        M = PowerManifold(Sphere(2), R, 2)
        A = PowerAtlas(StereographicAtlas())
        p_c = P(q[:, 2:3])
        Xi = P(log(PowerManifold(Ms, 2), q[:, 2:3], r[:, 2:3]))
        i_p = get_chart_index(M, A, p_c)
        @test i_p == [:north, :north]
        Bi = induced_basis(M, A, i_p)
        ci = get_coordinates(M, p_c, Xi, Bi)
        @test isapprox(M, p_c, get_vector(M, p_c, ci, Bi), Xi)
        Z = allocate(M, Xi)
        Z = get_vector!(M, Z, p_c, ci, Bi)
        @test isapprox(M, p_c, Z, Xi)
    end

    @testset "Chart indices with two indices" begin
        M22 = PowerManifold(Sphere(2), 2, 2)
        A = PowerAtlas(StereographicAtlas())
        p22 = [0.53 -0.72; -0.59 -0.68; -0.59 -0.07;;; -0.5 -0.24; 0.34 0.54; 0.79 -0.8]
        i_p = get_chart_index(M22, A, p22)
        @test get_chart_index(M22, A, i_p, fill(3.0, 8)) == Any[:south :north; :north :north]
    end

    @testset "Componentwise conversions" begin
        Mspd = SymmetricPositiveDefinite(3)
        N = PowerManifold(Mspd, NestedPowerRepresentation(), 2)
        e = EuclideanMetric()
        P = [Matrix(1.0I, 3, 3), Matrix(Diagonal([2.0, 2.0, 1.0]))]
        XP = [log(Mspd, P[1], P[2]), log(Mspd, P[2], P[1])]
        @test change_metric(N, e, P, XP) ≈ change_metric.(Ref(Mspd), Ref(e), P, XP)
        @test change_representer(N, e, P, XP) ≈ change_representer.(Ref(Mspd), Ref(e), P, XP)

        N2 = PowerManifold(Ms, 2)
        ph = [1.0 0.0; 0.0 0.0; 0.0 1.0]
        Xh = [0.0 1.0; 1.0 1.0; 1.0 0.0]
        Yh = [0.0 1.0; 1.0 0.0; 0.0 0.0]
        Zh = [0.5 2.0; 2.0 1.0; 0.0 3.0]
        @test riemannian_Hessian(N2, ph, Yh, Zh, Xh) ==
            reduce(hcat, [riemannian_Hessian(Ms, ph[:, i], Yh[:, i], Zh[:, i], Xh[:, i]) for i in 1:2])
        Vh = [0.2 0.0; 0.0 0.0; 0.0 0.3]
        @test Weingarten(N2, ph, Xh, Vh) ==
            reduce(hcat, [Weingarten(Ms, ph[:, i], Xh[:, i], Vh[:, i]) for i in 1:2])
        # the densities of the components multiply
        Msn = PowerManifold(Ms, NestedPowerRepresentation(), 5)
        @test volume_density(Msn, [p[:, i] for i in 1:5], [X[:, i] for i in 1:5]) ≈
            prod(sinc.(θ ./ π))
    end

    @testset "Nested replacing allocation" begin
        M = PowerManifold(Ms, NestedReplacingPowerRepresentation(), 2)
        @test allocate(M, [SVector{3}(1.0, 0.0, 0.0), SVector{3}(1.0, 0.0, 0.0)]) isa
            Vector{SVector{3, Float64}}
        PSE2 = PowerManifold(Euclidean(2) × Rotations(2), NestedReplacingPowerRepresentation(), 2)
        pse = ArrayPartition(SA[1.0, 2.0], SA[cos(1.0) -sin(1.0); sin(1.0) cos(1.0)])
        @test allocate(PSE2, [pse, pse]) isa
            Vector{ArrayPartition{Float64, Tuple{SVector{2, Float64}, SMatrix{2, 2, Float64, 4}}}}
        pse_ap = ArrayPartition([1.0, 2.0], [cos(1.0) -sin(1.0); sin(1.0) cos(1.0)])
        @test allocate(PSE2, [pse_ap, pse_ap]) isa
            Vector{ArrayPartition{Float64, Tuple{Vector{Float64}, Matrix{Float64}}}}
    end

    @testset "Distributions" begin
        ext = Base.get_extension(Manifolds, :ManifoldsDistributionsExt)
        Ms1 = PowerManifold(Ms, 5)
        sphere_dist = Manifolds.uniform_distribution(Ms, @SVector [1.0, 0.0, 0.0])
        pd = ext.PowerPointDistribution(Ms1, sphere_dist, zeros(3, 5))
        @test Distributions.support(pd) isa ext.MPointSupport
        pp = rand(MersenneTwister(42), pd)
        @test is_point(Ms1, pp)
        sphere_tv_dist = Manifolds.normal_tvector_distribution(Ms, (@MVector [1.0, 0.0, 0.0]), 1.0)
        vd = ext.PowerFVectorDistribution(TangentSpace(Ms1, pp), sphere_tv_dist)
        @test Distributions.support(vd) isa ext.FVectorSupport
        @test is_vector(Ms1, pp, rand(MersenneTwister(42), vd); atol = 1.0e-14)

        Mrn = PowerManifold(Mr, NestedPowerRepresentation(), 3)
        id_rot = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        rd = ext.PowerPointDistribution(
            Mrn, Manifolds.normal_rotation_distribution(Mr, id_rot, 1.0), [zeros(3, 3) for _ in 1:3]
        )
        pr2 = rand(MersenneTwister(42), rd)
        @test is_point(Mrn, pr2)
        rvd = ext.PowerFVectorDistribution(
            TangentSpace(Mrn, pr2), Manifolds.normal_tvector_distribution(Mr, MMatrix(id_rot), 1.0)
        )
        @test is_vector(Mrn, pr2, rand(MersenneTwister(42), rvd); atol = 1.0e-14)
    end
end

@testset "Riemannian gradient on a power of hyperboloids" begin
    M = PowerManifold(Hyperbolic(2), 2)
    p = [[1.0, 1.0, sqrt(3)] [0.0, 0.0, 1.0]]
    G = [[0.4, -0.7, 1.1] [0.2, 0.1, -0.3]]
    Y = riemannian_gradient(M, p, G)
    @test is_vector(M, p, Y)
    for V in (
            project(M, p, [[1.0, 0.0, 0.0] [0.0, 1.0, 0.0]]),
            project(M, p, [[0.0, 1.0, 0.0] [1.0, 0.0, 0.0]]),
        )
        @test inner(M, p, Y, V) ≈ dot(G, V)
    end
    N = PowerManifold(Hyperbolic(2), NestedPowerRepresentation(), 2)
    @test riemannian_gradient(N, [p[:, 1], p[:, 2]], [G[:, 1], G[:, 2]]) ≈ [Y[:, 1], Y[:, 2]]
end
