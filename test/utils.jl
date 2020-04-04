const TEST_FLOAT32 = false
const TEST_DOUBLE64 = false
const TEST_STATIC_SIZED = false

using Manifolds
using ManifoldsBase

using LinearAlgebra
using Distributions
using DoubleFloats
using ForwardDiff
using Random
using ReverseDiff
using StaticArrays
using Test
using LightGraphs
using SimpleWeightedGraphs

find_eps(x::Type{TN}) where TN<:Number = eps(real(TN))
find_eps(x) = find_eps(number_eltype(x))
find_eps(x...) = find_eps(Base.promote_type(map(number_eltype, x)...))

"""
    test_manifold(m::Manifold, pts::AbstractVector;
    args
    )

Tests general properties of manifold `m`, given at least three different points
that lie on it (contained in `pts`).

# Arguments
- `default_inverse_retraction_method = ManifoldsBase.LogarithmicInverseRetraction()` - default method for inverse retractions ([`log`](@ref))
- `default_retraction_method = ManifoldsBase.ExponentialRetraction()` - default method for retractions ([`exp`](@ref))
- `exp_log_atol_multiplier = 0`, change absolute tolerance of exp/log tests (0 use default, i.e. deactivate atol and use rtol)
- `exp_log_rtol_multiplier = 1`, change the relative tolerance of exp/log tests (1 use default). This is deactivated if the `exp_log_atol_multiplier` is nonzero.
- `retraction_atol_multiplier = 0`, change absolute tolerance of (inverse) retraction tests (0 use default, i.e. deactivate atol and use rtol)
- `retraction_rtol_multiplier = 1`, change the relative tolerance of (inverse) retraction tests (1 use default). This is deactivated if the `exp_log_atol_multiplier` is nonzero.
- `inverse_retraction_methods = []`: inverse retraction methods that will be tested.
- `point_distributions = []` : point distributions to test
- `projection_tvector_atol_multiplier = 0` : chage absolute tolerance in testing projections (0 use default, i.e. deactivate atol and use rtol)
-  tvector_distributions = []` : tangent vector distributions to test
- `basis_types = ()` : basis types that will be tested
- `rand_tvector_atol_multiplier = 0` : chage absolute tolerance in testing random vectors (0 use default, i.e. deactivate atol and use rtol)
  random tangent vectors are tangent vectors
- `retraction_methods = []`: retraction methods that will be tested.
- `test_forward_diff = true`: if true, automatic differentiation using
  ForwardDiff is tested.
- `test_reverse_diff = true`: if true, automatic differentiation using
  ReverseDiff is tested.
- `test_musical_isomorphisms = false` : test musical isomorphisms
- `test_mutating_rand = false` : test the mutating random function for points on manifolds
- `test_project_tangent = false` : test projections on tangent spaces
- `test_representation_size = true` : test repersentation size of points/tvectprs
- `test_tangent_vector_broadcasting = true` : test boradcasting operators on TangentSpace
- `test_vector_transport = false` : test vector transport
- `test_vector_spaces = true` : test Vector bundle of this manifold
"""
function test_manifold(M::Manifold, pts::AbstractVector;
    test_exp_log = true,
    test_is_tangent = true,
    test_injectivity_radius=true,
    test_forward_diff = true,
    test_reverse_diff = true,
    test_tangent_vector_broadcasting = true,
    test_project_tangent = false,
    test_project_point = false,
    test_representation_size = true,
    test_musical_isomorphisms = false,
    test_vector_transport = false,
    test_mutating_rand = false,
    test_vector_spaces = true,
    test_vee_hat = false,
    is_mutating = true,
    default_inverse_retraction_method = ManifoldsBase.LogarithmicInverseRetraction(),
    default_retraction_method = ManifoldsBase.ExponentialRetraction(),
    retraction_methods = [],
    inverse_retraction_methods = [],
    point_distributions = [],
    tvector_distributions = [],
    basis_types_vecs = (),
    basis_types_to_from = (),
    basis_has_specialized_diagonalizing_get = false,
    exp_log_atol_multiplier = 0,
    exp_log_rtol_multiplier = 1,
    retraction_atol_multiplier = 0,
    retraction_rtol_multiplier = 1,
    projection_atol_multiplier = 0,
    rand_tvector_atol_multiplier = 0,
    is_tangent_atol_multiplier=1,
)

    length(pts) ≥ 3 || error("Not enough points (at least three expected)")
    isapprox(M, pts[1], pts[2]) && error("Points 1 and 2 are equal")
    isapprox(M, pts[1], pts[3]) && error("Points 1 and 3 are equal")

    # get a default tangent vector for every of the three tangent spaces
    n = length(pts)
    if default_inverse_retraction_method === nothing
        tv = [ zero_tangent_vector(M,pts[i]) for i=1:n ] # no other available
    else
        tv = [ inverse_retract(M,pts[i],pts[((i+1)%n)+1],default_inverse_retraction_method) for i=1:n]
    end
    @testset "dimension" begin
        @test isa(manifold_dimension(M), Integer)
        @test manifold_dimension(M) ≥ 0
        @test manifold_dimension(M) == vector_space_dimension(Manifolds.VectorBundleFibers(Manifolds.TangentSpace, M))
        @test manifold_dimension(M) == vector_space_dimension(Manifolds.VectorBundleFibers(Manifolds.CotangentSpace, M))
    end

    test_representation_size && @testset "representation" begin
        function test_repr(repr)
            @test isa(repr, Tuple)
            for rs ∈ repr
                @test rs > 0
            end
        end

        test_repr(Manifolds.representation_size(M))
        for fiber ∈ (Manifolds.TangentSpace, Manifolds.CotangentSpace)
            test_repr(Manifolds.representation_size(Manifolds.VectorBundleFibers(fiber, M)))
        end
    end

    test_injectivity_radius && @testset "injectivity radius" begin
        @test injectivity_radius(M, pts[1]) > 0
        @test injectivity_radius(M, pts[1]) ≥ injectivity_radius(M)
        for rm ∈ retraction_methods
            @test injectivity_radius(M, rm) > 0
            @test injectivity_radius(M, pts[1], rm) ≥ injectivity_radius(M, rm)
            @test injectivity_radius(M, pts[1], rm) ≤ injectivity_radius(M, pts[1])
        end
    end

    @testset "is_manifold_point" begin
        for pt ∈ pts
            @test is_manifold_point(M, pt)
            @test check_manifold_point(M, pt) === nothing
            @test check_manifold_point(M, pt) === nothing
        end
    end

    test_is_tangent && @testset "is_tangent_vector" begin
        for (p, X) in zip(pts, tv)
            atol = is_tangent_atol_multiplier * find_eps(p)
            if !( check_tangent_vector(M, p, X; atol = atol) === nothing )
                print( check_tangent_vector(M, p, X; atol = atol) )
            end
            @test is_tangent_vector(M, p, X; atol = atol)
            @test check_tangent_vector(M, p, X; atol = atol) === nothing
        end
    end

    test_exp_log && @testset "log/exp tests" begin
        epsp1p2 = find_eps(pts[1], pts[2])
        atolp1p2 = exp_log_atol_multiplier * epsp1p2
        rtolp1p2 = exp_log_atol_multiplier == 0. ? sqrt(epsp1p2)*exp_log_rtol_multiplier : 0
        X1 = log(M, pts[1], pts[2])
        X2 = log(M, pts[2], pts[1])
        @test isapprox(M, pts[2], exp(M, pts[1], X1); atol = atolp1p2, rtol = rtolp1p2)
        @test isapprox(M, pts[1], exp(M, pts[1], X1, 0); atol = atolp1p2, rtol = rtolp1p2)
        @test isapprox(M, pts[2], exp(M, pts[1], X1, 1); atol = atolp1p2, rtol = rtolp1p2)
        @test isapprox(M, pts[1], exp(M, pts[2], X2); atol = atolp1p2, rtol = rtolp1p2)
        @test is_manifold_point(M, exp(M, pts[1], X1); atol = atolp1p2, rtol = rtolp1p2)
        @test isapprox(M, pts[1], exp(M, pts[1], X1, 0); atol = atolp1p2, rtol = rtolp1p2)
        for p ∈ pts
            epsx = find_eps(p)
            @test isapprox(M, p, zero_tangent_vector(M, p), log(M, p, p);
                atol = epsx * exp_log_atol_multiplier,
                rtol = exp_log_atol_multiplier == 0. ? sqrt(epsx)*exp_log_rtol_multiplier : 0
            )
            @test isapprox(M, p, zero_tangent_vector(M, p), inverse_retract(M, p, p);
                atol = epsx * exp_log_atol_multiplier,
                rtol = exp_log_atol_multiplier == 0. ? sqrt(epsx)*exp_log_rtol_multiplier : 0.
            )
        end
        atolp1 = exp_log_atol_multiplier * find_eps(pts[1])
        if is_mutating
            zero_tangent_vector!(M, X1, pts[1])
        else
            X1 = zero_tangent_vector(M, pts[1])
        end
        @test isapprox(M, pts[1], X1, zero_tangent_vector(M, pts[1]); atol = atolp1)
        if is_mutating
            log!(M, X1, pts[1], pts[2])
        else
            X1 = log(M,pts[1], pts[2])
        end

        @test isapprox(M, exp(M, pts[1], X1, 1), pts[2]; atol = atolp1)
        @test isapprox(M, exp(M, pts[1], X1, 0), pts[1]; atol = atolp1)

        @test distance(M, pts[1], pts[2]) ≈ norm(M, pts[1], X1)

        X3 = log(M, pts[1], pts[3])

        @test real(inner(M, pts[1], X1, X3)) ≈ real(inner(M, pts[1], X3, X1))
        @test imag(inner(M, pts[1], X1, X3)) ≈ -imag(inner(M, pts[1], X3, X1))
        @test imag(inner(M, pts[1], X1, X1)) ≈ 0

        @test norm(M, pts[1], X1) isa Real
        @test norm(M, pts[1], X1) ≈ sqrt(inner(M, pts[1], X1, X1))
    end

    @testset "(inverse &) retraction tests" begin
        for (p, X) in zip(pts, tv)
            epsx = find_eps(p)
            for retr_method ∈ retraction_methods
                @test is_manifold_point(M, retract(M, p, X, retr_method))
                @test isapprox(M, p, retract(M, p, X, 0, retr_method);
                    atol = epsx * retraction_atol_multiplier,
                    rtol = retraction_atol_multiplier == 0 ? sqrt(epsx)*retraction_rtol_multiplier : 0
                )
                if is_mutating
                    new_pt = allocate(p)
                    retract!(M, new_pt, p, X, retr_method)
                else
                    new_pt = retract(M, p, X, retr_method)
                end
                @test is_manifold_point(M, new_pt)
            end
        end
        for p ∈ pts
            epsx = find_eps(p)
            for inv_retr_method ∈ inverse_retraction_methods
                @test isapprox(M, p, zero_tangent_vector(M, p), inverse_retract(M, p, p, inv_retr_method);
                    atol = epsx * retraction_atol_multiplier,
                    rtol = retraction_atol_multiplier == 0 ? sqrt(epsx)*retraction_rtol_multiplier : 0
                )
            end
        end
    end

    test_vector_spaces && @testset "vector spaces tests" begin
        for p ∈ pts
            X = zero_tangent_vector(M, p)
            mts = Manifolds.VectorBundleFibers(Manifolds.TangentSpace, M)
            @test isapprox(M, p, X, zero_vector(mts, p))
            zero_vector!(mts, X, p)
            @test isapprox(M, p, X, zero_tangent_vector(M,p))
        end
    end

    @testset "basic linear algebra in tangent space" begin
        for (p, X) in zip(pts, tv)
            @test isapprox(M, p, 0*X, zero_tangent_vector(M, p); atol = find_eps(pts[1]))
            @test isapprox(M, p, 2*X, X+X)
            @test isapprox(M, p, 0*X, X-X)
            @test isapprox(M, p, (-1)*X, -X)
        end
    end

    test_tangent_vector_broadcasting && @testset "broadcasted linear algebra in tangent space" begin
        for (p, X) in zip(pts, tv)
            @test isapprox(M, p, 3*X, 2 .* X .+ X)
            @test isapprox(M, p, -X, X .- 2 .* X)
            @test isapprox(M, p, -X, .-X)
            if (isa(X, AbstractArray))
                Y = allocate(X)
                Y .= 2 .* X .+ X
            else
                Y = 2*X+X
            end
            @test isapprox(M, p, Y, 3*X)
        end
    end

    test_project_tangent && @testset "project tangent test" begin
        for (p, X) in zip(pts, tv)
            atol = find_eps(p) * projection_atol_multiplier
            @test isapprox(M, p, X, project(M, p, X); atol = atol)
            if is_mutating
                X2 = allocate(X)
                project!(M, X2, p, X)
            else
                X2 = project(M, p, X)
            end
            @test isapprox(M, p, X2, X; atol = atol)
        end
    end

    test_project_point && @testset "project point test" begin
        for p in pts
            atol = find_eps(p) * projection_atol_multiplier
            @test isapprox(M, p, project(M, p); atol = atol)
            if is_mutating
                p2 = allocate(p)
                project!(M, p2, p)
            else
                p2 = project(M, p)
            end
            @test isapprox(M, p2, p; atol = atol)
        end
    end

    test_vector_transport && !( default_inverse_retraction_method === nothing) && @testset "vector transport" begin
        tvatol = is_tangent_atol_multiplier*find_eps(pts[1])
        X1 = inverse_retract(M, pts[1], pts[2], default_inverse_retraction_method)
        X2 = inverse_retract(M, pts[1], pts[3], default_inverse_retraction_method)
        v1t1 = vector_transport_to(M, pts[1], X1, pts[3])
        v1t2 = vector_transport_direction(M, pts[1], X1, X2)
        @test is_tangent_vector(M, pts[3], v1t1; atol=tvatol)
        @test is_tangent_vector(M, pts[3], v1t2; atol=tvatol)
        @test isapprox(M, pts[3], v1t1, v1t2)
        @test isapprox(M, pts[1], vector_transport_to(M, pts[1], X1, pts[1]), X1)
    end

    for btype ∈ basis_types_vecs
        @testset "Basis support for $(btype)" begin
            p = pts[1]
            b = get_basis(M, p, btype)
            @test isa(b, CachedBasis)
            bvectors = get_vectors(M, p, b)
            N = length(bvectors)
            @test real_dimension(number_system(btype)) * N == manifold_dimension(M)

            # test orthonormality
            for i in 1:N
                @test norm(M, p, bvectors[i]) ≈ 1
                for j in i+1:N
                    @test real(inner(M, p, bvectors[i], bvectors[j])) ≈ 0 atol = sqrt(find_eps(p))
                end
            end
            if isa(btype, ProjectedOrthonormalBasis)
                # check projection idempotency
                for i in 1:N
                    @test norm(M, p, bvectors[i]) ≈ 1
                    for j in i+1:N
                        @test real(inner(M, p, bvectors[i], bvectors[j])) ≈ 0 atol = sqrt(find_eps(p))
                    end
                end
                # check projection idempotency
                for i in 1:N
                    @test isapprox(M, p, project(M, p, bvectors[i]), bvectors[i])
                end
            end
            if !isa(btype, ProjectedOrthonormalBasis) &&
                (basis_has_specialized_diagonalizing_get || !isa(btype, DiagonalizingOrthonormalBasis))

                X1 = inverse_retract(M, p, pts[2], default_inverse_retraction_method)
                Xb = get_coordinates(M, p, X1, btype)

                @test get_coordinates(M, p, X1, b) ≈ Xb
                @test isapprox(M, p, get_vector(M, p, Xb, b), get_vector(M, p, Xb, btype))
            end
        end
    end

    for btype ∈ (basis_types_to_from..., basis_types_vecs...)
        p = pts[1]
        N = manifold_dimension(M)
        if !isa(btype, ProjectedOrthonormalBasis) &&
            (basis_has_specialized_diagonalizing_get || !isa(btype, DiagonalizingOrthonormalBasis))

            X1 = inverse_retract(M, p, pts[2], default_inverse_retraction_method)

            Xb = get_coordinates(M, p, X1, btype)
            #@test isa(Xb, AbstractVector{<:Real})
            @test length(Xb) == N
            Xbi = get_vector(M, p, Xb, btype)
            @test isapprox(M, p, X1, Xbi)

            Xs = [[ifelse(i==j, 1, 0) for j in 1:N] for i in 1:N]
            Xs_invs = [get_vector(M, p, Xu, btype) for Xu in Xs]
            # check orthonormality of inverse representation
            for i in 1:N
                @test norm(M, p, Xs_invs[i]) ≈ 1 atol = find_eps(p)
                for j in i+1:N
                    @test real(inner(M, p, Xs_invs[i], Xs_invs[j])) ≈ 0 atol = sqrt(find_eps(p))
                end
            end

            if is_mutating
                Xb_s = allocate(Xb)
                @test get_coordinates!(M, Xb_s, p, X1, btype) === Xb_s
                @test isapprox(Xb_s, Xb; atol = find_eps(p))

                Xbi_s = allocate(Xbi)
                @test get_vector!(M, Xbi_s, p, Xb, btype) === Xbi_s
                @test isapprox(M, p, X1, Xbi_s)
            end
        end
    end

    test_vee_hat && @testset "vee and hat" begin
        p = pts[1]
        q = pts[2]
        X = inverse_retract(M, p, q, default_inverse_retraction_method)
        Y = vee(M, p, X)
        @test length(Y) == manifold_dimension(M)
        @test isapprox(M, p, X, hat(M, p, Y))
        Y2 = allocate(Y)
        vee_ret = vee!(M, Y2, p, X)
        @test vee_ret === Y2
        @test isapprox(Y, Y2)
        X2 = allocate(X)
        hat_ret = hat!(M, X2, p, Y)
        @test hat_ret === X2
        @test isapprox(M, p, X2, X)
    end

    test_forward_diff && @testset "ForwardDiff support" begin
        for (p, X) in zip(pts, tv)
            exp_f(t) = distance(M, p, exp(M, p, t[1]*X))
            d12 = norm(M, p, X)
            for t ∈ 0.1:0.1:0.9
                @test d12 ≈ ForwardDiff.derivative(exp_f, t)
            end

            retract_f(t) = distance(M, p, retract(M, p, t[1]*X))
            for t ∈ 0.1:0.1:0.9
                @test ForwardDiff.derivative(retract_f, t) ≥ 0
            end
        end
    end

    test_reverse_diff && @testset "ReverseDiff support" begin
        for (p, X) in zip(pts, tv)
            exp_f(t) = distance(M, p, exp(M, p, t[1]*X))
            d12 = norm(M, p, X)
            for t ∈ 0.1:0.1:0.9
                @test d12 ≈ ReverseDiff.gradient(exp_f, [t])[1]
            end

            retract_f(t) = distance(M, p, retract(M, p, t[1]*X))
            for t ∈ 0.1:0.1:0.9
                @test ReverseDiff.gradient(retract_f, [t])[1] ≥ 0
            end
        end
    end

    test_musical_isomorphisms && @testset "Musical isomorphisms" begin
        if default_inverse_retraction_method !== nothing
            tv_m = inverse_retract(M, pts[1], pts[2],default_inverse_retraction_method)
        else
            tv_m = zero_tangent_vector(M,pts[1])
        end
        ctv_m = flat(M, pts[1], FVector(TangentSpace, tv_m))
        @test ctv_m.type == CotangentSpace
        tv_m_back = sharp(M, pts[1], ctv_m)
        @test tv_m_back.type == TangentSpace
    end

    @testset "number_eltype" begin
        for (p, X) in zip(pts, tv)
            @test number_eltype(X) == number_eltype(p)
            p = retract(M, p, X, default_retraction_method)
            @test number_eltype(p) == number_eltype(p)
        end
    end

    is_mutating && @testset "copyto!" begin
        for (p, X) in zip(pts, tv)
            p2 = allocate(p)
            copyto!(p2, p)
            @test isapprox(M, p2, p)

            X2 = allocate(X)
            if default_inverse_retraction_method === nothing
                X3 = zero_tangent_vector(M, p)
                copyto!(X2, X3)
                @test isapprox(M, p, X2, zero_tangent_vector(M, p))
            else
                q = retract(M, p, X, default_retraction_method)
                X3 = inverse_retract(M, p, q, default_inverse_retraction_method)
                copyto!(X2, X3)
                @test isapprox(M, p, X2, inverse_retract(M, p, q, default_inverse_retraction_method))
            end
        end
    end

    is_mutating && @testset "point distributions" begin
        for p in pts
            prand = allocate(p)
            for pd ∈ point_distributions
                for _ in 1:10
                    @test is_manifold_point(M, rand(pd))
                    if test_mutating_rand
                        rand!(pd, prand)
                        @test is_manifold_point(M, prand)
                    end
                end
            end
        end
    end

    @testset "tangent vector distributions" begin
        for tvd ∈ tvector_distributions
            supp = Manifolds.support(tvd)
            for _ in 1:10
                randtv = rand(tvd)
                atol = rand_tvector_atol_multiplier * find_eps(randtv)
                @test is_tangent_vector(M, supp.point, randtv; atol = atol)
            end
        end
    end
end
