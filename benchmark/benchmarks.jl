using Manifolds
using ManifoldsBase
using BenchmarkTools
using RecursiveArrayTools
using StaticArrays
using HybridArrays
using LinearAlgebra
using Random

# Define a parent BenchmarkGroup to contain our suite
SUITE = BenchmarkGroup()

Random.seed!(12334)

function add_manifold(
        M::AbstractManifold,
        pts,
        name;
        test_tangent_vector_broadcasting = true,
        test_exp_log::Bool = true,
        retraction_methods = [],
        inverse_retraction_methods = [],
    )
    SUITE["manifolds"][name] = BenchmarkGroup()
    m = default_inverse_retraction_method(M)
    tv = test_exp_log ? log(M, pts[1], pts[2]) : inverse_retract(M, pts[1], pts[2], m)
    tv1 = test_exp_log ? log(M, pts[1], pts[2]) : inverse_retract(M, pts[1], pts[2], m)
    tv2 = test_exp_log ? log(M, pts[1], pts[3]) : inverse_retract(M, pts[1], pts[3], m)
    p = similar(pts[1])
    SUITE["manifolds"][name]["similar"] = @benchmarkable similar($(pts[1]))
    if test_exp_log
        SUITE["manifolds"][name]["log"] = @benchmarkable log($M, $(pts[1]), $(pts[2]))
        SUITE["manifolds"][name]["log!"] = @benchmarkable log!($M, $tv, $(pts[1]), $(pts[2]))
    end
    for iretr in inverse_retraction_methods
        SUITE["manifolds"][name]["inverse_retract: " * string(iretr)] =
            @benchmarkable inverse_retract($M, $(pts[1]), $(pts[2]), $iretr)
        SUITE["manifolds"][name]["inverse_retract!: " * string(iretr)] =
            @benchmarkable inverse_retract!($M, $tv, $(pts[1]), $(pts[2]), $iretr)
    end
    if test_exp_log
        SUITE["manifolds"][name]["exp"] = @benchmarkable exp($M, $(pts[1]), $tv1)
        SUITE["manifolds"][name]["exp!"] = @benchmarkable exp!($M, $p, $(pts[1]), $tv1)
    end
    for retr in retraction_methods
        SUITE["manifolds"][name]["retract: " * string(retr)] =
            @benchmarkable retract($M, $(pts[1]), $tv1, $retr)
        SUITE["manifolds"][name]["retract!: " * string(retr)] =
            @benchmarkable retract!($M, $p, $(pts[1]), $tv1, $retr)
    end
    SUITE["manifolds"][name]["norm"] = @benchmarkable norm($M, $(pts[1]), $tv1)
    SUITE["manifolds"][name]["inner"] = @benchmarkable inner($M, $(pts[1]), $tv1, $tv2)
    if test_exp_log
        SUITE["manifolds"][name]["distance"] = @benchmarkable distance($M, $(pts[1]), $(pts[2]))
    end
    SUITE["manifolds"][name]["isapprox (pt)"] =
        @benchmarkable isapprox($M, $(pts[1]), $(pts[2]))
    SUITE["manifolds"][name]["isapprox (tv)"] =
        @benchmarkable isapprox($M, $(pts[1]), $tv1, $tv2)
    SUITE["manifolds"][name]["2 * tv1 + 3 * tv2"] = @benchmarkable 2 * $tv1 + 3 * $tv2
    SUITE["manifolds"][name]["tv = 2 * tv1 + 3 * tv2"] =
        @benchmarkable $tv = 2 * $tv1 + 3 * $tv2
    if test_tangent_vector_broadcasting
        SUITE["manifolds"][name]["tv = 2 .* tv1 .+ 3 .* tv2"] =
            @benchmarkable $tv = 2 .* $tv1 .+ 3 .* $tv2
        SUITE["manifolds"][name]["tv .= 2 .* tv1 .+ 3 .* tv2"] =
            @benchmarkable $tv .= 2 .* $tv1 .+ 3 .* $tv2
    end
    return nothing
end

# General manifold benchmarks
function add_manifold_benchmarks()
    SUITE["manifolds"] = BenchmarkGroup()

    s2 = Manifolds.Sphere(2)
    array_s2 = ValidationManifold(s2)
    r2 = Manifolds.Euclidean(2)

    pts_r2 = [SizedVector{2}([1.0, 1.0]), SizedVector{2}([-2.0, 3.0]), SizedVector{2}([3.0, -2.0])]

    add_manifold(r2, pts_r2, "Euclidean{2} -- SizedArray")

    add_manifold(
        r2,
        [
            MVector{2, Float64}([1.0, 1.0]),
            MVector{2, Float64}([-2.0, 3.0]),
            MVector{2, Float64}([3.0, -2.0]),
        ],
        "Euclidean{2} -- MVector",
    )

    pts_s2 = [SizedVector{3}([1.0, 0.0, 0.0]), SizedVector{3}([0.0, 1.0, 0.0]), SizedVector{3}([0.0, 0.0, 1.0])]

    add_manifold(
        s2,
        pts_s2,
        "Sphere{2} -- SizedArray"
    )

    add_manifold(
        array_s2,
        [SizedVector{3}([1.0, 0.0, 0.0]), SizedVector{3}([0.0, 1.0, 0.0]), SizedVector{3}([0.0, 0.0, 1.0])],
        "ValidationManifold{Sphere{2}} -- SizedArray";
        test_tangent_vector_broadcasting = false,
    )

    retraction_methods_rot = [Manifolds.PolarRetraction(), Manifolds.QRRetraction()]

    inverse_retraction_methods_rot =
        [Manifolds.PolarInverseRetraction(), Manifolds.QRInverseRetraction()]

    so2 = Manifolds.Rotations(2)
    angles = (0.0, π / 2, 2π / 3)
    add_manifold(
        so2,
        [SizedMatrix{2, 2}([cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]) for ϕ in angles],
        "Rotations(2) -- SizedArray",
        retraction_methods = retraction_methods_rot,
        inverse_retraction_methods = inverse_retraction_methods_rot,
    )

    so3 = Manifolds.Rotations(3)
    angles = (0.0, π / 2, 2π / 3)
    add_manifold(
        so3,
        [SizedMatrix{3, 3}([cos(ϕ) -sin(ϕ) 0; sin(ϕ) cos(ϕ) 0; 0 0 1]) for ϕ in angles],
        "Rotations(3) -- SizedArray",
        retraction_methods = retraction_methods_rot,
        inverse_retraction_methods = inverse_retraction_methods_rot,
    )

    m_prod = ProductManifold(s2, r2)
    pts_prod_mpoints = [ArrayPartition(p[1], p[2]) for p in zip(pts_s2, pts_r2)]
    add_manifold(
        m_prod,
        pts_prod_mpoints,
        "ProductManifold with AbstractManifoldPoint";
        test_tangent_vector_broadcasting = false,
    )

    # vector spaces and bundles
    begin
        T = MVector{3, Float64}
        TB = TangentBundle(s2)

        pts_tb = [
            ArrayPartition(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, -1.0, -1.0])),
            ArrayPartition(convert(T, [0.0, 1.0, 0.0]), convert(T, [2.0, 0.0, 1.0])),
            ArrayPartition(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, 2.0, -1.0])),
        ]
        add_manifold(
            TB,
            pts_tb,
            "Tangent bundle of S² using MVectors, ArrayPartition";
            test_tangent_vector_broadcasting = false,
            test_exp_log = false,
            retraction_methods = [Manifolds.FiberBundleProductRetraction()],
            inverse_retraction_methods = [Manifolds.FiberBundleInverseProductRetraction()],
        )
    end

    # power manifolds
    begin
        Ms = Sphere(2)
        Ms1 = PowerManifold(Ms, 5)
        Ms2 = PowerManifold(Ms, 5, 7)
        Mr = Manifolds.Rotations(3)
        Mr1 = PowerManifold(Mr, 5)
        Mr2 = PowerManifold(Mr, 5, 7)

        types_s1 =
            (Array{Float64, 2}, HybridArray{Tuple{3, StaticArrays.Dynamic()}, Float64, 2})
        types_s2 = (
            Array{Float64, 3},
            HybridArray{Tuple{3, StaticArrays.Dynamic(), StaticArrays.Dynamic()}, Float64, 3},
        )

        types_r1 =
            (Array{Float64, 3}, HybridArray{Tuple{3, 3, StaticArrays.Dynamic()}, Float64, 3})
        types_r2 = (
            Array{Float64, 4},
            HybridArray{Tuple{3, 3, StaticArrays.Dynamic(), StaticArrays.Dynamic()}, Float64, 4},
        )

        retraction_methods =
            [ExponentialRetraction()]
        inverse_retraction_methods =
            [LogarithmicInverseRetraction()]


        trim(s::String) = s[1:min(length(s), 20)]

        for T in types_s1
            pts1 = [convert(T, rand(Ms1)) for _ in 1:3]
            add_manifold(
                Ms1,
                pts1,
                "power manifold S²^(5,), type $(trim(string(T)))";
                test_tangent_vector_broadcasting = true,
            )
        end
        for T in types_s2
            pts2 = [convert(T, rand(Ms2)) for _ in 1:3]
            add_manifold(
                Ms2,
                pts2,
                "power manifold S²^(5,7), type $(trim(string(T)))";
                test_tangent_vector_broadcasting = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
            )
        end

        for T in types_r1
            pts1 = [convert(T, rand(Mr1)) for _ in 1:3]
            add_manifold(
                Mr1,
                pts1,
                "power manifold SO(3)^(5,), type $(trim(string(T)))";
                test_tangent_vector_broadcasting = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
            )
        end
        for T in types_r2
            pts2 = [convert(T, rand(Mr2)) for _ in 1:3]
            add_manifold(
                Mr2,
                pts2,
                "power manifold SO(3)^(5,7), type $(trim(string(T)))";
                test_tangent_vector_broadcasting = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
            )
        end

        types_spd = [Matrix{Float64}, MMatrix{3, 3, Float64, 9}]

        A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
        pts_spd = [#
            [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1],
            [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1],
            A(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1] * transpose(A(π / 6)),
        ]

        M_spd_1 = Manifolds.SymmetricPositiveDefinite(3)
        M_spd_2 = MetricManifold(
            Manifolds.SymmetricPositiveDefinite(3),
            Manifolds.AffineInvariantMetric(),
        )
        M_spd_3 = MetricManifold(
            Manifolds.SymmetricPositiveDefinite(3),
            Manifolds.LogCholeskyMetric(),
        )

        for T in types_spd
            pts = [convert(T, a) for a in pts_spd]
            for M in (M_spd_1, M_spd_2, M_spd_3)
                add_manifold(
                    M,
                    pts,
                    "SPD manifold $(trim(string(M))), type $(trim(string(T)))";
                    test_tangent_vector_broadcasting = true,
                )
            end
        end
    end
    return nothing
end

add_manifold_benchmarks()
