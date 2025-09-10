using Manifolds, ManifoldsBase
using Test
using Random

@testset "Flag manifold" begin
    M = Flag(5, 1, 2)

    @test repr(M) == "Flag(5, 1, 2)"
    @test manifold_dimension(M) == 7

    @testset "Flag constructor errors" begin
        @test_throws ErrorException Flag(0, 0)
        @test_throws ErrorException Flag(5, 2, 2)
        @test_throws ErrorException Flag(5, 2, 5)
    end

    p1 = [
        -0.3146651787309489 -0.48660897424165994
        0.6064615017863313 -0.34830198287474623
        0.7186856131733219 0.21637936237010186
        -0.04188711028762897 0.09081337776276559
        0.12217500380489567 -0.7660485212260545
    ]
    p2 = [
        -0.03091072893493796 0.13880702776118134
        0.1223349326386293 -0.21714532137397113
        0.49375943573694303 0.2005115408134847
        0.6786375747132262 0.5164995985838793
        0.5288963536470062 -0.7915831005730196
    ]
    p3 = [
        -0.7170895889528139 -0.11876117758587547
        -0.23046861559362633 -0.31457380030454196
        -0.425250070689939 0.7225016888982094
        -0.4413689679425084 -0.5121935612141849
        -0.238793949164145 0.32029388419128463
    ]

    X1 = [
        -0.6804229240952372 -0.034412196486963954
        0.8344934559155401 0.48097313913408224
        -1.0810467279810454 0.11289047449146702
        -1.1353782173810583 0.34131188565927895
        0.0751500891676202 -0.12447744196142059
    ]
    X2 = [
        -0.9739291698646104 0.8292235723623507
        0.3096151247876431 -0.9579823665757918
        -0.5250283055037331 -0.05809529998521694
        0.5756364579603817 -0.37414047858670213
        -0.7594778759730498 -0.1519330114054877
    ]

    X1_wrong = [
        -1.6804229240952372 -0.034412196486963954
        0.8344934559155401 0.48097313913408224
        -1.0810467279810454 0.11289047449146702
        -1.1353782173810583 0.34131188565927895
        0.0751500891676202 -0.12447744196142059
    ]

    X1_wrong2 = [
        -0.6804229240952372 -0.034412196486963954
        0.8344934559155401 0.48097313913408224
        -1.0810467279810454 0.11289047449146702
        -1.1353782173810583 0.34131188565927895
        0.0751500891676202 -1.12447744196142059
    ]

    p1o = convert(Manifolds.OrthogonalPoint, M, p1)
    p2o = convert(Manifolds.OrthogonalPoint, M, p2)
    X1o = convert(Manifolds.OrthogonalTangentVector, M, p1, X1)
    X2o = convert(Manifolds.OrthogonalTangentVector, M, p1, X2)

    @test check_point(M, p1) === nothing
    @test check_vector(M, p1, X1) === nothing
    @test check_vector(M, p1, X2) === nothing
    @test check_vector(M, p1, X1_wrong) isa DomainError
    @test check_vector(M, p1, X1_wrong2) isa DomainError

    @testset "injectivity_radius" begin
        @test injectivity_radius(M) == π / 2
        @test injectivity_radius(M, p1) == π / 2
        @test injectivity_radius(M, PolarRetraction()) == π / 2
        @test injectivity_radius(M, p1, PolarRetraction()) == π / 2
    end

    @testset "conversion between Stiefel and orthogonal coordinates" begin
        p1os = convert(AbstractMatrix, M, p1o)
        @test isapprox(p1, p1os)

        X1os = convert(AbstractMatrix, M, p1o, X1o)
        @test isapprox(X1, X1os)
    end
    @test inner(M, p1, X1, X2) ≈ inner(M, p1o, X1o, X2o)

    @test eltype(p1o) === Float64
    @test eltype(X1o) === Float64

    @testset "projection" begin
        X_to_project = [
            -0.5847114032301931 0.3817639698271648
            -1.0084766091896127 1.5421885878147963
            -0.22275127208629275 0.4376200479885611
            -1.2310630343987463 0.6594777460928264
            0.4109127063901268 -0.2827478363715998
        ]
        @test project(M, p1, X_to_project) ≈ [
            -0.43871844928855513 0.4038883314300648
            -0.4998326553201793 1.047159312796425
            -0.00642898530655063 0.07227663814064052
            -1.3071994157012459 0.7171493149930495
            0.9408262991884606 -0.6272415047537456
        ]

        @test project(Flag(5, 2), p1, X_to_project) ≈
            project(Grassmann(5, 2), p1, X_to_project)
        @test ManifoldsBase.embed_project(Flag(5, 2), p1) ≈ p1
    end

    @testset "retraction, inverse retraction and VT" begin
        @test default_retraction_method(M) === PolarRetraction()
        @test default_inverse_retraction_method(M) === PolarInverseRetraction()
        @test default_vector_transport_method(M) === ProjectionTransport()

        @test isapprox(
            M,
            retract(M, p1, X1, PolarRetraction()),
            [
                -0.4688597409064468 -0.45541418524363636
                0.6719554741291286 0.129306063763389
                -0.16490485737874422 0.2766729757843485
                -0.5429179652489367 0.3552156355310046
                0.08180987206864788 -0.7570678823578711
            ],
        )
    end

    test_manifold(
        M,
        [p1, p2, p3],
        test_exp_log = false,
        default_inverse_retraction_method = PolarInverseRetraction(),
        test_injectivity_radius = false,
        test_is_tangent = true,
        test_project_tangent = true,
        test_default_vector_transport = false,
        test_vee_hat = false,
        default_retraction_method = PolarRetraction(),
        is_point_atol_multiplier = 10.0,
        projection_atol_multiplier = 200.0,
        retraction_atol_multiplier = 10.0,
        is_tangent_atol_multiplier = 4 * 10.0^2,
        rand_tvector_atol_multiplier = 10,
        retraction_methods = [PolarRetraction()],
        inverse_retraction_methods = [PolarInverseRetraction()],
        vector_transport_methods = [ProjectionTransport()],
        vector_transport_retractions = [PolarRetraction()],
        vector_transport_inverse_retractions = [PolarInverseRetraction()],
        test_vector_transport_direction = [true, true, false],
        mid_point12 = nothing,
        test_inplace = true,
        test_rand_point = true,
        test_rand_tvector = true,
    )

    @testset "Orthogonal representation" begin
        p1_ortho = Manifolds.OrthogonalPoint(
            [
                0.5483710839601286 0.11431583140179802 -0.4549972669468796 -0.24420811180644114 -0.6477352315306864
                -0.14598104021651825 -0.3863352173686217 -0.7998129702578309 -0.12512220381981268 0.41722689562953885
                -0.27361606694009666 -0.0062486771708479966 -0.269431859703507 0.8487540190060319 -0.36348079748958584
                -0.5860445034111339 0.7091711045363329 -0.2369412411721794 -0.29834711991839624 -0.09206535059673804
                -0.5095731332277309 -0.5785449758108591 0.1566589317170019 -0.3395768594835529 -0.5155254295115427
            ],
        )
        p2_ortho = Manifolds.OrthogonalPoint(
            [
                0.6899566465726515 -0.09915024703518771 -0.6226523787333554 0.3430242785649697 -0.09363446980550538
                -0.11411076737780768 0.674849625290569 0.09305040872989712 0.4360207581932596 -0.5768745408435015
                0.17343507284015292 -0.4980917752801034 0.24595262802592188 -0.2569777450904701 -0.7715533696398492
                0.6921672300039265 0.30834121120497526 0.6003132332813077 -0.1586226925653731 0.20073137904172766
                0.041992378311914824 0.43769566210896327 -0.4275242255210552 -0.7752557788698422 -0.15119742542861503
            ],
        )
        p3_ortho = Manifolds.OrthogonalPoint(
            [
                -0.7552523075773412 -0.3463492733971505 0.08681321812496035 -0.5044534625926571 0.21823451137121694
                0.5275217893709137 -0.33414916745355194 -0.48394871031627773 -0.47707946364713544 0.38503756541197426
                -0.22718317005515387 0.0007090779127115407 -0.708207049417279 -0.0698341229845377 -0.6647956640012455
                0.17906151029959072 0.5340613892960445 0.2927002843490751 -0.7131058180286085 -0.2975265430648627
                -0.26007594354521 0.6951003124578901 -0.41355341812109553 0.06726943589156384 0.5231103636568496
            ],
        )

        X1_ortho = Manifolds.OrthogonalTangentVector(
            [
                0.0 -0.4015327292726182 0.780864290667572 0.4338924112653854 -1.7532409316999389
                0.4015327292726182 0.0 -0.03497663360767567 -0.13649144092765172 -0.6576516594131526
                -0.780864290667572 0.03497663360767567 0.0 0.0 0.0
                -0.4338924112653854 0.13649144092765172 0.0 0.0 0.0
                1.7532409316999389 0.6576516594131526 0.0 0.0 0.0
            ],
        )
        X2_ortho = Manifolds.OrthogonalTangentVector(
            [
                0.0 -0.2979953307015468 0.7855622797662635 -1.783621666926397 -0.7481810438379631
                0.2979953307015468 0.0 -0.10452766698617191 -0.018998219248410615 -0.014502786096688508
                -0.7855622797662635 0.10452766698617191 0.0 0.0 0.0
                1.783621666926397 0.018998219248410615 0.0 0.0 0.0
                0.7481810438379631 0.014502786096688508 0.0 0.0 0.0
            ],
        )

        X2_ortho_wrong1 = Manifolds.OrthogonalTangentVector(
            [
                0.0 -0.1979953307015468 0.7855622797662635 -1.783621666926397 -0.7481810438379631
                0.3979953307015468 0.0 -0.10452766698617191 -0.018998219248410615 -0.014502786096688508
                -0.7855622797662635 0.10452766698617191 0.0 0.0 0.0
                1.783621666926397 0.018998219248410615 0.0 0.0 0.0
                0.7481810438379631 0.014502786096688508 0.0 0.0 0.0
            ],
        )
        X2_ortho_wrong2 = Manifolds.OrthogonalTangentVector(
            [
                1.0 -0.2979953307015468 0.7855622797662635 -1.783621666926397 -0.7481810438379631
                0.2979953307015468 0.0 -0.10452766698617191 -0.018998219248410615 -0.014502786096688508
                -0.7855622797662635 0.10452766698617191 0.0 0.0 0.0
                1.783621666926397 0.018998219248410615 0.0 0.0 0.0
                0.7481810438379631 0.014502786096688508 0.0 0.0 0.0
            ],
        )

        @test check_point(M, p1_ortho) === nothing
        @test check_vector(M, p1_ortho, X1_ortho) === nothing
        @test check_vector(M, p1_ortho, X2_ortho_wrong1) isa DomainError
        @test check_vector(M, p1_ortho, X2_ortho_wrong2) isa DomainError
        @test is_point(M, p1_ortho)
        @test is_vector(M, p1_ortho, X1_ortho)
        @test isapprox(M, p1_ortho, X2_ortho, project(M, p1_ortho, X2_ortho_wrong1))
        @test isapprox(
            M,
            p1_ortho,
            X2_ortho,
            project!(M, similar(X2_ortho_wrong1), p1_ortho, X2_ortho_wrong1),
        )

        q_tmp = similar(p1_ortho)
        Y_tmp = similar(X1_ortho)
        rand!(M, q_tmp)
        @test is_point(M, q_tmp)
        rand!(M, Y_tmp; vector_at = p1_ortho)
        @test is_vector(M, p1_ortho, Y_tmp)

        q_tmp = similar(p1_ortho)
        Y_tmp = similar(X1_ortho)
        rand!(Random.default_rng(), M, q_tmp)
        @test is_point(M, q_tmp)
        rand!(Random.default_rng(), M, Y_tmp; vector_at = p1_ortho)
        @test is_vector(M, p1_ortho, Y_tmp)

        q_tmp = similar(p1_ortho.value)
        embed!(M, q_tmp, p1_ortho)
        @test isapprox(M, q_tmp, p1_ortho.value)
        Y_tmp = similar(X1_ortho.value)
        embed!(M, Y_tmp, p1_ortho, X1_ortho)
        @test isapprox(Y_tmp, X1_ortho.value)

        @test retract(M, p1_ortho, X1_ortho, QRRetraction()).value ≈
            retract(OrthogonalMatrices(5), p1_ortho.value, X1_ortho.value, QRRetraction())

        @testset "field parameters" begin
            M = Flag(5, 1, 2; parameter = :field)
            @test get_embedding(M, p1_ortho) == OrthogonalMatrices(5; parameter = :field)
        end
    end

    @testset "field parameters" begin
        M = Flag(5, 1, 2; parameter = :field)
        @test Manifolds.get_parameter(M.size)[1] == 5
        @test get_embedding(M) == Stiefel(5, 2; parameter = :field)
        @test repr(M) == "Flag(5, 1, 2; parameter=:field)"
    end
end
