using Manifolds
using Test

@testset "Flag manifold" begin
    M = Flag(5, 1, 2)

    @test repr(M) == "Flag(5, 1, 2)"

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

    X1_ortho = Manifolds.OrthogonalTVector(
        [
            0.0 -0.4015327292726182 0.780864290667572 0.4338924112653854 -1.7532409316999389
            0.4015327292726182 0.0 -0.03497663360767567 -0.13649144092765172 -0.6576516594131526
            -0.780864290667572 0.03497663360767567 0.0 0.0 0.0
            -0.4338924112653854 0.13649144092765172 0.0 0.0 0.0
            1.7532409316999389 0.6576516594131526 0.0 0.0 0.0
        ],
    )
    X2_ortho = Manifolds.OrthogonalTVector(
        [
            0.0 -0.2979953307015468 0.7855622797662635 -1.783621666926397 -0.7481810438379631
            0.2979953307015468 0.0 -0.10452766698617191 -0.018998219248410615 -0.014502786096688508
            -0.7855622797662635 0.10452766698617191 0.0 0.0 0.0
            1.783621666926397 0.018998219248410615 0.0 0.0 0.0
            0.7481810438379631 0.014502786096688508 0.0 0.0 0.0
        ],
    )

    X2_ortho_wrong1 = Manifolds.OrthogonalTVector(
        [
            0.0 -0.1979953307015468 0.7855622797662635 -1.783621666926397 -0.7481810438379631
            0.3979953307015468 0.0 -0.10452766698617191 -0.018998219248410615 -0.014502786096688508
            -0.7855622797662635 0.10452766698617191 0.0 0.0 0.0
            1.783621666926397 0.018998219248410615 0.0 0.0 0.0
            0.7481810438379631 0.014502786096688508 0.0 0.0 0.0
        ],
    )

    @test check_point(M, p1_ortho) === nothing
    @test check_vector(M, p1_ortho, X1_ortho) === nothing
    @test check_vector(M, p1_ortho, X2_ortho_wrong1) isa DomainError
    @test isapprox(M, p1_ortho, X2_ortho, project(M, p1_ortho, X2_ortho_wrong1))

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

    p1o = Manifolds.stiefel_point_to_orthogonal(M, p1)
    p2o = Manifolds.stiefel_point_to_orthogonal(M, p2)
    X1o = Manifolds.stiefel_tv_to_orthogonal(M, p1, X1)
    X2o = Manifolds.stiefel_tv_to_orthogonal(M, p1, X2)

    @test check_point(M, p1) === nothing
    @test check_vector(M, p1, X1) === nothing
    @test check_vector(M, p1, X2) === nothing

    @testset "conversion between Stiefel and orthogonal coordinates" begin
        p1os = Manifolds.orthogonal_point_to_stiefel(M, p1o)
        @test isapprox(p1, p1os)

        X1os = Manifolds.orthogonal_tv_to_stiefel(M, p1o, X1o)
        @test isapprox(X1, X1os)
    end
    @test inner(M, p1, X1, X2) ≈ inner(M, p1o, X1o, X2o)

    @testset "projection" begin
        X_to_project = [
            -0.5847114032301931 0.3817639698271648
            -1.0084766091896127 1.5421885878147963
            -0.22275127208629275 0.4376200479885611
            -1.2310630343987463 0.6594777460928264
            0.4109127063901268 -0.2827478363715998
        ]
        @test project(M, p1, X_to_project) ≈ [
            -0.21816592446975824 0.2612682744329179
            -0.3419669237188776 1.3220342586044491
            -0.10450160051834301 0.39801647223092396
            -1.3483600206492228 0.6981642398704684
            1.288033080060952 -0.5718664356833303
        ]
    end
end
