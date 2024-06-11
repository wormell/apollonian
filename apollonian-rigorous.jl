using Distributed
addprocs(Base.Sys.CPU_THREADS-2)
### DISTRIBUTED PART ###
@everywhere using JLD, IntervalArithmetic # BEWARE! IntervalArithmetic resets the BigFloat precision
@eval @everywhere PREC = $(parse(Int,ARGS[1]))
@everywhere setprecision(BigFloat,PREC)
@everywhere println("On this process, PREC = ",precision(BigFloat))
@everywhere const JLDbox = load("apollonian-nonrigorous-"*string(PREC)*".jld")
@everywhere begin
    const TYPE = Interval{BigFloat}

    import IntervalArithmetic.@round
end # separate so we can use the macros in the next block
@everywhere begin
#    println("BigFloat precision: ",precision(BigFloat))
    const V = TYPE.(JLDbox["eigvec"])
    const s_est = TYPE(JLDbox["s"])
    const mlogϵ = TYPE(JLDbox["mlogϵ"])

    const s_allowed = TYPE(13//10,131/100)
    const Rad = TYPE(14//10)
    const rad = TYPE(9//10)
    const ν = 10
    const W = 3
    const Wt2 = 68//10
    
    const Dminus = TYPE(33//10)
    const Dplus = TYPE(59//100)

    function CLerror(N,ΔR)
         16(N + 2/ΔR)*exp(-ΔR*(N-1))/ΔR 
    end
    function H22norm(cfs::Matrix,rad)
        N1, N2 = size(cfs)
        sum(cosh.(hypot.((0:N1-1),(0:2:2N2-2)')*rad).*abs.(cfs))
    end

    # The IntervalArithmetic routines for powers and sin/cos are kind of garbage because they keep calling "rationalise"
    # In any case, we know something about the sign of s so we can simplify some things

    # because the IntervalArithmetic routine passes to rationalise which has a type overflow for Int, thus limiting your accuracy to 10^{-38}ish!!!! 
    function pow_plus(x::Interval{T},y::Interval{T}) where T <: AbstractFloat
        @assert y >= 0
        @assert x >= 0
        x_yhi = @round(x.lo^y.hi,x.hi^y.hi)
        x_ylo = @round(x.lo^y.lo,x.hi^y.lo)
        hull(x_ylo,x_yhi)
    end
    
    function pow_minus(x::Interval{T},s::Interval{T}) where T <: AbstractFloat
        @assert s >= 0
        @assert x >= 0
        x_shi = @round(x.hi^(-s.hi),x.lo^(-s.hi))
        x_slo = @round(x.hi^(-s.lo),x.lo^(-s.lo))
        hull(x_shi,x_slo)
    end
    function pow_minus(x::Complex{Interval{T}},s::Interval{T}) where T <: AbstractFloat
        absxs = pow_minus(abs(x),s)
        θ = -s*angle(x)
        absθ = abs(θ)
        @assert absθ < pi/2 # kind of arbitrary from the perspective of the problem, will be true in the routines if N > sqrt(2)*τ
        sinθ = @round(sin(θ.lo),sin(θ.hi))
        cosθ = @round(cos(absθ.hi),cos(absθ.lo))
        Complex(absxs*cosθ,absxs*sinθ)
    end


    using LinearAlgebra

    acosnodesx(N1) = ((TYPE(pi)*(1:2:2N1))/2N1);
    dctmatx(N1) = [cos.(k*acosnodex)*(k==0 ? 1 : 2) for k = 0:N1-1, acosnodex in acosnodesx(N1)]/N1
    acosnodesy(N2) = ((TYPE(pi)*(1:2:2N2))/4N2);
    dctmaty(N2) = [cos.(2k*acosnodey)*(k==0 ? 1 : 2) for k = 0:N2-1, acosnodey in acosnodesy(N2)]/N2
    const dct_V = dctmatx(size(V,1))*V*dctmatx(size(V,2))'
    const V_Hrnorm = H22norm(dct_V,rad)

    epstype(::Type{T}) where T<:Real = T
    epstype(::Type{Complex{T}}) where T<:Real = T
    epstype(::Interval{T}) where T = T

    # Akiyama–Tanigawa algorithm for second Bernoulli numbers B+n
    function bernoulli2k(T,p::Integer)
      B2 = Array{T}(undef,p+1)
      A = Array{BigInt}(undef,2p+1)
      fact = one(BigInt)
      for m = 0:2p
          A[m+1] = fact
          fact *= m+1
          for j = m:(-1):1
              A[j] = j*((m+1)*A[j]-A[j+1])
          end
          iseven(m) && (B2[div(m,2)+1] = T(A[1])/fact
          )
      end
      return B2
    end

    # in rationals
    const XMID = 1//4 # centre of domain in x direction
    const XWIDTH = 1//4 # width of domain in x direction
    const YMID = 0 # centre of domain in y direction
    const YWIDTH = 1//4 # width of domain in y direction

    sqrt3(T::Type) = sqrt(convert(T,3))
    const SQRT3 = sqrt3(TYPE)
    const PI = TYPE(π)
    const EULER = exp(one(TYPE))

    chebypoints(T::Type,N,mid=0,width=1) = mid .+ width * cos.((1:2:2N)*(T(π)/2N))
    chebypointsx(T::Type,N1) = chebypoints(T,N1,XMID,XWIDTH)
    chebypoints_even(T::Type,N,mid=0,width=1) = mid .+ width * cos.((1:2:2N)*(T(π)/4N))
    chebypointsy(T::Type,N2) = chebypoints_even(T,N2,YMID,YWIDTH)

    function lagrangepolynomials!(v::Vector{T},x::T,points::Vector{T},monomialcons::Vector{T},
            errortol=0) where T
        # gives errors and certainly won't play nice with validated numerics if x is too close to a point
        # I think we don't expect it though
        fullpoly = prod(2(x .- points)) # for Chebyshev points this is T_N(x)
        abs(fullpoly) < errortol && error("Evaluating Lagrange polynomial too close to a Lagrange point")
        for (n,p) in enumerate(points)
            @inbounds v[n] = fullpoly / (x - p) * monomialcons[n]
        end
        v
    end

    lagrangepolynomials_even!(v,x,points,monomialcons,errortol=0) = 
        lagrangepolynomials!(v,2x^2-1,points,monomialcons,errortol)

    # I think this is a nice optimised way of doing this so you don't have to keep allocating vectors but lol idk
    struct Lagrange2DPoly{T,S}
        V::Matrix{T}
        T1::Vector{T}
        T2::Vector{T}
        N1::Int
        N2::Int
        P1::Vector{T}
        P2::Vector{T}
        C1::Vector{T}
        C2::Vector{T}
        tol1::S
        tol2::S
        function Lagrange2DPoly(V::Matrix{T},T1::Vector{T},T2::Vector{T},N1::Int,N2::Int,
                P1::Vector{T},P2::Vector{T},C1::Vector{T},C2::Vector{T},
                tol1::S,tol2::S) where {T,S} # force checking that everything lines up
            @assert size(V,1)==length(T1)==length(P1)==length(C1)==N1
            @assert size(V,2)==length(T2)==length(P2)==length(C2)==N2
            new{T,S}(V,T1,T2,N1,N2,P1,P2,C1,C2,tol1,tol2)
        end
    end
    Lagrange2DPoly(T::Type,V::Matrix,N1=size(V,1),N2=size(V,2)) = Lagrange2DPoly(V,Array{T}(undef,N1),
                                    Array{T}(undef,N2),N1,N2,
                                    chebypoints(T,N1),
                                    chebypoints(T,N2),
                                    ((-1).^(0:N1-1)).*(sin.((1:2:2N1)*(T(π)/2N1))/2N1),
                                    ((-1).^(0:N2-1)).*(sin.((1:2:2N2)*(T(π)/2N2))/2N2),
                                    1e5*N1*eps(epstype(T)),
                                    1e5*N2*eps(epstype(T))
                                )

    @inline function (l::Lagrange2DPoly{T})(x,y,α) where T
        # evaluates matrix (α T_i(x) T_j(y))_{ij} for i < size(t)[1], j < size(t)[2]
        # !! if you change remember to change for Lagrange2DShifted
        lagrangepolynomials!(l.T1,x,l.P1,l.C1,l.tol1)
        lagrangepolynomials_even!(l.T2,y,l.P2,l.C2,l.tol2)
        # the rest of this is just α*(transpose(l.T1)*l.V*l.T2)
        val = zero(T)
        for (j,lxv) in enumerate(l.T2)
            valx = zero(T)
            for (i,lyv) in enumerate(l.T1)
                valx += lyv*(l.V[i,j])
            end
            val += lxv*valx
        end
        α*val
    end

    # Container that generates shifted polynomial via Lagrange interpolation
    struct Lagrange2DPolyShifted{T}
        l::Lagrange2DPoly{T}
        xmid::T
        xwidth::T
        ymid::T
        ywidth::T
    end
    Lagrange2DPolyShifted(T::Type,V::Matrix,N1=size(V,1),N2=size(V,2),xmid=T(XMID),xwidth=T(XWIDTH),ymid=T(YMID),ywidth=T(YWIDTH)) = 
        Lagrange2DPolyShifted(Lagrange2DPoly(T,convert.(T,V),N1,N2), xmid, xwidth, ymid, ywidth)

    @inline function (ls::Lagrange2DPolyShifted)(x,y,α=1)
        ls.l((x-ls.xmid)/ls.xwidth,(y-ls.ymid)/ls.ywidth,α)
    end


    function F0_nos3(x,y,N)
        # computes F0^{N√3}(x,y) and its |jacobian|^{-1}. Computing sqrt(3) every time will be slow
        # this also allows for x, y, N complex
        ξ = x-1
        UmNξ = 1 - N*ξ
        Ny = N*y
        denominator = UmNξ*UmNξ + Ny*Ny
        1 + (ξ*UmNξ - y*Ny) / denominator, y / denominator, denominator
    end
    F0(x::T,y::T,n,sqrt3=sqrt3(T)) where T = F0_nos3(x,y,n/sqrt3)

    function R(x::T,y::T,b::Int,sqrt3=sqrt3(T)) where T 
        # rotation in R^2 by exp(2πi/3*b) if b = ± 1\
        # the jacobian is = 1
        sqrt3b = sqrt3*b
         -(x + sqrt3b*y)/2, (sqrt3b*x - y)/2
    end

    function evaluate_branch(φ,s::T,x::T,y::T,n,α=1,sqrt3=sqrt3(T),jacnorm=false) where T
        # evaluate the transfer operator of G_{n,+} and G_{n,-} for fixed n 
        # for functions φ that take a multiplicative constant α as an argument
        # if jacnorm = true multiply by n^(2s)

        # note jiP, jiM, jin are the inverse of the jacobian
        xn, yn, jin = F0(x,y,n,sqrt3)

        FxP, FyP, jiP = F0(R(xn,yn,+1,sqrt3)...,1,sqrt3)
        FxM, FyM, jiM = F0(R(xn,yn,-1,sqrt3)...,1,sqrt3)

        if jacnorm # it's really jiP, jiM that need to be normalised but we do it here for Efficiency
            jin_npow = jin/n^2
        else
            jin_npow = jin
        end

        φ(FxP,FyP,α*pow_minus(jin_npow*jiP,s)) + φ(FxM,FyM,α*pow_minus(jin_npow*jiM,s))
    end
    
    ## BASIC CONSTANTS
    const N2 = ceil(Int,sup(mlogϵ/Rad/2)) # number of even Chebyshev modes in y direction
    const N1 = 2N2 # number of Chebyshev modes
                                    # Allows us to choose other parametrs in terms of number of basis functions
    const Nstar = ceil(Int,sup(ν+mlogϵ/2PI)) # branch to start the E-M formula at. N in the paper
    const K = floor(Int,sup((2PI*(Nstar-ν) - 1)/2))::Int;   # number of odd derivatives to take in the E-M formula
    const halfM = K                 # same scaling (~ mlogϵ) and we need M >= 2K anyway
    const M = 2halfM # number of points to use for the Taylor expansion to estimate derivatives in E-M formula
    const τ = (Nstar-ν)/exp(one(TYPE)) # radius for the above
    @assert M >= 2K
    @assert 2EULER*PI*(Nstar-ν) > 2K

    const halfMt = ceil(Int,sup(mlogϵ/log(Nstar/ν)/2))::Int
    const Mt = 2halfMt # tilde-M̃, number of points to use for the Taylor expansion to estimate the integral in E-M formula

    function empointwiseerror(s::T) where T
        Nmν = Nstar-ν
        νdN = ν/T(Nstar)

        sharedconst = V_Hrnorm * 2pow_plus(T(Wt2)/ν^2,s)

    #     ErrorR = T(Nmν)/K
    #     for p = 1:2K+1
    #         ErrorR *= p /(2PI*Nmν) # to compute the factorial without overflow
    #     end
        ErrorR = factorial(big(2K+1))/K/(2PI)^(2K+1)/(TYPE(Nmν))^(2K)

        ErrorD = PI^2*EULER/6 * Nmν / ((2EULER*Nmν*PI/(2K-1))^2 - 1) / (exp(TYPE(M)) - 1)

        ErrorI = 2Nstar*pow_plus(νdN,2s) / (2s-1) / (1-νdN) / (1/νdN^Mt - 1)

#        println(sharedconst, ErrorR, ErrorD, ErrorI)
        sharedconst * (ErrorR + ErrorD + ErrorI)
    end

    const EMError = begin err = empointwiseerror(s_allowed); hull(err,-err); end

    ## PRECOMPUTED CONSTANTS
    const taylorpoints = Nstar .+ τ*exp.(im*(PI*(1:2:2halfM-1)/M)) # z_m in the paper
    const FFTweights = [exp(-(im*PI*k*(2m-1))/M)/halfM for m=1:halfM, k=1:2:2K-1]
            # we only need to take a FFT on the upper half-plane for n because 
            # we are computing derivatives of a real-analytic function
    const EMderivativeweights = -bernoulli2k(TYPE,K)[2:end] ./ τ .^ (1:2:2K-1) ./ (2:2:2K)
    const taylorpointweights = FFTweights*EMderivativeweights # c_m in the paper

    const xpts = chebypointsx(TYPE,N1); 
    const ypts = chebypointsy(TYPE,N2);

    ## 
    const tsp = Lagrange2DPolyShifted(TYPE,V)
    const tsp_cplx = Lagrange2DPolyShifted(Complex{TYPE},V);

    const integraltaylorpoints = Nstar*exp.(im*(PI*(1:2:2halfMt-1))/Mt)
    const integralFFTweights = [exp((im*2PI*j*(2m-1))/2Mt)/halfMt for m=1:halfMt, j=0:Mt-1]

    integralweights(s) = pow_minus(convert(typeof(s),Nstar),2s-1) ./ (collect(0:Mt-1) .+ (2s - 1))
    integraltaylorpointweights(s) = integralFFTweights*integralweights(s)
                
    function transferaction(s::T,x::T,y::T,
        integraltaylorpointweights::Vector{Complex{T}}=integraltaylorpointweights(s)) where T

        @assert s ⊆ s_allowed

        L_slice = zero(T)        

            # add the derivatives (all in one go)
            # written as a linear operator straight from point evaluations
            for (z,weight) in zip(taylorpoints,taylorpointweights)
                L_slice += real(evaluate_branch(tsp_cplx,s,x,y,z,weight,SQRT3))
            end

            # 1/2 sum with the N
            L_slice += evaluate_branch(tsp,s,x,y,Nstar,one(TYPE)/2,SQRT3)

            # do the integral, again as a linear combination of point evaulations
            for (z,weight) in zip(integraltaylorpoints,integraltaylorpointweights)
                L_slice += real(evaluate_branch(tsp_cplx,s,x,y,z,weight,SQRT3,true))
            end


            # add branches naively up to Nstar
            for n = Nstar-1:-1:0
                L_slice += evaluate_branch(tsp,s,x,y,n,1,SQRT3)
            end
        L_slice + EMError
    end
end
  
function applytransfer(s::T) where T
    itpw = integraltaylorpointweights(s)
    @time convert(Matrix{TYPE},pmap((x,y)->transferaction(s,x,y,itpw),repeat(xpts,1,N2),repeat(ypts',N1,1)))
end

function minmax(s)
    Vout = applytransfer(s)::Matrix{TYPE}
    save("Vout.jld","sup",sup.(Vout),"inf",inf.(Vout))
    dct_Vout = dctmatx(N1)*Vout*dctmaty(N2)'
    
    (VN1, VN2) = size(V)
    matN1 = max(VN1,N1); matN2 = max(VN2,N2)
    dct_err = [dct_Vout zeros(N1, matN2-N2); zeros(matN1-N1, matN2)] - 
                [dct_V zeros(VN1, matN2-VN2); zeros(matN1-VN1, matN2)]
    
    minimaxerrorsize = sup(norm(dct_err[2:end],1) + # variation in Vout - V
                    H22norm(dct_V,rad)*CLerror(min(N1,2N2),Rad)*W # Chebyshev-Legendre L∞ error
                    )
    dct_err[1,1] + Interval(-minimaxerrorsize,minimaxerrorsize)#, Vout
end

function dimensionbound(s)
    minmaxest = minmax(s)
    
    φplus = dct_V[1] + norm(dct_V[2:end],1)
    φminus = dct_V[1] - norm(dct_V[2:end],1)

    # println("minmax = $minmaxest")
    # println("φ^+ = $φplus, φ^- = $φminus")

    s + hull(minmaxest.lo/(Dminus*φplus),minmaxest.hi/(Dplus*φminus))
end

function printlogfile(str,PREC=PREC) 
    # Write to file
    open("apollonian-rigorous-$PREC.log", "a") do io
       write(io, str*"\n")
   end;
end


printlogfile("ready to go",PREC)
dbound = dimensionbound(s_est)
printlogfile("Theorem: d_A ∈ [$(dbound.lo),",PREC)
printlogfile("                $(dbound.hi)]",PREC)
printlogfile("Width of bound: $(round(diam(dbound),sigdigits=2))",PREC)
printlogfile("",PREC)
