using Distributed
addprocs(max(Base.Sys.CPU_THREADS-2,40))

### DISTRIBUTED PART ###
@eval @everywhere PREC = $(parse(Int,ARGS[1]))
# NOT TRUE FOR NOW SEE IF IT WORKS PROPERLY -> actually we do it in double the precision because I think there's some horrid source of numerical error
@eval @everywhere setprecision(BigFloat,PREC)
@everywhere begin open("test.log","a") do io; write(io,"$(precision(BigFloat))\n"); end; end
@everywhere begin
    using FFTW, LinearAlgebra

    epstype(::Type{T}) where T<:Real = T
    epstype(::Type{Complex{T}}) where T<:Real = T

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
    const TYPE = BigFloat
    const XMID = 1//4 # centre of domain in x direction
    const XWIDTH = 1//4 # width of domain in x direction
    const YMID = 0 # centre of domain in y direction
    const YWIDTH = 1//4 # width of domain in y direction

    sqrt3(T::Type) = sqrt(convert(T,3))
    const SQRT3 = sqrt3(TYPE)
    const PI = TYPE(π)
    
    const Rad = TYPE(14//10)
    const rad = TYPE(9//10)
    const ν = 10


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
    struct Lagrange2D{T,S}
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
        function Lagrange2D(T1::Vector{T},T2::Vector{T},N1::Int,N2::Int,
                P1::Vector{T},P2::Vector{T},C1::Vector{T},C2::Vector{T},
                tol1::S,tol2::S) where {T,S} # force checking that everything lines up
            @assert length(T1)==length(P1)==length(C1)==N1
            @assert length(T2)==length(P2)==length(C2)==N2
            new{T,S}(T1,T2,N1,N2,P1,P2,C1,C2,tol1,tol2)
        end
    end
    Lagrange2D(T::Type,N1,N2=div(N1,2)) = Lagrange2D(Array{T}(undef,N1),
                                    Array{T}(undef,N2),N1,N2,
                                    chebypoints(T,N1),
                                    chebypoints(T,N2),
                                    ((-1).^(0:N1-1)).*(sin.((1:2:2N1)*(T(π)/2N1))/2N1),
                                    ((-1).^(0:N2-1)).*(sin.((1:2:2N2)*(T(π)/2N2))/2N2),
                                    1e5*N1*eps(epstype(T)),
                                    1e5*N2*eps(epstype(T))
                                )

    @inline function (l::Lagrange2D)(x,y,α)
        # evaluates matrix (α T_i(x) T_j(y))_{ij} for i < size(t)[1], j < size(t)[2]
        # !! if you change remember to change for Lagrange2DShifted
        lagrangepolynomials!(l.T1,x,l.P1,l.C1,l.tol1) .* transpose(α*lagrangepolynomials_even!(l.T2,y,l.P2,l.C2,l.tol2))
    end

    # Container that generates shifted Chebyshev polynomials
    struct Lagrange2DShifted{T}
        l::Lagrange2D{T}
        xmid::T
        xwidth::T
        ymid::T
        ywidth::T
    end
    Lagrange2DShifted(T::Type,N1,N2=N1,xmid=T(XMID),xwidth=T(XWIDTH),ymid=T(YMID),ywidth=T(YWIDTH)) = 
        Lagrange2DShifted(Lagrange2D(T,N1,N2), xmid, xwidth, ymid, ywidth)

    @inline function (ls::Lagrange2DShifted)(x,y,α=1)
        # evaluates matrix (α T̂_i(x) T̂_j(y))_{ij} for i < size(t)[1], j < size(t)[2] 
        # where T̂ are the Chebyshev polynomials shifted according to Chebys2DShifted 
        lagrangepolynomials!(ls.l.T1,(x-ls.xmid)/ls.xwidth,ls.l.P1,ls.l.C1,ls.l.tol1) .* 
                    transpose(α * lagrangepolynomials_even!(ls.l.T2,(y-ls.ymid)/ls.ywidth,ls.l.P2,ls.l.C2,ls.l.tol2))
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

        φ(FxP,FyP,α*(jin_npow*jiP)^(-s)) + φ(FxM,FyM,α*(jin_npow*jiM)^(-s))
    end
    
    ## BASIC CONSTANTS
    const mlogϵ = log(TYPE(2))*PREC # precision of our approximation, we hope
    const N2 = ceil(Int,mlogϵ/Rad/2) # number of even Chebyshev modes in y direction
    const N1 = 2N2 # number of Chebyshev modes
                                    # Allows us to choose other parametrs in terms of number of basis functions
    const Nstar = ceil(Int,ν+mlogϵ/2PI) # branch to start the E-M formula at. N in the paper
    const K = floor(Int,(2PI*(Nstar-ν) - 1)/2);   # number of odd derivatives to take in the E-M formula
    const halfM = K                 # same scaling (~ mlogϵ) and we need M >= 2K anyway
    const M = 2halfM # number of points to use for the Taylor expansion to estimate derivatives in E-M formula
    const τ = (Nstar-ν)/exp(one(TYPE)) # radius for the above
    @assert M >= 2K
    @assert exp(one(TYPE))*2PI*(Nstar-ν) > 2K

    const halfMt = ceil(Int,mlogϵ/log(Nstar/ν)/2)
    const Mt = 2halfMt # M̃, number of points to use for the Taylor expansion to estimate the integral in E-M formula
    
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
    const ts = Lagrange2DShifted(TYPE,N1,N2) 
            # evaluating ts(x,y) gives you the matrix of Lagrange polynomials evaluated at (x,y)
    const ts_cplx = Lagrange2DShifted(Complex{TYPE},N1,N2) # same as ts but admits complex x,y

    const integraltaylorpoints = Nstar*exp.(im*(PI*(1:2:2halfMt-1))/Mt) # tilde-z_m in the paper
    const integralFFTweights = [exp((im*2PI*j*(2m-1))/2Mt)/halfMt for m=1:halfMt, j=0:Mt-1]
             # again we only need to take an FFT in the upper half-plane because we're looking at real-analytic functions
    integralweights(s) = Nstar^(1-2s) ./ ((0:Mt-1) .+ (2s - 1))
    integraltaylorpointweights(s) = integralFFTweights*integralweights(s) # tilde-c_m in the paper
    
    function transferaction(s::T,x::T,y::T,
        integraltaylorpointweights::Vector{Complex{T}}=integraltaylorpointweights(s)) where T

        L_slice = zeros(T,N1,N2)        

        # add the derivatives (all in one go)
        # written as a linear operator straight from point evaluations
        for (z,weight) in zip(taylorpoints,taylorpointweights)
            L_slice += real(evaluate_branch(ts_cplx,s,x,y,z,weight,SQRT3))
        end

        # 1/2 sum with the Nstar
        L_slice += evaluate_branch(ts,s,x,y,Nstar,one(TYPE)/2,SQRT3)

        # do the integral, again as a linear combination of point evaulations
        for (z,weight) in zip(integraltaylorpoints,integraltaylorpointweights)
            L_slice += real(evaluate_branch(ts_cplx,s,x,y,z,weight,SQRT3,true))
        end


        # add branches naively up to Nstar
        for n = Nstar-1:-1:0
            L_slice += evaluate_branch(ts,s,x,y,n,1,SQRT3)
        end
        L_slice
    end
end
### DISTRIBUTED PART ENDS HERE ###

function computeeigs(s)
    # Filling the transfer operator by rows in parallel 
    # i.e. the call we farm out to the worker nodes is
    # evaluating L_s[all the N1xN2 Lagrange-Chebyshev polynomials](x,y) for (x,y) Chebyshev nodes
    Lcontainer = pmap(transferaction,fill(s,N1,N2),repeat(xpts,1,N2),repeat(ypts',N1,1))
    L = Array{TYPE}(undef,N1,N2,N1,N2)
    for j = 1:N2
        for i = 1:N1
            L[:,:,i,j] = Lcontainer[i,j]
        end
    end
    Lm = reshape(reshape(L,N1*N2,N1,N2),N1*N2,N1*N2)'; # reshape to be a N1*N2 x N1*N2 matrix


    # Power method for leading eigenvector/eigenvalue
    # The leading eigenvalues of L_s are approximately 1, -0.25, 0.09...
    # The leading eigenvalues of (14L_s + I)/15 are approximately 1, -0.172, 0.168...
    # so the larger spectral gap of the latter means we get a bit of a speedup
    eigvec = fill(one(TYPE),N1*N2)
    Iterations = ceil(Int,mlogϵ/-log(0.18)+5)
    eigval_shifted = zero(TYPE)
    for i = 1:Iterations
        axpy!(14,Lm*eigvec,eigvec) # i.e. eigvec = 14(Lm*eigvec) + eigvec
        eigval_shifted = norm(eigvec,1)/(N1*N2) # we can use a norm because the eigenvalue is positive
        eigvec /= eigval_shifted
    end
    eigval = (eigval_shifted-1)/14
    
    eigval-1, eigvec
end


using JLD

function printlogfile(str,PREC=PREC) 
    # We'll write a lot of output to apollonian-nonrigorous.log, opening and closing when we do it
    # This is better than piping output to standard output as writing to standard output is often
    # delayed so you don't know what your program is doing
    # Plus we don't call this very often so the file stuff overhead doesn't matter
    open("apollonian-nonrigorous-$PREC.log", "a") do io
       write(io, str*"\n")
   end;
end


# Using the secant method
# i.e. iteratively estimate root of f by linearly interpolating f(x_{n-1}), f(x_{n-2})
function secant(f,est0,est1,args...)
    @time err1, _ = f(est1,args...);
    printlogfile("PREC = $PREC, s = $est1, err = $err1 (first guess)",PREC)
    @time err0, eigvec = f(est0,args...);
    printlogfile("PREC = $PREC, s = $est0, err = $err0 (second guess)",PREC)

    save("apollonian-nonrigorous-$PREC-working.jld","s",est0,"eigval_err",err0,"eigvec",reshape(eigvec,N1,N2),
                        "mlogϵ",mlogϵ,"floatingpointprec",precision(BigFloat)) ## NOTE code repeated below

    besterr = TYPE(Inf)
    for i = 1:log((1+sqrt(5))/2,log(2)*PREC)+5 
                    # Newton's method has quadratic convergence but this has golden mean-ic convergence
        est0, est1, err1 = est0 - err0*(est0-est1)/(err0-err1), est0, err0
        @time err0, eigvec = f(est0,args...);
        printlogfile("PREC = $PREC, s = $est0, err = $err0",PREC)
        (1.30 < est0 < 1.31) || break # otherwise we're in garbage zone
        
        if abs(err0) < besterr
            besterr = abs(err0)
            save("apollonian-nonrigorous-$PREC-working.jld","s",est0,"eigval_err",err0,"eigvec",reshape(eigvec,N1,N2),
                        "mlogϵ",mlogϵ,"floatingpointprec",precision(BigFloat)) ## NOTE code repeated above
        end
        if abs(err0) <= abs(err1) && log(2,abs(err1)) < PREC*2/(1+sqrt(5))
	    break
        end
    end
end

# Let's gooooooo
printlogfile("Ready to go: precision = $PREC",PREC)
printlogfile("mlogϵ = $mlogϵ, -log eps(BigFloat) = $(-log(eps(BigFloat)))",PREC)
secant(computeeigs,
    BigFloat("1.3056867280498771846459862068510408911060264414964682964461883889969864205029698645452161231505387132807924668824226"),
    BigFloat("1.30568672804987718464598620685104089110602644149646829644618838899698642050296986454521612315053871")-1e4eps(BigFloat)
    )
printlogfile("Done :-)",PREC)
