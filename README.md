# apollonian

Code for computing the Hausdorff dimension of the Apollonian circle packing.

The code is in two parts: a non-rigorous step where you find a dimension and eigenvalue estimate, and a rigorous step where you certify it.

The non-rigorous code takes one argument: the number of bits in the `BigFloat` precision. So far the highest we've run it is 448 bits (at which point memory on our research server started to be a problem). You run the code for NBITS-bit arithmetic as ` julia apollonian-nonrigorous.jl NBITS `.

This program finds a good dimension estimate iteratively using the secant method: it tells you what numbers it gets as it goes in `apollonian-nonrigorous-NBITS.log`.
Eventually it spits out a JLD file called `apollonian-nonrigorous-NBITS.jld` containing a bunch of relevant data including the dimension estimate and eigenfunction estimate.

Then to validate it, you make sure this file is in the same folder as `apollonian-rigorous.jl` and call `julia apollonian-rigorous.jl NBITS `.

This then does all the rigorously validated min-max stuff, and prints out a Theorem into `apollonian-rigorous-NBITS.log`.

With `NBITS=448`, you should get

```
Theorem: d_A ∈ [1.30568672804987718464598620685104089110602644149646829644618838899698642050296986454521612315053871328079246688242186910196730564360845303608397826,
                1.305686728049877184645986206851040891106026441496468296446188388996986420502969864545216123150538713280792466882421869101967305643746971829783186659]
         Width of bound: 1.4e-130
```

NB: this code is designed to be heavy duty: run in `BigFloat`, and in parallel (and the `addprocs` line in both code files tries to fill your CPU up with processes). If you want to use other kinds of float, change `const TYPE =` to something involving your favoured float type. If you don't want the overheads of parallelisation, remove all the distributed wrapping around the code.

If you use this, of course please cite our paper: https://arxiv.org/abs/2406.04922 (and let us know if you get any better results).

Caroline Wormell (@wormell) and Polina Vytnova (@Polevita)
