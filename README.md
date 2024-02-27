# PSMCplus

A new implementation of Li and Durbin's pairwise sequentially Markovian coalescent ([PSMC](https://pubmed.ncbi.nlm.nih.gov/21753753/)) that accounts for heterogeneity across the genome. 

## Installation

PSMC+ is written in Python. You will need numpy, numba, pandas, joblib, scipy, psutil and matplotlib. You can install these in a `conda` environment with:

```
conda create --name PSMCplus
conda activate PSMCplus
conda install numpy numba pandas joblib scipy psutil matplotlib
git clone https://github.com/trevorcousins/PSMCplus.git
```

To check whether the installation worked, you can run with some test data:<br>

```python /path/to/installation/PSMCplus/PSMCplus.py -in /path/to/installation/PSMCplus/simulations/constpopsize.mhs -D 10 -its 1 -o testing```

This should save an output file to `testingfinal_parameters.txt`, and print log information to stdout. The `/path/to/installation/` should be changed to the path from which you performed the `git clone` operation. 

If you are having problems, see the Troubleshooting section. 

## Quick start

### Input files

PSMC+ takes multi-hetsep (mhs) files as introduced by Stephan Schiffels. To generate these files, you can use [his tutorial](https://github.com/stschiff/msmc-tools/blob/master/msmc-tutorial/guide.md). If you have a CRAM/BAM file, you can use my [Snakefile](https://github.com/trevorcousins/PSMCplus/blob/master/Snakefiles/processing_data/Snakefile) as guide for how to process this into mhs files. 

### Inference of population size history

You can run PSMC+ with the following command line: 

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o <outprefix> | tee <logfile>```

`D` is the number of discrete time interval windows, `b` is the genomic bin size, and `its` is the number of iterations (see the Advanced section for a more detailed explanation). `<infiles>` is a string (separated by a space if more than one) that points to the mhs files. The inferred parameters will be saved to `<outprefix>final_parameters.txt` and a log file will be saved to `<logfile>`. The output file contains `D` rows and 3 columns, which are the left time boundary, the right time boundary, and the coalescence rate per discrete time window. To scale the times into generations, you must divide by `mu`. We get the effective population size by taking the inverse of the coalescence rate and dividing this by `mu`. 

You can plot this in Python using the following code:
```
final_parameters_file = "PATH_TO_PSMCPLUS_OUTPUT"
final_params = np.loadtxt(final_parameters_file)
time_array = list(final_params[:,1])
time_array.insert(0,0)
time_array = np.array(time_array)
plt.stairs(edges=(time_array/mu)*gen,values=(1/final_params[:,2])/mu,label=population,linewidth=4,linestyle="solid",baseline=None)
plt.xscale('log')
plt.ylabel('$N_e(t)$')
plt.xlabel('Years')
```   

Alternatively, you can plot this in R using the following code:
```
mu = 1.25e-8
gen = 30
final_parameters_file<-"PATH_TO_PSMCPLUS_OUTPUT"
inference<-read.table(final_parameters_file, header=FALSE)
time = inference[,1]/mu*gen
N = (1/inference[,3])/mu
plot(time,N,log="x",type="n",xlab="Years",ylab="Time")
lines(time, N, type="s", col="red")
```

See the [Inference Tutorial notebook](https://github.com/trevorcousins/PSMCplus/blob/master/tutorial/Inference_tutorial.ipynb) for a specific example, and the Advanced section for a more detailed explanation of the hyperparameters. 

### Decoding 

You can decode the HMM (that is, get the inferred coalescence times across the genome) with: 

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infile> -D <D> -o <outprefix> -decode -decode_downsample <decode_downsize> | tee <logfile.txt>```

The argument `-decode` tells PSMC+ to decode the HMM, as opposed to inferring the $N_e$ parameters. The decoding file is large - you can reduce disc space by saving the posterior probabilities only at every X base pairs with the `-decode_downsample` argument, where `X = b*decode_downsize` (`b` is the genomic binsize, see above or Advanced). For example, if `b=100` and you want to save every the posterior every 10kb, you should do `-decode_downsample 100` (as `b*decode_downsample = 100*100 = 10000`). I recommend that when decoding you provide the command line with the inferred inverse coalescence rate parameters, as assuming an incorrect demography can induce bias. See `-lambda_A_fg` in the Advanced section for more information.  See the [Tutorial](https://github.com/trevorcousins/PSMCplus/blob/master/tutorial/Inference_tutorial.ipynb) for an example.

For the output file, the first row is the position, and the remaining rows are the probability of coalescing in each time window (of which there are `D`), with the first row being the most recent and the last row the most ancient. The time windows are in coalescent units (scale to years with `2*N_0*gen`), and the boundaries are detailed in the comment at the top of the file. See the [Inference Tutorial](https://github.com/trevorcousins/PSMCplus/blob/master/tutorial/Inference_tutorial.ipynb) for parsing this file. 

Note that if you want *extremely* fast decoding of pairwise coalescence times, the current state of the art method is Regev Schweiger's `gamma-smc` ([paper](https://genome.cshlp.org/content/33/7/1023/suppl/DC1) [code](https://github.com/regevs/gamma_smc)).

# Advanced

Here is a more in depth explanation of the hyperparameters for PSMC+. Quick preliminaries: PSMC+ works in coalescent units, so uses `theta` and `rho`. `theta` is the scaled mutation rate and is equal to `4*N*mu` where `mu` is the rate per generation per base; `rho` is the scaled recombination rate and is equal to `4*N*r` where `r` is the rate per generation per neighbouring base pairs. In the previous sentence, `N` is the "long-term effective population size" and can be thought of as the mean of changes in population size over time. The inferred parameters are scaled to time in years by a user-provided mutation rate per base pair per generation (`mu`) and generation time (`gen`), and to real units of `N` also by `mu`. 

-D<br>
The number of discrete time intervals. 

Default behaviour is to use `D=32`.

-o<br>
The output prefix (if inferring $N_e$), or the output file (if decoding). 

If you are inferring $N_e$, and you set `-o /path/to/installation/PSMC+_inference/Ne_` then the file describing the inferred parameters will be saved to `/path/to/installation/PSMC+_inference/Ne_final_parameters`. Default behaviour is to save to the current working directory. 
If you are decoding and you set `-o /path/to/installation/PSMC+_inference/TMRCA.txt.gz`, then the matrix of posterior distributions will be saved to `/path/to/installation/PSMC+_inference/TMRCA.txt.gz`. You must give this argument if decoding. 

-theta<br> 
The scaled mutation rate, `theta = 4*N*mu`. 

Default behaviour is to calculate this from the data with theta=number_hets/(sequence_length-number_masked_bases). It is not subsequently updated as part of the expectation-maximisation (EM) algorithm. Usage: if you instead want to give a particular value of theta, e.g. 0.001, you can do: 

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o <outprefix> -theta 0.001 | tee <logfile>```

-mu_over_rho_ratio <br>
The ratio of the mutation rate to the recombination rate (this parameter should really be called "theta_over_rho_ratio").

If you know the ratio of the mutation to recombination rate, then the starting value of rho is `rho=theta/mu_over_rho_ratio`. 
Default behaviour is to set `mu_over_rho_ratio=1.5`.

In humans, the genome wide average rate of `mu` and `r` are currently taken to be 1.25e-8 and 1e-8, respectively. This ratio will not be true in all species and it requires a bit of thought. Note that if `r>mu` then it is not recommended to use PSMC, as the algorithm is not able to accurately infer the coalescence times because a typical stretch of the genome will experience a recombination before a mutation. If there are no mutations on a genomic segment then there is no way for its age to be estimated.

Usage: if you want to give a particular value of `mu_over_rho_ratio`, e.g. 2, you can do: 

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o <outprefix> -mu_over_rho_ratio 2 | tee <logfile>```

-rho<br>
The scaled recombination rate, `rho = 4*N*r`. 

Default behaviour is to set `rho=theta/mu_over_rho_ratio` (see above), then update rho every iteration as part of the EM algorithm. Useage: if you have a particualrly good idea about what rho is, e.g. 0.0005, you can give it with

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o <outprefix> -rho 0.0005 | tee <logfile>```

-rho_fixed<br>
Do not infer rho as part of the EM algorithm. 

By default rho is updated as part of the EM algorithm. This has been shown not to be particularly accurate, but historically has been standard practise. If you do not want to update rho, you can add the argument `-rho_fixed` which will force rho to remain at its starting value. Usage: 

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o <outprefix> -rho_fixed | tee <logfile>```

-b <br>
Genomic bin size. 

You can bin the genome into windows of size `b`, which enforces the assumption that recombinations can occur only between windows. The primary advantage of doing this is speeding up the code by a factor of `b`, and also reducing RAM by a factor of `b`. Default behaviour is `b=100`. In humans, `b=1` and `b=100` seem to be indistinguishable, and one could probably get away with using even `b=200` or possibly even greater. Usage: if you want to use a bin size of 100 you can do:

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b 100 -its <its> -o <outprefix> -rho_fixed | tee <logfile>```

-lambda_segments<br>
Fix some adjacent time windows to have the same coalescence rate.

You may want to force adjacent intervals to have the same coalescent rates, as in PSMC and MSMC/MSMC2, though I don't recommend this. For the undeterred, if for example you want to use 64 discrete time windows with the first 4 fixed to be the same, the next 20 in pairs of two, then the final 20 intervals to be free you can do: 

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o <outprefix> -lambda_segments 1*4,20*2,20*1 | tee <logfile>```

so the argument is parsed as "1 lot of four fixed intervals, 20 lots of two, and 20 lots of one". If you don't want to infer parameters for some time windows, instead leaving it at its starting value, you can give it a "0". For example if you want 32 time windows with the first 10 to be fixed at their starting value, you can do: 

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o <outprefix> -lambda_segments 10*0,22*1 | tee <logfile>```

Default behaviour is to assume all intervals are free.

-lambda_A_fg<br>
The first guess for the inverse coalescence rates, in each discrete time interval. 

A comma separated list of floats that are taken are used as the starting guess for the inverse coalescence rates. You should also use this to decode with the inferred population size parameters. The length of this list must be equal to the number of segments as specified by the `-lambda_segments` argument. If you have 10 discrete time intervals and know your inverse coalescence rates are [1,1,5,5,5,5,5,1,1,1] (a population that experiences a five-fold bottleneck) you can do:

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o inference/final_parameters.txt -lambda_A_fg 1,1,5,5,5,5,5,1,1,1 | tee <logfile>```

Default behaviour is to assume `lambda_A_fg` is 1 everywhere. 
See the [Inference Tutorial](https://github.com/trevorcousins/PSMCplus/blob/master/tutorial/Inference_tutorial.ipynb) for more information on this, especially for the decoding. 

-its <br>
Number of iterations of the EM algorithm. 

Default behaviour is `its=20`.

-thresh  <br>
Stop iterating the EM algorithm after the change in log-likelihood is less than thresh. 

Typically, PSMC or MSMC/MSMC2 are run for 20 or 30 iterations. Heng Li demonstrates that this is sufficient (atleast in humans) for a satisfactory "goodness of fit" (see Figure S12 in the original [paper](https://pubmed.ncbi.nlm.nih.gov/21753753/)), and for recovering $N_e$ parameters. However you might wish for a more specific convergence criteria, such as the change in log-likelihood of the EM algorithm to be less than a particular value: you can achieve this with `thresh`. 
Default behaviour is to run for 30 iterations - if both `its` and `thresh` are given, then the algorithm will terminate when the first of either is satisfied. Usage: if you want a `thresh` of 1 then you can do 

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o <outprefix> -thresh 1 | tee <logfile>```

-spread_1, spread_2<br>
The spread of the discrete time window boundaries. 

The time window boundaries are given by the following equation: <br>
$\tau_i = $\omega exp\left( \frac{i}{D}log\left(1+\frac{\psi}{\omega}\right)-1\right)$<br>
Where $\omega$ = spread_1 controls the dispersion of intervals in recent time, and $\psi$ = spread_2 controls the dispersion in ancient time. In humans, I recommend `spread_1=0.05` and `spread_2=50`, which spans ~10kya to ~5Mya. This may not be optimal in other species and it is probably a good idea to experiment with different combinations. 
Default behaviour is to set `spread1=0.05` and `spread2=50`. Usage: if you want `spread_1=0.05` and `spread_2=50`

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -D <D> -b <b> -its <its> -o <outprefix> -spread_1 0.05 -spread_2 50 | tee <logfile>```

## Simulation

If you want to simulate a panmictic demography, you probably want to use msprime or SLiM as these are extremely powerful and flexible. However, I also provide functionality to simulate directly from the PSMC HMM (this is necessarily simulating from the SMC' model, which is marginally a very good approximation to the full coalescent with recombination - see [Wilton et al 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4423375/)). An example command line to simulate a constant population size is:

```python /path/to/installation/PSMCplus/simulate_HMM.py -D 10 -theta 0.001 -rho 0.0005 -o_mhs simulations/sim1_variants.mhs -o_coal simulations/sim1_coal.txt.gz -spread_1 0.1 -spread_2 50 -L 1000000```

`D`, `theta`, `rho`, `b`, `spread_1` and `spread_2`, are as described above. `L` is an integer for the sequence length. 
This will save an mhs file detailing the variant positions to `simulations/sim1_variants.mhs`, and a file detailing the coalescent data (pairwise coalescence times across the genome) to `simulations/sim1_coal.txt.gz`. The third column of this file is the index of the coalescent time between the genomic positions described in the first and second column. 

To simulate with a changing population size, you must give an argument `-lambda_A`, which is a comma separated string where each value is the inverse coalescence rate in each time interval. For example if you want a simulation with 32 discrete time windows with a population of relative size 1 that doubles in size between time index 10 and 20, then you can do: 

```python /path/to/installation/PSMCplus/simulate_HMM.py -D 32 -lambda_A 1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1 -theta 0.001 -rho 0.0005 -o_mhs simulations/sim2_variants.mhs -o_coal simulations/sim2_coal.txt.gz -L 1000000```

You can visualise the desired demography and parameters from the simulation in the [Simulation Tutorial](https://github.com/trevorcousins/PSMCplus/blob/master/tutorial/Simulation_tutorial.ipynb).


## Varying rates across the genome

### Local coalescence rate variation

PSMC+ enables a user to provided a map of varying coalescence rates across the genome. The local coalescent rate may change either do to selection or mutation rate variation. The map that PSMC+ requires is bed file where the 1st column is chrom, 2nd column is start position, 3rd column is end position, 4th column is the rate between the start and end (2nd and 3rd column, respectively). The rate is given in units of the mean, e.g. if your mutation rate looks like this:
```
chr1    0   100 1e-08
chr1    100 200 1e-08
chr1    200 300 2e-08
chr1    300 400 0.5e-08
chr1    400 500 1e-08
```
Then you should give PSMC+ the following (divide the fourth column by its mean, adjusting for length of starts and stops):
```
chr1    0   100 1
chr1    100 200 1
chr1    200 300 2
chr1    300 400 0.5
chr1    400 500 1
```

You can feed this to PSMC+ with the `-in_M` argument:

```python /path/to/installation/PSMCplus/PSMCplus.py -in <infiles> -in_M <rate_maps> -D <D> -b <b> -its <its> -o <outprefix> | tee <logfile>```

where `<rate_maps>` is a string or list of strings that corresponds directly to the input mhs files in `<infiles>`. As a particular example if you have input files `/data/chr1.mhs /data/chr2.mhs /data/chr3.mhs` and rate files `/data/chr1_ratemap.bed /data/chr2_ratemap.bed /data/chr3_ratemap.bed` you would do:

```python /path/to/installation/PSMCplus/PSMCplus.py -in /data/chr1.mhs /data/chr2.mhs /data/chr3.mhs -in_M /data/chr1_ratemap.bed /data/chr2_ratemap.bed /data/chr3_ratemap.bed -D <D> -b <b> -its <its> -o <outprefix> | tee <logfile>```

If there is a large discrepancy between the mhs and corresponding rate file, you will get an error. The functionality is exactly the same for the decoding. See the [Tutorial](https://github.com/trevorcousins/PSMCplus/blob/master/tutorial/Inference_tutorial.ipynb) for more information and a particular example.

## Varying recombination rates

Incoming... 

## Troubleshooting

### Installation
We use PSMC+ with the following versions of each package: 
```
python 3.10.5
numba 0.55.2
joblib 1.1.0
matplotlib 3.5.2
pandas 1.4.3
psutil 5.9.1
scipy 1.8.1
```
If your installation attempt fails, please try with versions listed above.
In the above commands you will of course need to change the `/path/to/installation/` path. For a particular example, if I am in my home directory (`/home/trevor`) and do `git clone https://github.com/trevorcousins/PSMCplus.git`, then the code is ready to run at `/home/trevor/PSMCplus/PSMCplus.py` . 


### Division by Zero error
If your heterozygosity is very high (e.g. `theta>0.01`), the default binsize of 100 will not be appropriate and you'll get an error. Instead, reduce the binsize such that there being more than ~3 heterozygotes per bin is rare. E.g. if theta=0.05 then we expect a heterozygous position every 20 base pairs on average, so a suitable bin size is 5 or 10 (add `-b 10` or `-b 20` to your command line). [See here](https://github.com/trevorcousins/PSMCplus/issues/2)

If you are still having problems, please submit a new issue.

## Citation

If you use PSMC+, please cite the following [paper](https://www.biorxiv.org/content/10.1101/2024.01.18.576291v1): 

* Cousins, T., Tabin, D., Patterson, N., Reich, D. and Durvasula, A., 2024. Accurate inference of population history in the presence of background selection. bioRxiv, pp.2024-01.
