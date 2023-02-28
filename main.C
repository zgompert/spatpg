// file: main.C for SPATpg

// Program to estimate variance effective population size and
// selection coefficients from spatio-temporal popoulation genetic
// data (allele frequencies). Assumes independence across loci and no
// gene flow among populations.

// Infile similar to TreeMix format:
// Row 1: number of populations; number of generations; number of SNPs
// Row 2-N SNPs: sample allele frequency for each population and generation, e.g.
// #p1,#g1 #p1,#g2, #p1,#g3, ... #p10,#g10


// Time-stamp: <Friday, 17 August 2012, 14:46 CDT -- zgompert>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <getopt.h>
#include <hdf5.h>
#include <omp.h>
#include "spatpg.H"

using namespace std;

gsl_rng * r;  /* global state variable for random number generator */

/* ----------------- */
/* beginning of main */
/* ----------------- */

int main(int argc, char *argv[]) {

  time_t start = time(NULL);
  time_t end;
  int rng_seed = 0;
  int ch = 0;
  int burn = 0, chainLength = 1000, thin = 1;
  int lb = 20, ub = 4000;
  double prop = 0.05, sigprior = 0.1;
  int n;

  int estNe = 1; // boolean estimate Ne from the data [1] or read Ne from a file [0]

  string geneFile = "undefined";
  string envFile = "undefined";
  string neFile = "undefined";
  char * outFile = (char *) "outfile.hdf5";

  dataset data;
  hdf5cont hdf5;

  // get command line arguments
  if (argc < 2) {
    usage(argv[0]);
  }
  
  while ((ch = getopt(argc, argv, "g:e:o:n:b:t:l:u:p:s:N:f:")) != -1){
    switch(ch){
    case 'g':
      geneFile = optarg;
      break;
    case 'e':
      envFile = optarg;
      break;
    case 'o':
      outFile = optarg;
      break;
    case 'n':
      chainLength = atoi(optarg);
      break;
    case 'b':
      burn = atoi(optarg);
      break;
    case 't':
      thin = atoi(optarg);
      break;
    case 'l':
      lb = atoi(optarg);
      break;
    case 'u':
      ub = atoi(optarg);
      break;
    case 'p':
      prop = atof(optarg);
      break;
    case 's':
      sigprior = atof(optarg);
      break;
    case 'N':
      estNe = atoi(optarg);
      break;
    case 'f':
      neFile = optarg;
      break;
    case '?':
    default:
      usage(argv[0]);
    }
  }
  
  // set up gsl random number generation 
  gsl_rng_env_setup();
  r = gsl_rng_alloc (gsl_rng_default);
  srand(time(NULL));
  rng_seed = rand();
  gsl_rng_set(r, rng_seed); /* seed gsl_rng with output of rand, which
                               was seeded with result of time(NULL) */

  // read infiles, record genotype likelihoods and data dimensions
  cout << "Reading input from files: " << geneFile << " and " << 
    envFile << endl;
  getdata(geneFile, envFile, &data);

  // set-up hdf5 object 

  // Create a new file using H5F_ACC_TRUNC access, default file
  // creation properties, and default file access properties.
  hdf5.file = H5Fcreate(outFile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  cout << "Initializing model paramters" << endl;
  initHdf5(&hdf5, &data, chainLength, burn, thin);

  // initialize parameters for mcmc
  mcmc chain = mcmc(&data,(chainLength - burn)/thin, lb, ub, sigprior, prop);
  
  // use Bayesian bootstrap to obtain posterior for Ne
  if(estNe==1){
    cout << "Estimating effective population sizes" << endl;
    chain.updateNe(&data,&hdf5,(chainLength-burn)/thin);
  }
  // read known Ne from file
  else{
    cout << "Reading effective population sizes from " << neFile << endl;
    chain.getNe(neFile,&data,&hdf5,(chainLength-burn)/thin);
  }
    
  // repeate MCMC to obtain samples of alpha, beta, and S
  cout << "Estimating selection coefficients (* = 1000 steps)" << endl;

  for(n=0; n<chainLength; n++){
    chain.update(&data,(chainLength-burn)/thin);
    if((n>=burn) && ((n-burn)%thin == 0)){
      chain.store(&data, &hdf5, (n-burn)/thin);
      chain.storeP(&data, &hdf5, (n-burn)/thin);
    }
    if(n%5000 == 0)
      cout << endl;
    if(n%1000 == 0)
      cout << "*";

  }
  cout << endl;
  cout << "Writing final results" << endl;
  chain.writeP(&data, &hdf5, (chainLength-burn)/thin);
  chain.write(&data, &hdf5, (chainLength-burn)/thin);

  H5Fclose(hdf5.file);

  // prints run time
  end = time(NULL);
  cout << "Runtime: " << (end-start)/3600 << " hr " << (end-start)%3600/60 << " min ";
  cout << (end-start)%60 << " sec" << endl;
  return 0;
}
