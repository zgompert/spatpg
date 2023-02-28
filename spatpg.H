#include <iostream>
#include <sstream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_sort.h>

// macro definitions
#define VERSION "1.0 --  June 2015"

extern gsl_rng * r;
// function declarations
using namespace std;

struct dataset{
  int nLoci;
  int nPops;
  int nGens;
  int nTotal;
  gsl_matrix_int * aCnts; // matrix of integer allele counts for the
			  // first allele; rows = SNPs, columns = pop
			  // x generation (same order as raw data)

  gsl_matrix_int * nCnts; // matrix of 2N; rows = SNPs, columns = pop
			  // x generation (same order as raw data)
  gsl_vector * envData; // environmental data vector for pop1,g1,g2,g3 pop2,g1,g2,g3
};

struct hdf5cont{
  hid_t file;
  hid_t datatype;
  hid_t dataspaceLTM, dataspaceLM, dataspaceTM;
  hid_t datasetP, datasetNe, datasetS, datasetAlpha, datasetBeta;
  hid_t locusVector, popVector, locusMcmcMatrix;

  // dimensions
  hsize_t dimsLTM[3], dimsLM[3], dimsTM[2];
  herr_t status; // not sure what this does, but it appears to be important

  hsize_t sr3[3]; /* Start of hyperslab */
  hsize_t sr2[2];
  hsize_t sr1[1];
  hsize_t cr3[3]; /* Block count */
  hsize_t cr2[2];
  hsize_t cr1[1]; 
  hsize_t br3[3]; /* Block size */
  hsize_t br2[2]; 
  hsize_t br1[1];
 
  // temporary storage gsl_vectors
  gsl_vector * auxLocusVector; 

};

class mcmc{
private:
  
  gsl_matrix * p; // allele frequencies (locus x pop and gen)
  gsl_vector * ne; // effective population size (pop x gen)
  gsl_vector * alpha; // intercept for selection coefficient
  gsl_vector * beta; // slope for selection coefficient
  gsl_matrix * s; // selection coefficient (locus x pop and gen)

  gsl_matrix * nemcmc; // matrix of stored effective sizes (pop x gen) x mcmc
  gsl_matrix * amcmc; // matrix of stored regression coefficients locus x mcmc
  gsl_matrix * bmcmc; // 
  gsl_matrix ** smcmc; // vector of matrixes for stored s; [pop x gen] x locus x mcmc
  gsl_matrix ** pmcmc; // vector of matrixes for stored p; [pop x gen] x locus x mcmc

  // priors
  double lbNe; // priors on Ne, upper and lower of U
  double ubNe;
  double sdSlabAlpha; // sd for gaussian slab
  double lbSpikeAlpha; // lower bounds of U high prob. region
  double ubSpikeAlpha; // upper bounds of U high prob. region
  double spikeMixAlpha; // proportion of prior from spike, roughly prob. no selection
  double sdSlabBeta; // sd for gaussian slab
  double lbSpikeBeta; // lower bounds of U high prob. region
  double ubSpikeBeta; // upper bounds of U high prob. region
  double spikeMixBeta; // proportion of prior from spike, roughly prob. no selection

  // proposal distributions parameters
  double propNe; // +- deviation for uniform Ne proposal
  double propP; // +- deviation for uniform Ne proposal
  double propAlpha; // sd for gaussian alpha proposal
  double propBeta; // sd for gaussian beta proposal

  // functions
  double neBoot(gsl_vector * num, gsl_vector * denom, dataset * data, int x, double nhat);
  void writeNe(dataset * data, hdf5cont * hdf5, int n);
  void sampleNeP(dataset * data, int N);
  void updatePs(dataset * data);
  void updateAlpha(dataset * data);
  void updateBeta(dataset * data);
  double calcp(double p0, double sel);
  double fixlogpr(double prob);

public:
  // functions
  mcmc(dataset * data, int n, int lb, int ub, double sdprior, double prop); // constructor function, initializes parameters for mcmc
  void updateNe(dataset * data, hdf5cont * hdf5, int N);
  void getNe(string neFile, dataset * data, hdf5cont * hdf5, int N);
  void updateP(dataset * data, int N);
  void storeP(dataset * data, hdf5cont * hdf5, int n);
  void writeP(dataset * data, hdf5cont * hdf5, int n);
  void update(dataset * data, int N);
  void store(dataset * data, hdf5cont * hdf5, int n);
  void write(dataset * data, hdf5cont * hdf5, int n);
};

// function definitions
void usage(char * name);
void getdata(string geneFile, string envFile, dataset * data);
void initHdf5(hdf5cont * hdf5, dataset * data, int mcmc, int burn, int thin);
