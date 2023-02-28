#include <iostream>
#include <sstream>
#include <fstream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_sf_gamma.h>
#include <float.h>
#include <math.h>
#include <hdf5.h>

#include "spatpg.H"

using namespace std;

// print software usage
void usage(char * name){
  fprintf(stdout,"\n%s version %s\n\n", name, VERSION);
  fprintf(stdout, "Usage: spatpg -g genefile -e envfile [options]\n");
  fprintf(stdout, "-g     Infile with allele frequency data in TreeMix format\n");
  fprintf(stdout, "-e     Infile with environmental covariate data\n");
  fprintf(stdout, "-f     (optional) Infile with known Ne\n");
  fprintf(stdout, "-o     Outfile for MCMC samples [outfile.hdf5]\n");
  fprintf(stdout, "-n     Number of MCMC steps [1000]\n");
  fprintf(stdout, "-N     Boolean, estimate Ne from the data [1]\n");
  fprintf(stdout, "-b     Number of MCMC steps to discard as a burnin [0]\n");
  fprintf(stdout, "-t     Thinning interval (integer), recore every nth step [1]\n");
  fprintf(stdout, "-l     Lower bound for uniform prior on Ne [20]\n");
  fprintf(stdout, "-u     Upper bound for uniform prior on Ne [4000]\n");
  fprintf(stdout, "-s     Standard deviation for prior on coefficients [0.1]\n");
  fprintf(stdout, "-p     Standard deviation of normal proposal distn. [0.05]\n");

  exit(1);
}

// ------ Functions for input and output ---------------

// read input from the infiles
void getdata(string geneFile, string envFile, dataset * data){
  int i, j, k;
  string line, element;
  ifstream infile;
  istringstream stream;
  string token;
  size_t pos = 0;
  string delim = ",";
  int p[2];

  // read allele frequency data, format similar to TreeMix
  infile.open(geneFile.c_str());
  if (!infile){
    cerr << "Cannot open file " << geneFile << endl;
    exit(1);
  }

  // read line with data dimensions
  getline(infile, line);
  stream.str(line);
  stream.clear();
  stream >> element; // number of populations
  data->nPops = atoi(element.c_str());  
  stream >> element; // number of generations
  data->nGens = atoi(element.c_str());
  data->nTotal = data->nPops * data-> nGens;// number of pops by gens
  stream >> element; // number of loci
  data->nLoci = atoi(element.c_str()); 

  // dynamic memory allocation for allele frequency data
  data->aCnts = gsl_matrix_int_calloc(data->nLoci, data->nTotal);
  data->nCnts = gsl_matrix_int_calloc(data->nLoci, data->nTotal);
  data->envData = gsl_vector_calloc(data->nTotal);

  // read and store allele frequencies
  for(i=0; i<data->nLoci; i++){
    getline(infile, line); // data for one locus
    stream.str(line);
    stream.clear();
    for(j=0; j<data->nTotal; j++){
      stream >> element;
      for(k=0; k<2; k++){// splits on comma
	pos = element.find(delim);
	token = element.substr(0,pos);
	p[k] = atoi(token.c_str());
	element.erase(0, pos + delim.length());
      }
      gsl_matrix_int_set(data->aCnts, i, j, p[0]);
      gsl_matrix_int_set(data->nCnts, i, j, p[0]+p[1]);
    }
  }
  infile.close();

  // print data to check
  // for(i=0; i<data->nLoci; i++){

  //   for(j=0; j<data->nTotal; j++){
  //     cerr << gsl_matrix_int_get(data->aCnts, i, j) << ",";
  //     cerr << gsl_matrix_int_get(data->nCnts, i, j) << "  ";
  //   }
  //   cerr << endl;
  // }
    

  // read environmental covariate, single column of values
  infile.open(envFile.c_str());
  if (!infile){
    cerr << "Cannot open file " << envFile << endl;
    exit(1);
  }
  // loop through file, one line per pop x gen
  for(j=0; j<data->nTotal; j++){
    getline(infile, line);
    stream.str(line);
    stream.clear();
    stream >> element;
    gsl_vector_set(data->envData, j, atof(element.c_str()));
  }
  infile.close();
  // for(j=0; j<data->nTotal; j++)
  //   cerr << gsl_vector_get(data->envData, j) << " ";
  // cerr << endl;

}

// setup hdf5 object
void initHdf5(hdf5cont * hdf5, dataset * data, int mcmc, int burn, int thin){
  int n;

  /* HDF5, create datasets */
  /*
   * Describe the size of the array and create the data space for
   * fixed size dataset.  I reuse dataspace and datatype, because they
   * are the same for each of the parameters.
   */
  // set dimensions
  n = (mcmc - burn) / thin;

  hdf5->dimsLTM[0] = data->nLoci;
  hdf5->dimsLTM[1] = data->nTotal;
  hdf5->dimsLTM[2] = n;

  hdf5->dimsLM[0] = data->nLoci;
  hdf5->dimsLM[1] = n;

  hdf5->dimsTM[0] = data->nTotal;
  hdf5->dimsTM[1] = n;

  // create dataspaces
  hdf5->dataspaceLTM = H5Screate_simple(3, hdf5->dimsLTM, NULL); // rank is 3
  hdf5->dataspaceLM = H5Screate_simple(2, hdf5->dimsLM, NULL); // rank is 2
  hdf5->dataspaceTM = H5Screate_simple(2, hdf5->dimsTM, NULL); // rank is 2

  // define datatype by copying an existing datatype, little endian double
  hdf5->datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
  hdf5->status = H5Tset_order(hdf5->datatype, H5T_ORDER_LE);

  // create datasets
  hdf5->datasetP = H5Dcreate2(hdf5->file, "p", hdf5->datatype, hdf5->dataspaceLTM,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hdf5->datasetNe = H5Dcreate2(hdf5->file, "ne", hdf5->datatype, hdf5->dataspaceTM,
		  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hdf5->datasetS = H5Dcreate2(hdf5->file, "s", hdf5->datatype, hdf5->dataspaceLTM,
		 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hdf5->datasetAlpha = H5Dcreate2(hdf5->file, "alpha", hdf5->datatype, hdf5->dataspaceLM,
		     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hdf5->datasetBeta = H5Dcreate2(hdf5->file, "beta", hdf5->datatype, hdf5->dataspaceLM,
		     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // create vectors to store parameters before writing
  hdf5->locusVector = H5Screate_simple(1, &hdf5->dimsLM[0], NULL);
  hdf5->popVector = H5Screate_simple(1, &hdf5->dimsTM[0], NULL);
  hdf5->locusMcmcMatrix = H5Screate_simple(2, &hdf5->dimsLM[0], NULL);

  // create temporary storage vectors to transfer from matrix to vector storage
  hdf5->auxLocusVector = gsl_vector_calloc(data->nLoci);
 
}
