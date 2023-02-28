#include <iostream>
#include <sstream>
#include <fstream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_sort.h>
#include <float.h>
#include <math.h>
#include <hdf5.h>

#include "spatpg.H"

using namespace std;

// constructor function for class mcmc, initializes the chain
mcmc :: mcmc(dataset * data, int n, int lb, int ub, double sdprior, double prop){
  int i, j, t;
  double x, a, b;
  double sel;

  // proposal distribution paramters
  propAlpha = prop;
  propBeta = prop;

  //prior parameters
  ubNe = lb;
  lbNe = ub;
  sdSlabAlpha = sdprior;
  sdSlabBeta = sdprior;
  lbSpikeAlpha = 0;// spike and slab not used
  ubSpikeAlpha = 0;
  lbSpikeBeta = 0;
  ubSpikeBeta = 0;
  spikeMixAlpha = 0;
  spikeMixBeta = 0;

  // allocate dynamic memory for parameters
  p = gsl_matrix_calloc(data->nLoci, data->nTotal);
  ne = gsl_vector_calloc(data->nTotal);
  alpha = gsl_vector_calloc(data->nLoci);
  beta = gsl_vector_calloc(data->nLoci);
  s = gsl_matrix_calloc(data->nLoci, data->nTotal);

  nemcmc = gsl_matrix_calloc(data->nTotal, n);
  amcmc = gsl_matrix_calloc(data->nLoci, n);
  bmcmc = gsl_matrix_calloc(data->nLoci, n);
  
  pmcmc = new gsl_matrix * [data->nTotal];
  smcmc = new gsl_matrix * [data->nTotal];
  for(j=0; j<data->nTotal; j++){
    pmcmc[j] = gsl_matrix_calloc(data->nLoci, n);
    smcmc[j] = gsl_matrix_calloc(data->nLoci, n);
  }

  // initialize population allele frequencies stochastically,
  // expectations from sample allele frequencies
  for(i=0; i<data->nLoci; i++){
    for(j=0; j<data->nTotal; j++){
      a = gsl_matrix_int_get(data->aCnts, i, j) + 0.5;
      b = gsl_matrix_int_get(data->nCnts, i, j) - gsl_matrix_int_get(data->aCnts, i, j) + 0.5;
      x = gsl_ran_beta(r, a, b);
      gsl_matrix_set(p, i, j, x);
    }
  }

  // selection parameters initialized near 0
  for(i=0; i<data->nLoci; i++){
    gsl_vector_set(alpha, i, gsl_ran_gaussian(r, 0.02));
    gsl_vector_set(beta, i, gsl_ran_gaussian(r, 0.02));
    for(j=0; j<data->nPops; j++){
      for(t=0; t<data->nGens; t++){
	x = t+j*data->nGens; // get index
	sel = gsl_vector_get(alpha, i) + gsl_vector_get(beta, i) * 
	  gsl_vector_get(data->envData, x);
	gsl_matrix_set(s, i, x, sel);
      }
    }
  }
}

// estimate Ne using unbiased estimator and bayesian bootstrap
void mcmc :: updateNe(dataset * data, hdf5cont * hdf5, int N){
  int j, t, i, x;
  int y, n0, n1, n;
  int br;
  double p0, p1;
  double a, b, z;
  double Fs, Fsprime;

  gsl_vector * Fsnum;
  gsl_vector * Fsdenom;
  Fsnum = gsl_vector_calloc(data->nLoci);
  Fsdenom = gsl_vector_calloc(data->nLoci);

  // loop over generation pairs and populations
  for(j=0; j<data->nPops; j++){
    for(t=0; t<(data->nGens-1); t++){
      x = t+j*data->nGens; // get index
      n = 0; // cnt for mean sample size across SNPs
      for(i=0; i<data->nLoci; i++){
	n0 = gsl_matrix_int_get(data->nCnts, i, x);
	n1 = gsl_matrix_int_get(data->nCnts, i, x+1);
	n += n0;
	if(n0 >= 1 && n1 >= 1){ // data for this locus
	  y = gsl_matrix_int_get(data->aCnts, i, x);
	  p0 = (double) y/n0;
	  y = gsl_matrix_int_get(data->aCnts, i, x+1);
	  p1 = (double) y/n0;
	  a = gsl_pow_2(p1 - p0);
	  z = (p1 + p0)/2.0;
	  b = z * (1 - z);
	}
	else{
	  a = 0;
	  b = 0;
	}
	gsl_vector_set(Fsnum, i, a);
	gsl_vector_set(Fsdenom, i, b);
      }
      // data for one or more loci
      if(gsl_stats_mean(Fsdenom->data, 1, data->nLoci) > 0){
	n = (double) n/data->nLoci;
	Fs = gsl_stats_mean(Fsnum->data, 1, data->nLoci)/
	  gsl_stats_mean(Fsdenom->data, 1, data->nLoci);
	a = Fs * (1 - (double) 1.0/(2.0 * n)) - (double) 2.0/n;
	b = (1 + Fs/4) * (1 -1/n);
	Fsprime = a/b;
	// uncomment to print point-estimates, prior does not apply to these
	//cout << "Unbiased estimate of Ne = " << 1/(2*Fsprime) << endl;
	for(br=0; br<N; br++){
	  Fsprime = neBoot(Fsnum, Fsdenom, data, x, n);
	  Fsprime = 1/(2*Fsprime);
	  if(Fsprime < lbNe)
	    Fsprime = lbNe;
	  else if (Fsprime > ubNe)
	    Fsprime = ubNe;
	  gsl_matrix_set(nemcmc, x, br, Fsprime);
	}
      }
      // no data for this population x generation, sample Ne from prior
      else{
	cerr << "no data for population " << j << " generation " << t << " or " << t+1 << endl;
	cerr << "sampling Ne from the prior" << endl;
	for(br=0; br<N; br++){
	  gsl_matrix_set(nemcmc, x, br, gsl_ran_flat(r, lbNe, ubNe));
	}
      }
   }
    //upate ne for last generation from the prior, no other information
    x = t+j*data->nGens; // get index
    for(br=0; br<N; br++){
      gsl_matrix_set(nemcmc, x, br, gsl_ran_flat(r, lbNe, ubNe));
    }
  }
  // write results to hdf5
  writeNe(data, hdf5, N);
}


// read input from the infiles
void mcmc :: getNe(string neFile, dataset * data, hdf5cont * hdf5, int N){
  int j, n;
  string line, element;
  ifstream infile;
  istringstream stream;
  double value;

  // read environmental covariate, single column of values
  infile.open(neFile.c_str());
  if (!infile){
    cerr << "Cannot open file " << neFile << endl;
    exit(1);
  }
  // loop through file, one line per pop x gen
  for(j=0; j<data->nTotal; j++){
    getline(infile, line);
    stream.str(line);
    stream.clear();
    stream >> element;
    value = atof(element.c_str());
    for(n=0; n<N; n++){
      gsl_matrix_set(nemcmc, j, n, value);
    }
  }
  infile.close();

  // write results to hdf5
  writeNe(data, hdf5, N);
}


// bayesian bootstrap to generate posterior for Ne
double mcmc :: neBoot(gsl_vector * num, gsl_vector * denom, dataset * data, int x, double nhat){
  
  int i;
  double u, g, a, b;
  double bbA = 0, bbB = 0;
  double Fs, Fsprime;
  int n = data->nLoci;
  gsl_vector * uv;
  uv = gsl_vector_calloc(n+1);
  gsl_vector * gv;
  gv = gsl_vector_calloc(n);

  // draw n-1 random uniform deviates
  gsl_vector_set(uv, 0, 0);
  for(i=1; i<n; i++){
    u = gsl_ran_flat(r, 0, 1);
    gsl_vector_set(uv, i, u);
  }
  gsl_vector_set(uv, n, 1);
  // sort them
  gsl_sort_vector(uv);
  
  // calculate the gaps beteen deviates
  for(i=0; i<n; i++){
    g = gsl_vector_get(uv, i+1) - gsl_vector_get(uv, i);
    gsl_vector_set(gv, i, g);
  }
  
  // weighted average across loci to get Fs
  for(i=0; i<n; i++){
    bbA += gsl_vector_get(gv, i) * gsl_vector_get(num, i);
    bbB += gsl_vector_get(gv, i) * gsl_vector_get(denom, i);
  }
  Fs = bbA/bbB;
  
  a = Fs * (1 - (double) 1.0/(2 * nhat)) - (double) 2.0/nhat;
  b = (1 + Fs/4) * (1 -1/nhat);
  Fsprime = a/b;
 
  gsl_vector_free(uv);
  gsl_vector_free(gv);

  return Fsprime;

}

// write Ne draws to hdf5 file
void mcmc :: writeNe(dataset * data, hdf5cont * hdf5, int n){
  int d, m;

  // select hyperslab for Ne: pop x mcmc
  hdf5->sr2[0] = 0;
  hdf5->sr2[1] = 0;
  hdf5->br2[0] = data->nTotal;
  hdf5->br2[1] = 1;
  for(d=0; d<2; d++){
    hdf5->cr2[d] = 1;
  }
  for(m=0; m<n; m++){
    hdf5->sr2[1]=m;
    hdf5->status = H5Sselect_hyperslab(hdf5->dataspaceTM, H5S_SELECT_SET,
				       hdf5->sr2, NULL, hdf5->cr2, hdf5->br2);
    gsl_matrix_get_col(ne, nemcmc, m);
    hdf5->status = H5Dwrite(hdf5->datasetNe, H5T_NATIVE_DOUBLE, hdf5->popVector,
			    hdf5->dataspaceTM, H5P_DEFAULT, ne->data);
  }
}

// gibbs update of P
void mcmc :: updateP(dataset * data, int N){
  int j, t, i;
  uint x;
  double val;
  double a, b, twoNe;
  double p0, pdraw;

  // obtain values of Ne to use from posterior
  if (N < 0)
    N = 0;
  for(x=0; x<nemcmc->size1; x++){
    val = gsl_matrix_get(nemcmc, x, N);
    gsl_vector_set(ne, x, val);
  }

  // loop over populations and generations
  for(j=0; j<data->nPops; j++){
    for(t=0; t<data->nGens; t++){
      x = t+j*data->nGens; // get index
      for(i=0; i<data->nLoci; i++){
	a = gsl_matrix_int_get(data->aCnts, i, x);
	b = gsl_matrix_int_get(data->nCnts, i, x) - gsl_matrix_int_get(data->aCnts, i, x);
	a+=DBL_MIN;
	b+=DBL_MIN;
	if (t == 0){ // initial gen
	  a += 0.5;
	  b += 0.5;
	  // from next generation
	  p0 = gsl_matrix_get(p, i, x+1);
	  twoNe = gsl_vector_get(ne, x) * 2.0;
	  a += p0 * twoNe;
	  b += (1-p0) * twoNe;
	}
	else if (t < (data->nGens-1)){
	  // from previous generation
	  p0 = gsl_matrix_get(p, i, x-1);
	  twoNe = gsl_vector_get(ne, x-1) * 2.0;
	  a += p0 * twoNe;
	  b += (1-p0) * twoNe;
	  // from next generation
	  p0 = gsl_matrix_get(p, i, x+1);
	  twoNe = gsl_vector_get(ne, x) * 2.0;
	  a += p0 * twoNe;
	  b += (1-p0) * twoNe;
	}
	else {
	  // from previous generation
	  p0 = gsl_matrix_get(p, i, x-1);
	  twoNe = gsl_vector_get(ne, x-1) * 2.0;
	  a += p0 * twoNe;
	  b += (1-p0) * twoNe;
	}
	pdraw = gsl_ran_beta(r, a, b);
	gsl_matrix_set(p, i, x, pdraw);
      }
    }
  }
}

// store mcmc samples for P
void mcmc :: storeP(dataset * data, hdf5cont * hdf5, int n){
  int j;

  for(j=0; j<data->nTotal; j++){
    gsl_matrix_get_col(hdf5->auxLocusVector, p, j);
    gsl_matrix_set_col(pmcmc[j], n, hdf5->auxLocusVector);
  }
}

// write  mcmc samples for P
void mcmc :: writeP(dataset * data, hdf5cont * hdf5, int n){
  int d, j;

  // write to hdf5
 // select hyperlab for allele frequencies
  hdf5->sr3[0] = 0;
  hdf5->sr3[1] = 0;
  hdf5->sr3[2] = 0;
  hdf5->br3[0] = data->nLoci;
  hdf5->br3[1] = 1;
  hdf5->br3[2] = n;
  for(d=0; d<3; d++){
    hdf5->cr3[d] = 1;
  }
  for(j=0; j<data->nTotal; j++){
    hdf5->sr3[1] = j;
    hdf5->status = H5Sselect_hyperslab(hdf5->dataspaceLTM, H5S_SELECT_SET,
    				       hdf5->sr3, NULL, hdf5->cr3, hdf5->br3);
    hdf5->status = H5Dwrite(hdf5->datasetP, H5T_NATIVE_DOUBLE, hdf5->locusMcmcMatrix,
    			    hdf5->dataspaceLTM, H5P_DEFAULT, pmcmc[j]->data);
  }
}


// main MCMC function
void mcmc :: update(dataset * data, int N){

  // sample Ne from stored posterior
  sampleNeP(data, N);
  // w22 = 1, w12 = 1 + 2hs, w11 = 1 + 2s
  // A11 = p^2, A12 = 2pq, A22 = q^2
  // E[p'] = p + 2sp(1-p)(p+h(1-2p))

  // update p with selection
  updatePs(data);
  // update alpha parameter for selection
  updateAlpha(data);
  // update beta parameter for selection
  updateBeta(data);  
}

// store mcmc samples for alpha, beta and s
void mcmc :: store(dataset * data, hdf5cont * hdf5, int n){
  int j;

  // alpha
  gsl_matrix_set_col(amcmc, n, alpha);

  // beta
  gsl_matrix_set_col(bmcmc, n, beta);

  // selection coefficients
  for(j=0; j<data->nTotal; j++){
    gsl_matrix_get_col(hdf5->auxLocusVector, s, j);
    gsl_matrix_set_col(smcmc[j], n, hdf5->auxLocusVector);
  }
}

// write and mcmc samples for Ne
void mcmc :: write(dataset * data, hdf5cont * hdf5, int n){
  int j, d;


  // select hyperslab for alpha and beta: locus x mcmc
  hdf5->sr2[0] = 0;
  hdf5->sr2[1] = 0;
  hdf5->br2[0] = data->nLoci;
  hdf5->br2[1] = n;
  for(d=0; d<2; d++){
    hdf5->cr2[d] = 1;
  }
  hdf5->status = H5Sselect_hyperslab(hdf5->dataspaceLM, H5S_SELECT_SET,
  				     hdf5->sr2, NULL, hdf5->cr2, hdf5->br2);
  hdf5->status = H5Dwrite(hdf5->datasetAlpha, H5T_NATIVE_DOUBLE, hdf5->locusMcmcMatrix,
  			  hdf5->dataspaceLM, H5P_DEFAULT, amcmc->data);
  hdf5->status = H5Dwrite(hdf5->datasetBeta, H5T_NATIVE_DOUBLE, hdf5->locusMcmcMatrix,
  			  hdf5->dataspaceLM, H5P_DEFAULT, bmcmc->data);

  // select hyperlab for selection coefficients
  hdf5->sr3[0] = 0;
  hdf5->sr3[1] = 0;
  hdf5->sr3[2] = 0;
  hdf5->br3[0] = data->nLoci;
  hdf5->br3[1] = 1;
  hdf5->br3[2] = n;
  for(d=0; d<3; d++){
    hdf5->cr3[d] = 1;
  }
  for(j=0; j<data->nTotal; j++){
    hdf5->sr3[1] = j;
    hdf5->status = H5Sselect_hyperslab(hdf5->dataspaceLTM, H5S_SELECT_SET,
    				       hdf5->sr3, NULL, hdf5->cr3, hdf5->br3);
    hdf5->status = H5Dwrite(hdf5->datasetS, H5T_NATIVE_DOUBLE, hdf5->locusMcmcMatrix,
    			    hdf5->dataspaceLTM, H5P_DEFAULT, smcmc[j]->data);
  }

}

// ------------------- private functions --------------------//

//  select a value of Ne from the posterior
void mcmc :: sampleNeP(dataset * data, int N){
  uint j;
  uint sam;
  double val;

  sam = gsl_rng_uniform_int(r, N);

  for(j=0; j<nemcmc->size1; j++){
    val = gsl_matrix_get(nemcmc, j, sam);
    gsl_vector_set(ne, j, val);
  }
}

// gibbs update of P with selection
void mcmc :: updatePs(dataset * data){
  int j, t, i, x;
  double a, b, twoNe;
  double p0, p1, pexp, pscale, pdraw;
  double sel;

  // loop over populations and generations
  for(j=0; j<data->nPops; j++){
    for(t=0; t<data->nGens; t++){
      x = t+j*data->nGens; // get index
      for(i=0; i<data->nLoci; i++){
	a = gsl_matrix_int_get(data->aCnts, i, x);
	b = gsl_matrix_int_get(data->nCnts, i, x) - gsl_matrix_int_get(data->aCnts, i, x);
	a+=DBL_MIN;
	b+=DBL_MIN;
	if (t == 0){ // initial gen
	  a += 0.5;
	  b += 0.5;
	  // from next generation
	  p0 = gsl_matrix_get(p, i, x);
	  p1 = gsl_matrix_get(p, i, x+1);
	  twoNe = gsl_vector_get(ne, x) * 2.0;
	  sel = gsl_matrix_get(s, i, x);
	  pexp = calcp(p0, sel);
	  pscale = p1 - (pexp - p0);
	  if(pscale >= 0.999)
	    pscale = 0.999;
	  if(pscale <= 0.001)
	    pscale = 0.001;
	  a += pscale * twoNe;
	  b += (1-pscale) * twoNe;
	}
	else if (t < (data->nGens-1)){
	  // from previous generation
	  p0 = gsl_matrix_get(p, i, x-1);
	  twoNe = gsl_vector_get(ne, x-1) * 2.0;
	  sel = gsl_matrix_get(s, i, x-1);
	  pexp = calcp(p0, sel);
	  if (pexp >= 0.999)
	    pexp = 0.999;
	  if (pexp <= 0.001)
	    pexp = 0.001;
	  a += pexp * twoNe;
	  b += (1-pexp) * twoNe;
	  // from next generation
	  p0 = gsl_matrix_get(p, i, x);
	  p1 = gsl_matrix_get(p, i, x+1);
	  twoNe = gsl_vector_get(ne, x) * 2.0;
	  sel = gsl_matrix_get(s, i, x);
	  pexp = calcp(p0, sel);
	  pscale = p1 - (pexp - p0);
	  if(pscale >= 0.999)
	    pscale = 0.999;
	  if(pscale <= 0.001)
	    pscale = 0.001;
	  a += pscale * twoNe;
	  b += (1-pscale) * twoNe;
	}
	else {
	  // from previous generation
	  p0 = gsl_matrix_get(p, i, x-1);
	  twoNe = gsl_vector_get(ne, x-1) * 2.0;
	  sel = gsl_matrix_get(s, i, x-1);
	  pexp = calcp(p0, sel);
	  if (pexp >= 0.999)
	    pexp = 0.999;
	  if (pexp <= 0.001)
	    pexp = 0.001;
	  a += pexp * twoNe;
	  b += (1-pexp) * twoNe;
	}
	pdraw = gsl_ran_beta(r, a, b);
	gsl_matrix_set(p, i, x, pdraw);
      }
    }
  }
}



// metropolis update for alpha, intercept for selection, where,
// s = alpha + beta x
// s Pr(p' | p, ne, s) ~ normal(E[p'],p(1-p)/2Ne)
// w22 = 1, w12 = 1 + 2hs, w11 = 1 + 2s
// A11 = p^2, A12 = 2pq, A22 = q^2
// E[p'] = p + 2sp(1-p)(p+h(1-2p))
void mcmc :: updateAlpha(dataset * data){
  int j, t, i, x;
  double a, b;
  double prob;
  double pr = 0;
  double prstar = 0;
  double p0, p1;
  //double h = 0.5;// future parameter for het. effect
  double sel, selstar, alphastar;
  double pexp, pexpstar;

  for(i=0; i<data->nLoci; i++){
    // loop over generations (except first) and populations
    pr = 0;
    prstar = 0;
    // proposal value
    alphastar = gsl_ran_gaussian(r, propAlpha) + gsl_vector_get(alpha, i);
    for(j=0; j<data->nPops; j++){
      for(t=1; t<data->nGens; t++){
	x = t+j*data->nGens; // get index
	p1 = gsl_matrix_get(p, i, x);
	p0 = gsl_matrix_get(p, i, x-1);
	sel = gsl_matrix_get(s, i, x-1);
	pexp = calcp(p0, sel);
	selstar = alphastar + gsl_vector_get(beta, i) * gsl_vector_get(data->envData, (x-1));
	pexpstar = calcp(p0,  selstar);	
	// calcluate pr(p1 | s, p0, ne)
	a = pexp * 2.0 * gsl_vector_get(ne, x-1) + DBL_MIN;
	b = (1 - pexp) * 2.0 * gsl_vector_get(ne, x-1) + DBL_MIN;
	prob = gsl_ran_beta_pdf(p1, a, b);
	pr += fixlogpr(prob);
	a = pexpstar * 2.0 * gsl_vector_get(ne, x-1) + DBL_MIN;
	b = (1 - pexpstar) * 2.0 * gsl_vector_get(ne, x-1) + DBL_MIN;
	prob = gsl_ran_beta_pdf(p1, a, b);
	prstar += fixlogpr(prob);
      }
    }
  
    // prior prob of alpha and alphastar
    prob = gsl_ran_gaussian_pdf(gsl_vector_get(alpha, i), sdSlabAlpha);
    pr += fixlogpr(prob);
    prob = gsl_ran_gaussian_pdf(alphastar, sdSlabAlpha);
    prstar += fixlogpr(prob);

    if((prstar-pr) > log(gsl_ran_flat(r, 0, 1))){ // metropolis ratio
      gsl_vector_set(alpha, i, alphastar);
      for(j=0; j<data->nPops; j++){
	for(t=0; t<data->nGens; t++){
	  x = t+j*data->nGens; // get index   
	  selstar = alphastar + gsl_vector_get(beta, i) * gsl_vector_get(data->envData, x);
	  gsl_matrix_set(s, i, x, selstar);
	}
      }
    }
  }
}

// metropolis update for beta, slope for selection, where,
// s = alpha + beta x
// s Pr(p' | p, ne, s) ~ normal(E[p'],p(1-p)/2Ne)
// w22 = 1, w12 = 1 + 2hs, w11 = 1 + 2s
// A11 = p^2, A12 = 2pq, A22 = q^2
// E[p'] = p + 2sp(1-p)(p+h(1-2p))
void mcmc :: updateBeta(dataset * data){
  int j, t, i, x;
  double a, b;
  double prob;
  double pr = 0;
  double prstar = 0;
  double p0, p1;
  //double h = 0.5;// future parameter for het. effect
  double sel, selstar, betastar;
  double pexp, pexpstar;

  for(i=0; i<data->nLoci; i++){
  // loop over generations (except first) and populations
    pr = 0;
    prstar = 0;
    betastar = gsl_ran_gaussian(r, propBeta) + gsl_vector_get(beta, i);
    for(j=0; j<data->nPops; j++){
      for(t=1; t<data->nGens; t++){
	x = t+j*data->nGens; // get index
	p1 = gsl_matrix_get(p, i, x);
	p0 = gsl_matrix_get(p, i, x-1);
	sel = gsl_matrix_get(s, i, x-1);
	pexp = calcp(p0, sel);
	// proposal value
	selstar = gsl_vector_get(alpha, i) + betastar * gsl_vector_get(data->envData, (x-1));
	pexpstar = calcp(p0, selstar);	
	// calcluate pr(p1 | s, p0, ne)
	a = pexp * 2.0 * gsl_vector_get(ne, x-1) + DBL_MIN;
	b = (1 - pexp) * 2.0 * gsl_vector_get(ne, x-1) + DBL_MIN;
	prob = gsl_ran_beta_pdf(p1, a, b);
	pr += fixlogpr(prob);
	a = pexpstar * 2.0 * gsl_vector_get(ne, x-1) + DBL_MIN;
	b = (1 - pexpstar) * 2.0 * gsl_vector_get(ne, x-1) + DBL_MIN;
	prob = gsl_ran_beta_pdf(p1, a, b);
	prstar += fixlogpr(prob);
      }
    }
  
    // prior prob of beta and betastar
    prob = gsl_ran_gaussian_pdf(gsl_vector_get(beta, i), sdSlabBeta);
    pr += fixlogpr(prob);
    prob = gsl_ran_gaussian_pdf(betastar, sdSlabBeta);
    prstar += fixlogpr(prob);

    if((prstar-pr) > log(gsl_ran_flat(r, 0, 1))){ // metropolis ratio
      gsl_vector_set(beta, i, betastar);
      for(j=0; j<data->nPops; j++){
	for(t=0; t<data->nGens; t++){
	  x = t+j*data->nGens; // get index   
	  selstar = gsl_vector_get(alpha, i) + betastar * gsl_vector_get(data->envData, x);
	  gsl_matrix_set(s, i, x, selstar);
	}
      }
    }
  }
}

/* calculate expected value for p in the next generation */
double mcmc :: calcp(double p0, double sel){
  double h = 0.5;
  double p1;
  double w[3];
  double wbar;
  double X, Y2, Z;

  X = p0 * p0;
  Y2 = 2 * p0 * (1 - p0);
  Z = (1 - p0) * (1 - p0);

  // w22 = 1, w12 = 1 + 2hs, w11 = 1 + 2s
  w[1] = 1 + 2.0 * sel;
  w[2] = 1 + 2.0 * h * sel;
  w[3] = 1;
  wbar = X * w[0] + Y2 * w[1] + Z * w[2];
  
  p1 = p0 + (X * w[0] + 0.5 * Y2 * w[1] - p0 * wbar)/wbar;
  
  if(p1 >= 0.999)
    p1 = 0.999;
  if(p1 <= 0.001)
    p1 = 0.001;

  return p1;

  // alternative
  //p0 + 2 * sel * p0 * (1-p0) * (p0+h * (1-2 * p0));
}

/* avoids numerical overflow/underflow with log probabilities */
double mcmc :: fixlogpr(double prob){
  double lp;
  int finite;
  
  // if prob = 0, set to min
  if (prob == 0)
    prob = GSL_DBL_MIN;
  
  lp = log(prob);
  
  finite = gsl_isinf(prob);
  
  if (finite == 1) // +Inf
    lp = GSL_DBL_MAX;
  else if (finite == -1) // -Inf
    lp = GSL_DBL_MIN;
  return lp;
}

