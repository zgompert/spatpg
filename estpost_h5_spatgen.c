/* read hdf5 and write data summary */
/* Time-stamp: <Saturday, 18 April 2015, 16:47 MDT -- zgompert> */


#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_histogram.h>

#include "hdf5.h"
#include <getopt.h>

/* header, function declarations */
#define VERSION "0.1 - 9 December 2014"
#define MAXFILEN 20 /* maximum number of infiles */
void usage(char * name);;
void estpost(hid_t * file, const char * param, double credint, int burn, int nbins, 
	     int sumtype, int wids, int nchains);
int paramexists(const char * param);
void calcparam(const char * param, hid_t dataspace, hid_t * dataset, 
	       gsl_matrix * sample, gsl_vector * onesample, gsl_vector * onechain,
	       gsl_histogram * hist, double credint, int nparam, int nsamples, int burn, 
	       int nbins, int sumtype, int wids, int nchains);
void calcparam2d(const char * param, hid_t dataspace, hid_t * dataset, 
		 gsl_matrix * sample, gsl_vector * onesample, gsl_vector * onechain,
		 gsl_histogram * hist, double credint, int nparam1, int nparam2, int nsamples, 
		 int burn, int nbins, int sumtype, int wids, int nchains);
void calcci(gsl_vector * sample, double credint, int nsamples);
void calchist(gsl_vector * sample,  gsl_histogram * hist, int nsamples, int nbins);
void writetext(gsl_vector * sample,  int nsamples);
void calcdiags(gsl_matrix * samples, gsl_vector * sample, gsl_vector * onechain,
	       int nchains, int nsamples, int ntotal);
void unwrapmatrix(gsl_matrix * m, gsl_vector * v, int d1, int d2);


FILE * outfp; 

/* beginning of main */

int main (int argc, char **argv) {
  int sumtype = 0; /* summary to perform
		      0 = estimate and ci, 1 = histogram, 2 = convert to text */
  int ch = 0;
  int burn = 0; /* discard the first burn samples as a burn-in */
  int nbins = 20; /* number of bins for histogram */
  int wids = 1; /* boolean, write ids in first column */
  int c;
  char * infile = "undefined"; /* filename */
  char * outfile = "postout.txt";
  char * param = "undefined"; /* parameter to summarize */

  int nchains = 0; /* number of mcmc chains = number of infiles */

  double credint = 0.95; /* default = 95% credible interval, this is ETPI */

  /* variables for getopt_long */
  static struct option long_options[] = {
    {"version", no_argument, 0, 'v'},
    {0, 0, 0, 0} 
  };
  int option_index = 0;

  /* variables for hdf5 */
  hid_t file[MAXFILEN];         /* file handle */
  
  /*  get command line arguments */
  if (argc < 2) {
    usage(argv[0]);
  }
  
  while ((ch = getopt_long(argc, argv, "o:v:p:c:b:h:s:w:", 
			   long_options, &option_index)) != -1){
    switch(ch){
    case 'o':
      outfile = optarg;
      break;
    case 'p':
      param = optarg;
      break;
    case 'c':
      credint = atof(optarg);
      break;
    case 'b':
      burn = atoi(optarg);
      break;
    case 'h':
      nbins = atoi(optarg);
      break;
    case 's':
      sumtype = atoi(optarg);
      break;
    case 'w':
      wids = atoi(optarg);
      break;
    case 'v':
      printf("%s version %s\n", argv[0], VERSION); 
      /* VERSION is a macro */
      exit(0); /* note program will exit if this option is specified */
    case '?':
    default:
      usage(argv[0]);
    }
  }

  if(!paramexists(param)){
    printf("The specified parameter does not exist\n");
    printf("Possible parameters are: ne, p, alpha, beta, s\n");
    exit(1);
  }

  /* open the h5 files, read only */
  while (optind < argc){
    infile = argv[optind];
    printf("file = %s\n",infile);
    file[nchains] = H5Fopen(infile, H5F_ACC_RDONLY, H5P_DEFAULT);
    nchains++;
    optind++;
  }

 /* open the outfile */
  outfp = fopen(outfile, "w");
  if ( !outfp ){
    fprintf(stderr, "Can't open %s for writing!\n", outfile);
    exit(1);
  }
  if ((sumtype == 0) && (wids == 1)){
    fprintf(outfp,"param,mean,median,ci_%.3f_LB,ci_%.3f_UB\n", credint, credint);
  }
  else if ((sumtype == 3) && (wids == 1)){
    fprintf(outfp,"param,ess,psrf\n");
  }
  
  /* main function */
  estpost(file, param, credint, burn, nbins, sumtype, wids, nchains);
  
  fclose(outfp);
  for(c=0; c<nchains; c++){
    H5Fclose(file[c]);
  }
  return 0;
}

/* ---------------- Functions ------------------ */

/* Prints usage */
void usage(char * name){
  fprintf(stderr,"\n%s version %s\n\n", name, VERSION); 
  fprintf(stderr, "Usage:   estpost [options] infile1.hdf5 infile2.hdf5\n");
  fprintf(stderr, "-o     Outfile [default = postout.txt]\n");
  fprintf(stderr, "-p     Name of parameter to summarize: ne, p, alpha, beta, s\n");
  fprintf(stderr, "-c     Credible interval to calculate [default = 0.95]\n");
  fprintf(stderr, "-b     Number of additinal MCMC samples to discard for burn-in [default = 0]\n");
  fprintf(stderr, "-h     Number of bins for posterior sample histogram [default = 20]\n");
  fprintf(stderr, "-s     Which summary to perform: 0 = posterior estimates and credible intervals\n");
  fprintf(stderr, "                                 1 = histogram of posterior samples\n");
  fprintf(stderr, "                                 2 = convert to plain text\n");
  fprintf(stderr, "                                 3 = MCMC diagnostics\n");  
  fprintf(stderr, "-w     Write parameter identification to file, boolean [default = 1]\n");
  fprintf(stderr, "-v     Display estpost software version\n");
  exit(1);
}

/* Function estimates the mean, median and credible interval of a
   continous posterior distribution, simply writes discrete
   posterior distribution, or write text for parameter to file */
void estpost(hid_t * file, const char * param, double credint, int burn, int nbins, 
	     int sumtype, int wids, int nchains){
  gsl_matrix * sample;
  gsl_vector * onesample;
  gsl_vector * onechain;
  gsl_histogram * hist;
  hid_t dataset[MAXFILEN], dataspace;
  hsize_t dims[3]; /* note 3 dimensions is maximum in this application */
  int rank = 0, status_n = 0;
  /* dimensions for mcmc samples */
  int nloci = 0, nsamples = 0;
  int ntotal = 0;
  int chain;

  hist = gsl_histogram_alloc((size_t) nbins);
  /* we already have checked that param should exist in input hdf5 file*/
  for (chain=0; chain<nchains; chain++){
    dataset[chain] = H5Dopen2(file[chain], param, H5P_DEFAULT);
  }
  dataspace = H5Dget_space(dataset[0]); // we assume that all chains have the same dimensions
  rank = H5Sget_simple_extent_ndims(dataspace); 
  status_n = H5Sget_simple_extent_dims(dataspace, dims, NULL);

  /* interpret dimensions based on parameter, then calculate (if
     necessary) and print desired quantities */

  /* locus x mcmc */
  if (strcmp(param, "alpha") == 0 ||
      strcmp(param, "beta") == 0){ 
    nloci = dims[0];
    nsamples = dims[1];
    if (burn >= nsamples){
      printf("Burnin exceeds number of samples\n");
      exit(1);
    }
    sample = gsl_matrix_calloc(nsamples - burn, nchains);
    onesample = gsl_vector_calloc(nsamples - burn);
    onechain = gsl_vector_calloc(nsamples - burn);
    printf("parameter dimensions for %s: loci = %d, samples = %d, chains = %d\n", 
	   param, nloci, nsamples, nchains);
    calcparam(param, dataspace, dataset, sample, onesample, onechain, hist, credint, 
	      nloci, nsamples, burn, nbins, sumtype, wids, nchains);
  }

  /* ntotal x mcmc */
  else if (strcmp(param, "ne") == 0){ 
    ntotal = dims[0];
    nsamples = dims[1];
    if (burn >= nsamples){
      printf("Burnin exceeds number of samples\n");
      exit(1);
    }
    sample = gsl_matrix_calloc(nsamples - burn, nchains);
    onesample = gsl_vector_calloc(nsamples - burn);
    onechain = gsl_vector_calloc(nsamples - burn);
    printf("parameter dimensions for %s: populationsXgenerations = %d,samples = %d, chains = %d\n", param, ntotal, nsamples, nchains);
    calcparam(param, dataspace, dataset, sample, onesample, onechain, hist, credint, 
	      ntotal, nsamples, burn, nbins, sumtype, wids, nchains);
  }

  /* locus x ntotal x mcmc */
  else if (strcmp(param, "p") == 0 ||
      strcmp(param, "s") == 0){ 
    nloci = dims[0];
    ntotal = dims[1];
    nsamples = dims[2];
    if (burn >= nsamples){
      printf("Burnin exceeds number of samples\n");
      exit(1);
    }
    sample = gsl_matrix_calloc(nsamples - burn, nchains);
    onesample = gsl_vector_calloc(nsamples - burn);
    onechain = gsl_vector_calloc(nsamples - burn);
    printf("parameter dimensions for %s: loci = %d, samples = %d, chains = %d\n", 
	   param, nloci, nsamples, nchains);
    calcparam2d(param, dataspace, dataset, sample, onesample, onechain, hist, credint, 
		nloci, ntotal, nsamples, burn, nbins, sumtype, wids, nchains);
  }


  else {
    printf("Error in finding parameter.  Should not be possible.\n");
    exit(1);
  }
}


/* calculate summaries for parameters indexed by locus and MCMC sample*/
void calcparam(const char * param, hid_t dataspace, hid_t * dataset, 
	       gsl_matrix * sample, gsl_vector * onesample, gsl_vector * onechain,
	       gsl_histogram * hist, double credint, int nparam, int nsamples, int burn, 
	       int nbins, int sumtype, int wids, int nchains){
  int i, c;
  hid_t mvector;
  hsize_t start[2];  /* Start of hyperslab */
  hsize_t count[2] = {1,1};  /* Block count */
  hsize_t block[2];
  herr_t ret;
  hsize_t samdim[1];

  gsl_vector * catsample;

  catsample = gsl_vector_calloc(nchains * (nsamples - burn));

  /* create vector for buffer */
  samdim[0] = nsamples - burn;
  mvector = H5Screate_simple(1, samdim, NULL);
  /* loop through parameters */
 
  for (i=0; i<nparam; i++){ 
    if (wids == 1){
      fprintf(outfp,"%s_param_%d,",param,i);
    }
    start[0] = i; start[1] = burn;
    block[0] = 1; block[1] = (nsamples - burn);
    for (c=0; c<nchains; c++){
      ret = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, block);
      ret = H5Dread(dataset[c], H5T_NATIVE_DOUBLE, mvector, dataspace, H5P_DEFAULT, 
		    onesample->data);
      gsl_matrix_set_col(sample, c, onesample);
    }
    unwrapmatrix(sample, catsample, (nsamples - burn), nchains);
    /* estimate and credible intervals */
    if (sumtype == 0){
      calcci(catsample, credint, nchains * (nsamples - burn));
    }
    /* generate histogram */
    else if (sumtype == 1){
      calchist(catsample, hist, nchains * (nsamples - burn), nbins);
    }
    /* write to text file */
    else if (sumtype == 2){
      writetext(catsample, nchains * (nsamples - burn));
    }
    /* MCMC diagnostics */
    else if (sumtype == 3){
      calcdiags(sample, catsample, onechain, nchains, (nsamples - burn), 
		nchains * (nsamples - burn));
    }
  }
  H5Sclose(mvector);
}

/* calculate summaries for parameters indexed by locus, pops x gen, and MCMC sample*/
void calcparam2d(const char * param, hid_t dataspace, hid_t * dataset, 
		 gsl_matrix * sample, gsl_vector * onesample, gsl_vector * onechain,
		 gsl_histogram * hist, double credint, int nparam1, int nparam2, int nsamples, 
		 int burn, int nbins, int sumtype, int wids, int nchains){
  int i, j, c;
  hid_t mvector;
  hsize_t start[3];  /* Start of hyperslab */
  hsize_t count[3] = {1,1,1};  /* Block count */
  hsize_t block[3] = {1,1,nsamples-burn};
  herr_t ret;
  hsize_t samdim[1];

  gsl_vector * catsample;

  catsample = gsl_vector_calloc(nchains * (nsamples - burn));

  /* create vector for buffer */
  samdim[0] = nsamples - burn;
  mvector = H5Screate_simple(1, samdim, NULL);
  /* loop through parameters */
 
  for (i=0; i<nparam1; i++){ 
    for (j=0; j<nparam2; j++){
      if (wids == 1){
	fprintf(outfp,"%s_param_%d_%d,",param,i,j);
      }
      start[0] = i; start[1] = j, start[2] = burn;;
      /*block[0] = 1; block[1] = (nsamples - burn);*/
      for (c=0; c<nchains; c++){
	ret = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, block);
	ret = H5Dread(dataset[c], H5T_NATIVE_DOUBLE, mvector, dataspace, H5P_DEFAULT, 
		      onesample->data);
	gsl_matrix_set_col(sample, c, onesample);
      }
      unwrapmatrix(sample, catsample, (nsamples - burn), nchains);
      /* estimate and credible intervals */
      if (sumtype == 0){
	calcci(catsample, credint, nchains * (nsamples - burn));
      }
      /* generate histogram */
      else if (sumtype == 1){
	calchist(catsample, hist, nchains * (nsamples - burn), nbins);
      }
      /* write to text file */
      else if (sumtype == 2){
	writetext(catsample, nchains * (nsamples - burn));
      }
      /* MCMC diagnostics */
      else if (sumtype == 3){
	calcdiags(sample, catsample, onechain, nchains, (nsamples - burn), 
		  nchains * (nsamples - burn));
      }
    }
  }
  H5Sclose(mvector);
}


/* Function calculates and prints mean, median, and ci */
void calcci(gsl_vector * sample, double credint, int nsamples){
  double lb = (1.0 - credint)/2.0;
  double ub = 1.0 - lb;
  int i;					
  double x;
  /* sort samples in ascending order */
  gsl_sort_vector(sample);
  fprintf(outfp,"%.6f,", gsl_stats_mean(sample->data, 1, nsamples));
  fprintf(outfp,"%.6f,", gsl_stats_median_from_sorted_data(sample->data, 1, nsamples));
  fprintf(outfp,"%.6f,", gsl_stats_quantile_from_sorted_data(sample->data, 1, nsamples,lb));
  fprintf(outfp,"%.6f\n", gsl_stats_quantile_from_sorted_data(sample->data, 1, nsamples,ub));
}

/* Function generates and prints a histogram of the posterior */
void calchist(gsl_vector * sample,  gsl_histogram * hist, int nsamples, int nbins){
  int i;
  double min = 0, max = 0;
  double upper = 0, lower = 0, midpoint = 0;
  
  min = gsl_vector_min(sample);
  max = gsl_vector_max(sample);
  max += max * 0.0001; /* add a small bit to max, as upper bound is exclusive */
  (void) gsl_histogram_set_ranges_uniform(hist, min, max);
  
  /* generate histogram */
  for (i=0; i<nsamples; i++){
    (void) gsl_histogram_increment(hist, gsl_vector_get(sample, i));
  }
  
  /* write histogram */
  for (i=0; i<nbins; i++){
    gsl_histogram_get_range(hist, i, &lower, &upper);
    midpoint = (lower + upper) / 2.0;
    if (i == 0){
      fprintf(outfp,"%.6f,%d", midpoint, (int) gsl_histogram_get(hist, i));
    }
    else{
      fprintf(outfp,",%.6f,%d", midpoint, (int) gsl_histogram_get(hist, i));
    }
  }
  fprintf(outfp, "\n");

}

/* Function prints the ordered samples from posterior as plain text */
void writetext(gsl_vector * sample,  int nsamples){
  int i = 0;

  fprintf(outfp,"%.5f", gsl_vector_get(sample, i));
  for (i=1; i<nsamples; i++){
    fprintf(outfp,",%.5f", gsl_vector_get(sample, i));
  }    
  fprintf(outfp,"\n");
}

int paramexists(const char * param){
  if(!strcmp("ne", param)){ 
    return(1);
  }
  else if(!strcmp("p", param)){
    return(1);
  }
  else if(!strcmp("alpha", param)){
    return(1);
  }
  else if(!strcmp("beta", param)){
    return(1);
  }
  else if(!strcmp("s", param)){
    return(1);
  }
  else{
    return(0);
  }
}

/* Function calculates MCMC diagnostics */
void calcdiags(gsl_matrix * samples, gsl_vector * sample, gsl_vector * onechain,
	       int nchains, int nsamples, int ntotal){
  int i, j, m;
  double autoc = 0, autocl = 0, ess = 0;
  double mn, var;
  int lag = 1;
  double rho = 1, minrho = 0.01;

  double W = 0, Bn = 0;
  double grmn = 0, shat, psrf;

  /* effective sample size */
  mn = gsl_stats_mean(sample->data, 1, ntotal);
  var = gsl_stats_variance_m(sample->data, 1, ntotal, mn);
  while(rho > minrho){
    autocl = 0;
    for(i=0,j=(i+lag); j<ntotal; i++,j++){
      autocl += (gsl_vector_get(sample, i) - mn) * (gsl_vector_get(sample, j) - mn);
    }
    autoc += (double) 1.0/(ntotal-lag) * autocl/var;

    rho = autocl;
    lag++;
  }
  ess = (double) ntotal / (1.0 + 2.0 * autoc);
  
  /* gelman rubin psrf */
  for(m=0; m<nchains; m++){ /* within chain variance */
    gsl_matrix_get_col(onechain, samples, m);
    W += gsl_stats_variance(onechain->data, 1, nsamples);
    grmn += gsl_stats_mean(onechain->data, 1, nsamples);
  }
  W /= (double) nchains;
  grmn /= (double) nchains;
  /* between chain variance */
  for(m=0; m<nchains; m++){
    gsl_matrix_get_col(onechain, samples, m);
    mn = gsl_stats_mean(onechain->data, 1, nsamples);
    Bn += (mn - grmn) * (mn - grmn);
  }
  Bn /= (double) (nchains - 1);

  /* variance estimates */
  shat = (double) (nsamples - 1) * W/nsamples + Bn;
  psrf = (double) ((nchains + 1)/nchains) * (shat/W) - ((nsamples -1)/(ntotal));
    
  fprintf(outfp,"%.1f,%.2f\n",ess,psrf);
}


/* this function copys the content of a matrix M to a vector V, by row */
void  unwrapmatrix(gsl_matrix * m, gsl_vector * v, int d1, int d2){
  int i, j;

  for (i=0; i<d1; i++){
    for (j=0; j<d2; j++){
      gsl_vector_set(v,j + (d2 * i), gsl_matrix_get(m, i, j));
    }
  }

}
