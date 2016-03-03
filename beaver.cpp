/** External Libraries */
#include <aubio/aubio.h>
#include <flann/flann.h>
#include "gist.h"
#include "RtAudio.h"
#include <SFML/Audio.hpp>

/** Standard Library */
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

using namespace std;

// Platform-dependent sleep routines.
#if defined( __WINDOWS_ASIO__ ) || defined( __WINDOWS_DS__ ) || defined( __WINDOWS_WASAPI__ )
  #include <windows.h>
  #define SLEEP( milliseconds ) Sleep( (DWORD) milliseconds ) 
#else // Unix variants
  #include <unistd.h>
  #define SLEEP( milliseconds ) usleep( (unsigned long) (milliseconds * 1000.0) )
#endif

/** Data structure passed to inputCallback function */
struct InputData
{
    unsigned long offset;
    fstream myfile1;
    smpl_t full_input[1024]; 
};

/** Variable Declarations */
aubio_onset_t * o;		// Aubio onset object
fvec_t * in, * out;		// Aubio onset input and output
int mfcc_count = 0;		// Keep track of MFCC for classification
Gist<float> gist(512, 44100);	// GIST Feature Extraction Library  Initialization

/** FLANN (Approximate Nearest Neighbours) INITIALIZATION */
float * dataset;
float testset[15];		// Array of size of feature vector
int nn = 1;			/* Number of nearest neighbors to search for
				   (should only be set to 1) */
int result;			// a-NN classification result
float dists;			// a-NN distances
struct FLANNParameters p;	// a-NN option params
float speedup;			// Not used?
flann_index_t index_id;		// a-NN search index object
int rows = 20;			// Total number of training samples
int cols = 15;			// Number of features in one training sample
int tcount = 1;			// Test set size

/** SFML (Audio playback) declaration */
sf::SoundBuffer buffer;		/* SoundBuffer objects store sound file;
				   paired with Sound objects */
sf::Sound sound;		/* Sound object paired with SoundBuffer;
				   Controls volume, playback etc. */
sf::Music music, music2;	// Music objects stream from file

/*
 * Function:  read_points 
 * --------------------
 *  reads feature vector training samples from file
 * 
 *  filename: name of file
 *  rows: number of training samples
 *  columns: number of features in a feature vector
 *  returns: float array containing feature vector data
 */
float *
read_points(const char * filename, int rows, int cols)
{
    float * data, * p;
    FILE * fin;
    int i, j;
    fin = fopen(filename,"r");
    if (!fin)
    {
	printf("Cannot open input file.\n");
	exit(EXIT_FAILURE);
    }

    p = data = (float *) malloc(rows * cols * sizeof(float));
    if (!data)
    {
	printf("Cannot allocate memory.\n");
	fclose(fin);
	exit(EXIT_FAILURE);
    }

    for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	    fscanf(fin, "%g ", p++);

    fclose(fin);
    return data;
}

/*
 * Function:  inputCallback 
 * --------------------
 *  reads data from the audio stream, detects audio onsets, and performs
 *  classification
 *  
 *  outputBuffer: For output (or duplex) streams, the client should write 
 *  nFrames of audio sample frames into this buffer. This argument should 
 *  be recast to the datatype specified when the stream was opened. For 
 *  input-only streams, this argument will be NULL.
 *   
 *  inputBuffer: For input (or duplex) streams, this buffer will hold 
 *  nFrames of input audio sample frames. This argument should be recast 
 *  to the datatype specified when the stream was opened. For output-only 
 *  streams, this argument will be NULL.
 *
 *  nFrames: The number of sample frames of input or output data in the 
 *  buffers. The actual buffer size in bytes is dependent on the data 
 *  type and number of channels in use.
 *
 *  streamTime: The number of seconds that have elapsed since the stream 
 *  was started.
 *  
 *  status: If non-zero, this argument indicates a data overflow or 
 *  underflow condition for the stream. The particular condition can be 
 *  determined by comparison with the RtAudioStreamStatus flags.
 * 
 *  userData: A pointer to optional data provided by the client when opening 
 *  the stream (default = NULL).
 */
int
inputCallback(void * outputBuffer, void * inputBuffer, unsigned int nBufferFrames,
	      double streamTime, RtAudioStreamStatus status, void * userData)
{
    if (status)
	std::cout << "Stream overflow detected!" << std::endl;
  
    // Do something with the data in the "inputBuffer".
    smpl_t * input = (smpl_t *) inputBuffer;
    InputData * data = (InputData *) userData;

    // Until entire buffer of 512 sample frames is copied to buffer in
    while (data->offset < 256)
    {
	std::copy(input + data->offset, input + data->offset + 255, in->data);

	// GIST runs FFT automatically to prep input for feature extraction   
	gist.processAudioFrame(input, 512);
    
	// aubio onset detection fills "out" variable  
	aubio_onset_do(o, in, out); 

	// If sound detected, do this:
	if (out->data[0] != 0)
	{ 
	    // Compute zero-crossings rate
	    float zcr = gist.zeroCrossingRate(); 
	    // Computer spectral centroid
	    float specCent = gist.spectralCentroid();
    
	    std::vector<float> feat;
	    std::vector<float> r = gist.melFrequencyCepstralCoefficients();
 
	    // Add MFCC coefficients 2 - 12;
	    feat.insert(feat.end(), r.begin() + 2, r.end());
	    feat.push_back(zcr);
	    feat.push_back(specCent);

	    // Calculate mean and standard deviation; add to feature vector
	    double sum	  = std::accumulate(feat.begin(), feat.end(), 0.0);
	    double mean	  = sum / feat.size();
	    double sq_sum = std::inner_product(feat.begin(), feat.end(), feat.begin(), 0.0);
	    double stdev  = std::sqrt(sq_sum / feat.size() - mean * mean);
	    
	    feat.push_back(mean);
	    feat.push_back(stdev);

	    // Normalization function
	    int sum_norm = 0;
	    for (unsigned j = 0 ; j < feat.size(); j++)
		sum_norm += pow(feat[j], 2);
	    float mag = sqrt(sum_norm); 
	    mfcc_count++;    

	    // Append feature vector to mfccs.data and print feature vector
	    for (unsigned i = 0; i < feat.size(); i++)
	    {
		// Ignores the first erroneous sample if there is one
		if (mfcc_count > 1 && mfcc_count <= 21)
		    data->myfile1 << feat[i] / mag << " ";
            
		else if (mfcc_count > 21) 
		    testset[i] = feat[i] / mag;
       
		fprintf(stderr, "%f ", feat[i] / mag);
	    }

	    fprintf(stderr, "\n");
	    // Add new line after each feature vector in file
	    if (mfcc_count > 1 && mfcc_count <= 21)
		data->myfile1 << endl;

	    // If there are 20 input samples, run training algorithm
	    if (mfcc_count == 21)
	    {
		printf("Reading input data file.\n");
		dataset = read_points("mfccs.data", rows, cols);
		printf("Computing index.\n");
		index_id = flann_build_index(dataset, rows, cols, &speedup, &p);
	    }

	    // Classify all samples after 20 training samples
	    if (mfcc_count > 21)
	    {
		flann_find_nearest_neighbors_index(index_id, testset, tcount,
						   &result, &dists, nn, &p);
		printf("Dist: %lf and Result: %i \n", dists, result);

		// If sound classified as being among first 10 samples and error distance < 0.1
		if ((unsigned)(result - 0) <= (9-0) && dists < 0.1)
		{
		    music.play();
		    printf("CLAP MATCH!\n");
		}
		// If sound classified as being among second 10 samples and error distance < 0.1
		else if ((unsigned)(result - 10) <= (19-10) && dists < 0.1)
		{
		    music2.play();
		    printf("SNAP MATCH\n");
		}
		else
		    printf("NO MATCH!\n");
	    }
	} 
	data->offset += 256; 
    }
    data->offset = 0;
    /* If necessary, we can abort the stream after 25 seconds:
       if (streamTime > 25)
       return 1; */

    // Otherwise, continue normal stream operation
    return 0; 
}

int
main()
{
    RtAudio adc;
    RtAudio::StreamParameters iParams;
    adc.showWarnings(true);
    
    if (adc.getDeviceCount() < 1)
    {
	std::cout << "\nNo audio devices found!\n";
	exit(EXIT_FAILURE);
    }
    
    iParams.deviceId	 = adc.getDefaultInputDevice();
    iParams.nChannels	 = 1; // Record in mono; 2 for stereo
    iParams.firstChannel = 0;
 
    unsigned int sampleRate   = 44100; 
    unsigned int bufferFrames = 512; 
    unsigned int hop_size     = bufferFrames / 2; // Necessary for aubio onset detection

    // Stream options
    RtAudio::StreamOptions options;
    options.flags = RTAUDIO_SCHEDULE_REALTIME | RTAUDIO_MINIMIZE_LATENCY;
  
    // Initialize onset input and output vectors and aubio onset object
    in	= new_fvec (hop_size);
    out = new_fvec (2);  
    // Initialize aubio onset object with "complex" detection method
    o	= new_aubio_onset("complex", bufferFrames, hop_size, sampleRate); 
    
    // Create InputData object
    InputData data;
    data.offset = 0;
    // Open new file to store sample data
    data.myfile1.open("mfccs.data", std::fstream::out | std::fstream::app);

    // FLANN Initialization 
    p		= DEFAULT_FLANN_PARAMETERS;
    p.algorithm = FLANN_INDEX_KDTREE;
    flann_set_distance_type(FLANN_DIST_CHI_SQUARE, -1);
    p.trees	= 20; 
    p.log_level = FLANN_LOG_INFO;
    p.checks	= 16; 
  
    // SFML (audio playback) streaming from file
    if (!music.openFromFile("kick-big.wav"))
	return EXIT_FAILURE;
    if (!music2.openFromFile("tom-rototom.wav"))
	return EXIT_FAILURE;
   
    // Open and Start input audio stream
    try
    {
	adc.openStream(NULL, &iParams, RTAUDIO_FLOAT32,
		       sampleRate, &bufferFrames, &inputCallback,
		       (void *) &data); 
	fprintf(stderr, "Listening! \n");
	adc.startStream();
    }
    catch (RtAudioError& e)
    {
	e.printMessage();
	exit(EXIT_FAILURE);
    }
  
    char input;
    std::cout << "\nRecording ... press <enter> to quit.\n";
    std::cin.get(input);

    /* In contrast, we can abort the stream:
       To be used in place of cin.get above if inputCallback returns 1 (i.e. is aborted)
       while (adc.isStreamRunning()) {
       SLEEP(100);
       }	*/


     if (adc.isStreamOpen() )
	adc.closeStream();
     
    // File close (mfccs.data)
    if (data.myfile1.is_open())
    {
	cout << "Closing file mfccs.data." << endl;
	data.myfile1.close();
    }
    
    // Clean up memory
    flann_free_index(index_id, &p);
    free(dataset);
  
    if (in)
	del_fvec(in);
    if (out)
	del_fvec(out);

    if (o)
    {
       del_aubio_onset(o);
       o = 0;
    }
    
    cout << "Exiting..." << endl;
   
    return 0;
}
