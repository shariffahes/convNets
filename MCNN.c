#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

int *outputResult;
pthread_mutex_t mutex;
int n, threadsNb;
int *pooled;
int pooledFilterSize = 2;
int outDim, stride, nbOfFilters;

void *findMax(void *);
int get3dAt(int *, int, int, int, int, int);
void set3dAt(int *, int, int, int, int, int, int);
int get4dAt(int *, int, int, int, int, int, int, int);
void set4dAt(int *, int, int, int, int, int, int, int, int);
int convolve(int *, int *, int, int, int, int, int, int);

int main(int argc, char **argv)
{
    int rank, nbOfProcessor;

    //define files.
    FILE *readF, *writeF;

    //clock to measure time.
    clock_t c1 = clock();
    int *image, *filter, imgDim, filterDim;
    int nbOfChannels, filterChannels, iterations;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbOfProcessor);

    /*open files for reading. Opening them globally  and not in rank 0 since
    all process needs to read from the file. aka: No need for Bcast.*/
    readF = fopen(argv[1], "r");
    //fscanf(readF,"%d",&iterations);
    // int i=0;

    if (rank == 0)
    {

        /*Start scanning:
    1) we scan the stride value.
    2) scan nb of filters. How many filters will convolve on the image.
    3) scan the size of filter. Filters are considered sequare matrices thus rows=columns= size of filter.
    4) scan filter Channels. how many channels or dimension each filter will contain. Channels of filter should be equal to that of image.*/
        fscanf(readF, "%d %d %d %d", &stride, &nbOfFilters, &filterDim, &filterChannels);

        //malloc filter lineary.
        filter = (int *)malloc(sizeof(int) * nbOfFilters * filterDim * filterDim * filterChannels);
        //represent the current filter that will be filled.
        int curr_f = 0;
        int dim1 = filterDim * filterDim * filterChannels;
        int dim = filterDim * filterDim;

        while (curr_f < nbOfFilters)
        {
            //current channel that will be filled.
            int curr_ch = 0;
            while (curr_ch < filterChannels)
            {
                for (int i = 0; i < filterDim; i++)

                    for (int j = 0; j < filterDim; j++)
                    {
                        //read x.
                        int x = 0;
                        fscanf(readF, "%d", &x);

                        //filter[curr_f][curr_ch][i][j] = x;
                        set4dAt(filter, dim1, dim, filterDim, curr_f, curr_ch, i, j, x);
                    }

                curr_ch++;
            }

            curr_f++;
        }

        /*read the img dimension
        scan:
        1)how many channels the image have. For instance colored image
          has 3 channels. 1 channel is assumed to be grey imagesa.
        2)the image dimension which is considered a sequare matrix column=rows=imgDim.*/
        fscanf(readF, "%d %d", &nbOfChannels, &imgDim);

        //channels must be equal otherwise raise an error.
        if (nbOfChannels != filterChannels)
        {
            printf("error. depth/channels of filter and image must be equal");
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
            return -1;
        }

        //current channel.
        int curr_c = 0;

        //malloc image
        image = (int *)malloc(sizeof(int) * nbOfChannels * imgDim * imgDim);
        while (curr_c < nbOfChannels)
        {
            for (int i = 0; i < imgDim; i++)

                for (int j = 0; j < imgDim; j++)
                {
                    //read x.
                    int x = 0;
                    fscanf(readF, "%d", &x);
                    //image[curr_c][i][j]=x;
                    set3dAt(image, imgDim * imgDim, imgDim, curr_c, i, j, x);
                }
            curr_c++;
        }
    }

    //Broad cast everything that has been modified in rank 0 to all processes.
    MPI_Bcast(&stride, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbOfFilters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&filterChannels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&filterDim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        filter = (int *)malloc(sizeof(int) * nbOfFilters * filterDim * filterDim * filterChannels);
    MPI_Bcast(filter, nbOfFilters * filterChannels * filterDim * filterDim, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imgDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbOfChannels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //malloc image in each rank except rank 0 since it has been allocated in 0.
    if (rank != 0)
        image = (int *)malloc(sizeof(int) * imgDim * imgDim * nbOfChannels);
    MPI_Bcast(image, imgDim * imgDim * nbOfChannels, MPI_INT, 0, MPI_COMM_WORLD);

    /*Our approach:
    Calculate how many rows will be produced after the convolution of filter
    on the image. Then make each process responsible to produce number of these rows. The rows will be divided equally among the processes*/

    /*To calculate the dimension of the matrix that will be produced after convolution. We used this formula: floor((n+2p-f)/s) +1. In our project, consider there is no padding thus formula will be floor((n-f)/s)+1.*/
    outDim = floor((imgDim - filterDim) / stride) + 1;

    /*Divide the number of rows that will be produced after convolution among the available processes*/
    int rowsPerRank = outDim / nbOfProcessor;
    int div = rowsPerRank;
    /*if it can't be divided equally. Then, assign all the remaining to the last process*/
    int rem = outDim - nbOfProcessor * rowsPerRank;
    if (rank == nbOfProcessor - 1)
    {
        rowsPerRank += rem;
    }

    /*The results of the convolution of each filter on the image will be stored in this localConvolve.
    The size of this array will be equal to how many rows each process will produce * the number of filters that will convolve on the image * the number of elements that will be produced by each row. */
    int *localConvolve = (int *)malloc(sizeof(int) * nbOfFilters * rowsPerRank * outDim);

    //current filter that will convolve the image.
    int curr_filter = 0;

    /* When the process finish from convolving the image by first filter it moves to the next filter. There is no need to wait for other processes to finish convolving their first filter because results are independent. */
    while (curr_filter < nbOfFilters)
    {
        //indY will track the y dimension of localConvolve.
        int indY = 0;
        /*i will specify this rank at which row it will star. For instance:
        if stride is 2 and each process is responsible for div=2 rows. 
        then:
        Rank 0 will start convolving from row 0 and stop at row 4.
        Rank 1 will start convolving from row 4. */
        int i = rank * div * stride;
        int curr_y = i;

        /*We already know the number of rows that each process will convolve and shift to from rowsPerRank.*/
        int steps = 0;
        while (steps < rowsPerRank && curr_y + (filterDim - 1) < imgDim)
        {

            //keep track of the x dimension of the localConvolve.
            //Reset value to 0 when we shift downward to next row.
            //Ex: localConvolve[0][1] then localConvolve[1][0].
            int indX = 0;

            /*For each row, The process will keep convolving and shifting to another column until the current column + the filter dimension will overflow outside the image dimension*/
            for (int curr_x = 0; curr_x + (filterDim - 1) < imgDim; curr_x += stride)
            {

                //convolve the image at current x and y by current filter.
                int res = convolve(image, filter, curr_y, curr_x, curr_filter, imgDim, filterDim, nbOfChannels);

                //ReLU. It is common to turn all negative values into zeros to
                //achieve non linearity.
                if (res < 0)
                    res = 0;

                //localConvolve[curr_filter][indY][indX]=res.
                set3dAt(localConvolve, outDim * rowsPerRank, outDim, curr_filter, indY, indX, res);

                indX++;
            }

            indY++;

            //move into the next row by adding the defined number of strides.
            curr_y += stride;
            steps++;
        }
        //move to the next filter.
        curr_filter++;
    }

    //image and filter are no longer required.
    free(image);
    free(filter);

    //Will be used to fill the output result of each filter from each process.
    outputResult = (int *)malloc(sizeof(int) * nbOfFilters * outDim * outDim);

    int counter = 0;
    for (int i = 0; i < nbOfFilters; i++)
    {

        int *disp = (int *)malloc(sizeof(int) * nbOfProcessor);
        int *recv = (int *)malloc(sizeof(int) * nbOfProcessor);

        for (int j = 0; j < nbOfProcessor; j++)
        {
            //always receiving same number of elements except for last rank.
            recv[j] = div * outDim;

            if (j == nbOfProcessor - 1)
                recv[j] = (div + rem) * outDim;

            disp[j] = counter;
            counter += recv[j];
        }
        //Gather all recv[j] elements from each process. Then start filling
        //the outputResult from new address which is equivalent to the numbers of //element that are filled from each process.
        MPI_Gatherv((localConvolve + i * rowsPerRank * outDim), rowsPerRank * outDim, MPI_INT, outputResult, recv, disp, MPI_INT, 0, MPI_COMM_WORLD);
    }

    //Wait till all the ranks has finished filling the outputResult.
    MPI_Barrier(MPI_COMM_WORLD);

    //localConvolve is no longer required. Results are in outputResults.
    free(localConvolve);

    if (rank == 0)
    {
        //open the file for writing.
        writeF = fopen("out.txt", "w");

        //for each filter move around all the element and print them.
        int i = 0;
        while (i < nbOfFilters)
        {

            for (int j = 0; j < outDim; j++)
            {
                for (int z = 0; z < outDim; z++)
                {
                    //x = outputResult[i][j][z].
                    int x = get3dAt(outputResult, outDim * outDim, outDim, i, j, z);
                    fprintf(writeF, "%d ", x);
                }
                fprintf(writeF, "\n");
            }
            fprintf(writeF, "\n");
            i++;
        }
        /*After the convolution is finished. We will enter the pooling layer which can be avg, min, or max pool. In this project, we chose the maxpooling method. This layer will downsize the size of the matrix resulted from the convolution. */

        //by default here the max pooling will have stride of 2.
        stride = 2;

        /*Max pooling will be done in parallel using threads.
        Our approach is to create thread for each row of the matrix that will be resulted from the max pooling*/

        //the formula is similar to the previous.
        threadsNb = floor((outDim - pooledFilterSize) / stride) + 1;

        //for each filter output. It will perform a max pooling. [down sampling].
        pooled = (int *)malloc(sizeof(int) * nbOfFilters * threadsNb * threadsNb);

        pthread_t tid[threadsNb];
        int threadID[threadsNb];
        for (int j = 0; j < threadsNb; j++)
        {
            threadID[j] = j;
            pthread_create(&tid[j], NULL, findMax, (void *)&threadID[j]);
        }

        for (int j = 0; j < threadsNb; j++)
        {
            threadID[j] = j;
            pthread_join(tid[j], NULL);
        }

        //results of max pooling will be printed in another file. Just for
        //clarification and keeping the output of the convolution layer for
        //validation.
        FILE *maxOut = fopen("maxOut.txt", "w");
        i = 0;
        while (i < nbOfFilters)
        {

            for (int j = 0; j < threadsNb; j++)
            {
                for (int z = 0; z < threadsNb; z++)
                {
                    //pooled[i][j][z];
                    //i: represent current filter.
                    //j: is the current row of the array.
                    //z: is the current column of the array.
                    int value = get3dAt(pooled, threadsNb * threadsNb, threadsNb, i, j, z);
                    fprintf(maxOut, "%d ", value);
                }
                fprintf(maxOut, "\n");
            }
            fprintf(maxOut, "\n");
            i++;
        }
        //measure the time for execution.
        clock_t c2 = clock();
        double time_spent = (double)(c2 - c1) / CLOCKS_PER_SEC;
        printf("time: %f\n", time_spent);
    }

    MPI_Finalize();
    return 0;
}

//method will be used for threads.
void *findMax(void *index)
{

    int ind = *((int *)index);
    //the current row. We know that each thread will be responsible for one
    //row of the output result after the pooling. Thus, row is ind * stride
    // where ind is the current thread.
    int row = ind * stride;

    //Each thread will produce the result of max pooling for one row in each
    //filter
    int curr_filt = 0;

    while (curr_filt < nbOfFilters)
    {
        //current x in the outputResult.
        int curr_x = 0;
        //current index of the pooling array.
        int poolInd = 0;

        //keep incrementing the column in the outputResult by stride
        //until you overflow the dimension of the outputResult.
        while (curr_x + (pooledFilterSize - 1) < outDim)
        {
            //Localmax in each filter will start by -1.
            //No negative values will be presented in outputResult, since
            //we used RelU and turned all possible negative values to 0.
            int Localmax = -1;

            //y of filter and it will move by the limit of the filter size.
            //it start same as the current row and then it will be increased
            //by 1
            for (int yFilt = row; yFilt < row + pooledFilterSize; yFilt++)
            {
                //x of filter and it start same as the current column
                for (int xFilt = curr_x; xFilt < curr_x + pooledFilterSize; xFilt++)
                {
                    //value = outputResult[curr_filt][yFilt][xFilt].
                    int value = get3dAt(outputResult, outDim * outDim, outDim, curr_filt, yFilt, xFilt);

                    //whenever the value is bigger than localMax then this is max.
                    if (value > Localmax)
                    {
                        Localmax = value;
                    }
                }
            }
            //pooled[curr_filt][ind][poolInd] = localmax.
            set3dAt(pooled, threadsNb * threadsNb, threadsNb, curr_filt, ind, poolInd++, Localmax);

            // move along the column of outputResult by stride.
            curr_x += stride;
        }

        //go to next filter same row.
        curr_filt++;
    }

    return NULL;
}

int convolve(int *image, int *filter, int y, int x, int filterId, int imageDim, int filterDim, int channelsNb)
{
    int result = 0;
    int dim1 = filterDim * filterDim * channelsNb;
    int dim = filterDim * filterDim;
    int curr_channel = 0;

    //convolve all the channels of the image and filter.
    //Each channel of the image will be convolved by its corresponding channel in
    //the filter (filterChannel = imageChannel).
    //The result of convolution from each channel will be accumulated and stored
    //in result thus the results will be a matrix of 1 channel from each filter.
    while (curr_channel < channelsNb)
    {
        //current coordination of filter.
        int indY = 0;

        //convolve the given matrix.
        for (int i = y; i < y + filterDim; i++)
        {
            int indX = 0;
            for (int j = x; j < x + filterDim; j++)
            {
                //a = image[curr_channel][i][j].
                int a = get3dAt(image, imageDim * imageDim, imageDim, curr_channel, i, j);
                //b = filter[filterId][curr_channel][indY][indX].
                int b = get4dAt(filter, dim1, dim, filterDim, filterId, curr_channel, indY, indX);

                //add the result.
                //note we can add the bias here if there is any.
                //In our project we ignored the presence of bias.
                result += a * b; // bias[curr_f]
                indX++;
            }
            indY++;
        }
        //move to next channel.
        curr_channel++;
    }

    return result;
}

//This method will simplify the work when dealing with 3d array which
//is defined linearly.
int get3dAt(int *arr, int dimension, int columnLen, int x, int y, int z)
{

    int a = arr[x * dimension + columnLen * y + z];

    return a;
}
void set3dAt(int *arr, int dimension, int columnLen, int x, int y, int z, int value)
{
    arr[x * dimension + columnLen * y + z] = value;
}

//This method will simplify the work when dealing with 4d array which
//is defined linearly.
int get4dAt(int *arr, int dimension1, int dimension2, int columnLen, int x, int y, int z, int w)
{
    int a = arr[dimension1 * x + dimension2 * y + z * columnLen + w];

    return a;
}
void set4dAt(int *arr, int dimension1, int dimension2, int columnLen, int x, int y, int z, int w, int value)
{
    arr[dimension1 * x + dimension2 * y + z * columnLen + w] = value;
}