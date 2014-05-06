#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>

const int ARRAY_SIZE = (100);
const int NUM_RANGES = 10   ;
const int RANGESIZE = ARRAY_SIZE / NUM_RANGES;

enum
{
    DATA = 0,
    RESULT,
    FINISH
};

int *generate_random_vector()
{
    int *result = new int[ARRAY_SIZE];
    
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        result[i] = rand() % ARRAY_SIZE;
    }

    return result;
}

bool is_prime(int a)
{
    for (int c = 2; c <= (int)sqrt(a); ++c)
    { 
        if (a%c == 0) return false;
    }

    return true;
}

bool not_prime(int a)
{
    return !is_prime(a);
}

int **create_tmp_result_vector(int world_size, int RANGESIZE)
{
    int **result = new int*[world_size];

    for (int i = 0; i < world_size; ++i)
    {
        result[i] = new int[RANGESIZE];
    }

    return result;
}

int main(int argc, char** argv)
{
    srand(time(0));

    int rank, world_size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        int num_sent      = 0;
        int requests_size = 3 * (world_size - 1);
        int *the_vector   = generate_random_vector();
        int **tmp_results = create_tmp_result_vector(world_size - 1, RANGESIZE);

        MPI_Request *requests = new MPI_Request[requests_size];

        for (int i = 1; i < world_size; ++i)
        {
            int chunk_index = (num_sent * RANGESIZE);
            MPI_Send(&the_vector[chunk_index], RANGESIZE, MPI_INT, i, DATA, MPI_COMM_WORLD);
            ++num_sent;
        }

        for (int i = 0; i < requests_size; ++i)
        {
            requests[i] = MPI_REQUEST_NULL;
        }

        for (int i = 1; i < world_size; ++i)
        {
            MPI_Irecv(
                tmp_results[i - 1],
                RANGESIZE, MPI_INT, i,
                RESULT, MPI_COMM_WORLD,
                &(requests[i - 1]));
        }

        for (int i = 1; i < world_size; ++i)
        {
            int chunk_index = (num_sent * RANGESIZE);
            MPI_Isend(
                &the_vector[chunk_index],
                RANGESIZE, MPI_INT, i,
                DATA, MPI_COMM_WORLD,
                &requests[world_size - 2 + i]);

            ++num_sent;
        }

        while ((num_sent * RANGESIZE) < ARRAY_SIZE)
        {
            int index;
            MPI_Waitany(2 * (world_size - 1), requests, &index, &status);
            if (index < (world_size - 1))
            {
                int count;
                MPI_Get_count(&status, MPI_INT, &count);
                std::cout << "Master received " << count << " numbers." << std::endl;
                getchar();

                MPI_Wait(&requests[world_size - 1 + index], MPI_STATUS_IGNORE);

                int chunk_index = (num_sent * RANGESIZE);
                MPI_Isend(
                    &the_vector[chunk_index],
                    RANGESIZE, MPI_INT, index,
                    DATA, MPI_COMM_WORLD,
                    &requests[world_size - 1 + index]);

                MPI_Irecv(
                    tmp_results[index],
                    RANGESIZE, MPI_INT, index + 1,
                    RESULT, MPI_COMM_WORLD,
                    &(requests[index]));
            }
        }

        delete [] requests;
        delete [] the_vector;
    }
    else
    {
        int *recv_vector = new int[RANGESIZE];
        MPI_Recv(recv_vector, RANGESIZE, MPI_INT, 0, DATA, MPI_COMM_WORLD, &status);
        int *pend = std::remove_if(recv_vector, recv_vector + RANGESIZE, not_prime);

       
    }

    MPI_Finalize();
    return 0;
}