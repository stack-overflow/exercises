#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>

const int ARRAY_SIZE = (1000);
const int NUM_RANGES = 10;
const int RANGESIZE = ARRAY_SIZE / NUM_RANGES;

enum
{
    DATA = 0,
    RESULT,
    FINISH
};

void generate_random_numbers(int *array, int size)
{   
    for (int i = 0; i < size; ++i)
    {
        array[i] = rand() % size;
    }
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

void allocate_tmp_result_arrays(int ***array, int num_arrays, int arrays_size)
{
    (*array) = new int*[num_arrays];

    for (int i = 0; i < num_arrays; ++i)
    {
        (*array)[i] = new int[arrays_size];
    }
}

void deallocate_tmp_result_arrays(int ***array, int num_arrays)
{
    std::for_each((*array),
                  (*array) + (num_arrays - 1),
                  std::default_delete<int[]>());
    
    delete [] (*array);
}

int main(int argc, char** argv)
{
    ios_base::sync_with_stdio(0);
    srand(time(0));

    int rank, world_size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        int num_sent      = 0;
        int num_recv      = 0;
        int requests_size = 3 * (world_size - 1);
        int *the_vector   = new int[ARRAY_SIZE];
        int **tmp_results;

        MPI_Request *requests = new MPI_Request[requests_size];

        generate_random_numbers(the_vector, ARRAY_SIZE);
        allocate_tmp_result_arrays(&tmp_results, world_size - 1, RANGESIZE);

        std::vector<int> final_result;
        final_result.reserve(ARRAY_SIZE / 2);

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

        int index, count;

        while ((num_sent * RANGESIZE) < ARRAY_SIZE)
        {
            MPI_Waitany(2 * (world_size - 1), requests, &index, &status);
            if (index < (world_size - 1))
            {
                ++num_recv;

                MPI_Get_count(&status, MPI_INT, &count);
                std::cout << "Master received " << count << " numbers." << std::endl;

                for (int i = 0; i < count; ++i)
                {
                    final_result.push_back(tmp_results[index][i]);
                }

                MPI_Wait(&requests[world_size - 1 + index], MPI_STATUS_IGNORE);

                int chunk_index = (num_sent * RANGESIZE);
                MPI_Isend(
                    &the_vector[chunk_index],
                    RANGESIZE, MPI_INT, index + 1,
                    DATA, MPI_COMM_WORLD,
                    &requests[world_size - 1 + index]);
                
                ++num_sent;

                MPI_Irecv(
                    tmp_results[index],
                    RANGESIZE, MPI_INT, index + 1,
                    RESULT, MPI_COMM_WORLD,
                    &(requests[index]));
            }
        }

        while (num_recv < NUM_RANGES)
        {
            MPI_Waitany(2 * (world_size - 1), requests, &index, &status);
            ++num_recv;
            MPI_Get_count(&status, MPI_INT, &count);
            std::cout << "Master received " << count << " numbers. Lasting." << std::endl;
        }

        for (int i = 0; i < final_result.size(); ++i)
        {
            std::cout << final_result[i] << ", ";
        }
        std::cout << std::endl;

        int endingChunk = -1;
        for (int i = 1; i < world_size; ++i)
        {
            MPI_Isend(&endingChunk, 1, MPI_INT, i, DATA, MPI_COMM_WORLD,
                    &(requests[2 * world_size - 3 + i]));
        }

        MPI_Waitall(3 * world_size - 3, requests, MPI_STATUSES_IGNORE);

        delete [] requests;
        delete [] the_vector;
        deallocate_tmp_result_arrays(&tmp_results, world_size - 1);
    }
    else
    {
        MPI_Request *requests = new MPI_Request[2];

        int *recv_vector_first = new int[RANGESIZE];
        int *recv_vector_second = new int[RANGESIZE];
        int *recv_vector_third = new int[RANGESIZE];

        requests[0] = requests[1] = MPI_REQUEST_NULL;

        MPI_Recv(recv_vector_first, RANGESIZE, MPI_INT, 0, DATA, MPI_COMM_WORLD, &status);
        
        int *to_solve = recv_vector_first;
        while (true)
        {
            MPI_Irecv(recv_vector_second, RANGESIZE, MPI_INT, 0, DATA, MPI_COMM_WORLD, &requests[0]);

            int *end_solved = std::remove_if(recv_vector_first, recv_vector_first + RANGESIZE, not_prime);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

            if (recv_vector_second[0] == -1)
            {
                break;
            }

            std::swap(recv_vector_first, recv_vector_third);
            std::swap(recv_vector_first, recv_vector_second);

            // frist -> master
            MPI_Isend(
                recv_vector_third,
                std::distance(recv_vector_third, end_solved),
                MPI_INT, 0, RESULT, MPI_COMM_WORLD,
                &requests[1]);
        }

        delete [] requests;
        delete [] recv_vector_third;
        delete [] recv_vector_second;
        delete [] recv_vector_first;
    }

    MPI_Finalize();
    return 0;
}