#ifndef TYPEEXG_ARMA_MATRIXCLASSKKH_H
#define TYPEEXG_ARMA_MATRIXCLASSKKH_H

#include "armadillo"
#include "matrix_class_KKH.h"
#include <cstring> // for memcpy
#include <cstdio> // printf
#include <type_traits> // is_same

// works for 2D matrices (real numbers, not complex) with one channel
template <typename T>
void matKKH2arma(const Matk<T> &matIn, arma::Mat<T> &matOut)
{	
	int nrows = matIn.nrows();
	int ncols = matIn.ncols();
	int nchannels = matIn.nchannels();
	
	matOut.set_size(nrows, ncols);	

	T* src_ptr = matIn.get_ptr();
	T * dst_ptr = matOut.memptr();
	std::memcpy(dst_ptr, src_ptr, sizeof(T)*nrows*ncols);
}

// works for 2D matrices (real numbers, not complex) with multiple channels
template <typename T>
void matKKH2arma(const Matk<T> &matIn, arma::Cube<T> &matOut)
{
	int nrows = matIn.nrows();
	int ncols = matIn.ncols();
	int nchannels = matIn.nchannels();
	
	matOut.set_size(nrows, ncols, nchannels);
	
	T* src_ptr = matIn.get_ptr();
	T * dst_ptr = matOut.memptr();
	std::memcpy(dst_ptr, src_ptr, sizeof(T)*nrows*ncols*nchannels);	
}

// works for 2D matrices (real numbers, not complex) with either one or multiple channels
// T1 is any native C++ type; T2 is either arma::Mat<T1> or arma::Cube<T1>
template <typename T1, typename T2>
void matKKH2arma(const Matk<T1> &matIn, T2 &matOut)
{
	int nrows = matIn.nrows();
	int ncols = matIn.ncols();
	int nchannels = matIn.nchannels();
	T1* src_ptr, dst_ptr;
	
	if (std::is_same<T2, arma::Cube<T1>>::value)
		matOut.set_size(nrows, ncols, nchannels);
	else if (std::is_same<T2, arma::Mat<T1>>::value)	
		matOut.set_size(nrows, ncols);
	else
	{
		printf("Error: Invalid T1 or T2.\n");
		return;
	}
	src_ptr = matIn.get_ptr();
	dst_ptr = matOut.memptr();
	std::memcpy(dst_ptr, src_ptr, sizeof(T)*nrows*ncols*nchannels);		
}

template <typename T1, typename T2>
void arma2matKKH(const T1 &matIn, Matk<T2> &matOut)
{	
	int nrows = matIn.n_rows;
	int ncols = matIn.n_cols;
	int nchannels; 
	
	if (std::is_same<T1, arma::Cube<T2>>::value)
		nchannels = matIn.n_slices;
	else if (std::is_same<T1, arma::Mat<T2>>::value)	
		nchannels = 1;
	else
	{
		printf("Error: Invalid T1 or T2.\n");
		return;
	}		
		
	matOut.create(nrows, ncols, nchannels);	
	T *dst_ptr = matOut.get_ptr();
	unsigned long count = 0;
	
	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < ncols; j++)
			for (int i = 0; i < nrows; i++)
				dst_ptr[count++] = matIn.at(i,j,k);		
}

template <typename T>
void arma2matKKH(const arma::Mat<T> &matIn, Matk<T2> &matOut)
{	
	int nrows = matIn.n_rows;
	int ncols = matIn.n_cols;
	int nchannels = 1;	
		
	matOut.create(nrows, ncols, nchannels);	
	T *dst_ptr = matOut.get_ptr();
	unsigned long count = 0;
	
	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < ncols; j++)
			for (int i = 0; i < nrows; i++)
				dst_ptr[count++] = matIn.at(i,j,k);		
}

template <typename T>
void arma2matKKH(const arma::Cube<T> &matIn, Matk<T2> &matOut)
{	
	int nrows = matIn.n_rows;
	int ncols = matIn.n_cols;
	int nchannels = matIn.n_slices;	
		
	matOut.create(nrows, ncols, nchannels);	
	T *dst_ptr = matOut.get_ptr();
	unsigned long count = 0;
	
	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < ncols; j++)
			for (int i = 0; i < nrows; i++)
				dst_ptr[count++] = matIn.at(i,j,k);		
}



#undef EigenMatrix
#undef EIGEN_NO_DEBUG

#endif
