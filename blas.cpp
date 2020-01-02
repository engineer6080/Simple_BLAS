#include <cstdio>
#include <iostream>
#include <vector>

// Integral types
#include <climits>

#include <fstream> //file output
#include <iomanip> //output formatting

#include "semaphore.h"
#include <atomic>
#include <mutex>
#include <thread>

// EXTERNAL LIBRARY FOR TESTING
#include "Eigen/Core"
#include "Eigen/Dense"
#include <random>

// Boolean for progress output, degrades performance
#define PROG 0

using Eigen::MatrixXd;

/*

Pending:
https://lemire.me/blog/2012/06/20/do-not-waste-time-with-stl-vectors/
semaphore implementation
expanding matrix on addElement and updating values
Figure out how to run multi-core eigen
Progress fix..
*/

std::mutex m;
std::unique_ptr<Semaphore[]> s;
static std::atomic<int> counter(0);

// Instantiate Single Instance only
// Matrix operations class
class Blas {

public:
  // Static function initalizer
  static int getTCount() { return threadcount; }

  static Blas &init(int n = -1) { // auto detect, optional parameter
    static Blas singleB;
    // Get number of threads
    unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
    std::cout << concurentThreadsSupported << " Threads detected" << std::endl;

    if (n <= concurentThreadsSupported && n > 0)
      threadcount = n;
    else
      threadcount = concurentThreadsSupported;

    // Dynamic Allocate Semaphore based on threadcount
    s = std::unique_ptr<Semaphore[]>(new Semaphore[threadcount]);
    return singleB;
  }

private:
  Blas(){};
  ~Blas(){};
  static int threadcount;
};

// must initialize
int Blas::threadcount = 2;

// int, float, double
template <typename T = long double> // default template parameter
class Matrix                        // 2 Dimension only, Matrix Representation
{
public:
private:
  int _R;
  int _C;
  // Single vector for contiguous memory access
  std::vector<T> _iRmtx; // Original Matrix (Row major)
  std::vector<T> _iCmtx; // Transposed Matrix (Col major)

public:
  std::vector<T> *Rmt; // pointer to active matrix
  std::vector<T> *Cmt;

  Matrix() : Rmt(&_iRmtx), Cmt(&_iCmtx){}; // No arg constructor

  // Dimension parameter constructor
  Matrix(int m, int n) : _R(m), _C(n), Rmt(&_iRmtx), Cmt(&_iCmtx){};

  /*
  Column major translation
  Cmt[(c * M) + r] = a[r][c]; //(col_index * rows_per_column) + row_index
  2D Array constructor
  */
  template <std::size_t M, std::size_t N> // ROW, COL
  Matrix(const T (&a)[M][N])              // = assignment operator
  {
    _R = M;
    _C = N;
    Cmt->resize(M * N);
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c) {
        Rmt->push_back(a[r][c]);
        Cmt->operator[]((c * M) + r) = a[r][c];
      }
    }
    // std::cout << M << "x" << N << std::endl;
  }

  ~Matrix(){};

  // Get # of Rows
  const int getR() const { return _R; }

  // Get # of Cols
  const int getC() const { return _C; }

  // Get # of elements
  const int getSize() const { return _R * _C; }

  // Add element to matrix
  // Push_back can be expensive..
  void addElement(T &x) {
    _iRmtx.push_back(x);
    _iCmtx.push_back(x);
  }

  // Transpose
  void transpose() {
    std::swap(Rmt, Cmt);
    std::swap(_R, _C);
  }

  // RHS Matrix type
  // = assignment operator
  Matrix &operator=(const Matrix<T> &a) {
    _R = a._R;
    _C = a._C;

    // Copy internal matrices
    this->_iRmtx = a._iRmtx;
    this->_iCmtx = a._iCmtx;

    // Initalize pointers
    this->Rmt = &_iRmtx;
    this->Cmt = &_iCmtx;

    return *this;
  }

  // Overload Multiply Operator
  Matrix operator*(const Matrix<T> &b) const {

    Matrix<> out;
    // Dimensionality check
    if (getC() != b.getR()) {
      std::cerr << "ERR" << std::endl;
      return b;
    }

    int adx = 0; // Matrix A index increment
    int bdx = 0; // Matrix B index increment

    // (mxn) * (oxp) = m x p
    int output_size = this->getR() * b.getC(); // (mxp products)

    // Dimensions of Output Matrix
    out._R = this->getR();
    out._C = b.getC();

    // Less cpu cycles for add
    (out.Rmt)->reserve(output_size);
    (out.Cmt)->reserve(output_size);

    // Convert vector pointer to reference
    auto &x = *(this->Rmt); // row major matrix
    auto &y = *(b.Cmt);     // column major matrix

    // Limit std::cout (progress indicator)
    int prod_count = 1;
    int mod_output = output_size / 10;

    for (int i = 0; i < this->getR(); i++) {

      // b.getC is actually row of internal transposed second matrix
      int sub_products = b.getC(); // number of sub dot products
      int split = sub_products / Blas::getTCount();
      int remain = split + (sub_products % Blas::getTCount());

      int col_size = b.getR();          // col size of second transposed matrix
      std::vector<std::thread> threads; // keep in scope

      // split dot products amongst cores
      for (int j = 0; j < Blas::getTCount(); j++) {
        /*
        adx doesn't change in this inner loop
        iterate through columns of second matrix
        */
        int aEnd = this->getC(); // position of first matrix column end

        if (j == Blas::getTCount() - 1) { // last thread
          threads.push_back(std::thread(
              t_mult<std::vector<long double>::iterator>, x.begin() + adx, aEnd,
              y.begin() + bdx, remain, j, col_size, std::ref(out)));
        } else {
          threads.push_back(std::thread(
              t_mult<std::vector<long double>::iterator>, x.begin() + adx, aEnd,
              y.begin() + bdx, split, j, col_size, std::ref(out)));
        }
        bdx += col_size * split;
      }

      // Wait for completion
      for (auto &thread : threads) {
        thread.join();
      }

      threads.clear(); // Remove finished threads

      // prod_count += ((threadcount - 1) * split) + remain; --- Progress output
      // not working

      // THIS NEEDS TO BE AFTER THREADS FINISH'
      // Reset ID
      counter = 0;
      bdx = 0;
      adx += this->getC();
    }

    return out;
  }

  /*
  Initalize Matrix with 2d vector
  Overloaded constructor

  */
  Matrix(const std::vector<std::vector<long double>> &a)
      : Rmt(&_iRmtx), Cmt(&_iCmtx) {

    _R = a.size();    // rows
    _C = a[0].size(); // columns

    Cmt->resize(_R * _C);

    for (int r = 0; r < _R; ++r) {
      for (int c = 0; c < _C; ++c) // filled row
      {
        Rmt->push_back(a[r][c]);
        Cmt->operator[]((c * _R) + r) = a[r][c];
      }
    }
  }

  // for std::cout
  friend std ::ostream &operator<<(std::ostream &o, Matrix<T> &n) {

    // Get Convert Matrix Internal active vector to reference
    const std::vector<T> &r = *(n.Rmt);

    int indx = n.getC() - 1;
    for (int i = 0; i < r.size(); i++) {
      o << std::right << std::setw(12) << r[i] << ' ';
      if (i != r.size() - 1)
        o << std::setfill(' ');
      if (i == indx) {
        if (i != r.size() - 1)
          o << std::endl;
        indx += n.getC();
      }
    }
    return o;
  }

  // Thread multiply
  template <typename TContainer>
  static void t_mult(TContainer a, int aEnd, TContainer b, int num_dot_prod,
                     int ID, int colsize, Matrix<> &out) {

    // Product list output of single THREAD
    std::vector<long double> prod_list;
    prod_list.reserve(num_dot_prod);

    int bdx = 0; // Increment index of second matrix columns

    /*
          compute products parallel
          second matrix start point need to be shifted
    */
    for (int j = 0; j < num_dot_prod; j++) {
      // m.lock(); hmm
      prod_list.push_back(std::inner_product(a, a + aEnd, b + bdx, 0.0));
      // m.unlock();
      bdx += colsize;
    }

    // wait to store in order
    if (ID != 0) {
      s[ID].wait(); // block self
    }

    // Store in order
    m.lock();
    for (auto x : prod_list) {
      out.addElement(x);
    }
    counter++;
    m.unlock();

    // wake up next thread
    if (counter < Blas::getTCount())
      s[counter].notify();
  }
};

// Generic Function Timer
template <typename F, typename... Args> double timef(F func, Args &&... args) {
  // Time measurement
  struct timespec start, finish;
  double elapsed;

  clock_gettime(CLOCK_MONOTONIC, &start);
  func(std::forward<Args>(args)...);
  clock_gettime(CLOCK_MONOTONIC, &finish);

  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
  return elapsed;
}

// For Timing Function with * operator
template <typename T> void mult(T &a, T &b, T &c) { c = a * b; }

// Test program
int main() {

  // Eigen Parallel?
  Eigen::initParallel();

  //  std::random_device rd;

  // RANDOM NUMBER GENERATION FROM STACK
  // OVERFLOW-------------------------------
  typedef std::mt19937
      MyRNG; // the Mersenne Twister with a popular choice of parameters
  uint32_t seed_val = 5; // populate somehow

  MyRNG rng; // e.g. keep one global instance (per thread)

  rng.seed(seed_val);
  std::uniform_int_distribution<long long int>
      uint_dist; // by default range [0, MAX]
  //----------------------------------------------------------------------------

  std::cout << LLONG_MAX << " " << std::endl;

  unsigned int DIMX, DIMY;

  std::cout << "Test program with one matrix M with any specified dimension"
            << std::endl;

  std::cout << "And then make a copy, N. Transpose M and multiply them "
               "together (M' x N)"
            << std::endl;

  std::cout << "Please enter matrix dimensions (\"M N\"): ";

  std::cin >> DIMX >> DIMY;

  // Initalize Library Matrices
  MatrixXd a(DIMX, DIMY);
  MatrixXd b(DIMX, DIMY);

  /*
   Can also input as 2D array
   Ex:
   long double temp[DIMX][DIMY];
   auto temp = new long double[DIMX][DIMY];
  */

  unsigned int rndnum;
  std::vector<std::vector<long double>> matrix;
  matrix.resize(DIMX); // resize top level vector

  // Fill random numbers for specified dimension
  for (int i = 0; i < DIMX; i++) {
    matrix[i].resize(DIMY); // resize each of the contained vectors
    for (int j = 0; j < DIMY; j++) {
      rndnum = uint_dist(rng);
      a(i, j) = rndnum;
      matrix[i][j] = rndnum;
    }
  }

  // 2d vector arg constructor
  Matrix<> m(matrix);
  Matrix<> n;
  n = m; // assignment operator

  // Output Matrix Mine
  Matrix<> result;

  // Output Matrix Library
  MatrixXd lib_out;

  // Initalize number of threads
  Blas::init();

  // library assignment operator
  b = a;

  // File output option.
  char in;
  bool f_out = false;
  std::cout << "Output to file? (y/n): ";
  std::cin >> in;
  if (in == 'y')
    f_out = true;

  // Avoid std::cout printing for large matrix
  bool large = DIMX * DIMY > 200 ? true : false;

  // Printing if Matrix is small
  if (!large) {
    std::cout << "My Matrix Print: " << std::endl;
    std::cout << m << std::endl;

    std::cout << "My Matrix Transpose M: " << std::endl;
    m.transpose();
    std::cout << m << std::endl;

    std::cout << "Library Transpose: " << std::endl;
    a = b.transpose();
    std::cout << a << std::endl;

    std::cout << "My Matrix Multiply (M' * N) output: " << std::endl;
    result = m * n;
    std::cout << result << std::endl;

    std::cout << "Library Multiply output: " << std::endl;
    std::cout << a * b << std::endl;

  } else {
    m.transpose();
    a = b.transpose();

    std::cout << "TIME Comparison: " << std::endl;

    // to time: lib_out = a * b
    std::cout << "EIGEN LIBRARY:   " << timef(mult<MatrixXd>, a, b, lib_out)
              << std::endl;

    // to time: result = m * n
    std::cout << "My MULTI THREAD: " << timef(mult<Matrix<>>, m, n, result)
              << std::endl;
  }
  // Fileoutput
  if (f_out) {
    std::ofstream libfile;
    libfile.open("lib.txt");
    libfile << lib_out;
    libfile.close();

    std::ofstream myfile;
    myfile.open("mine.txt");
    myfile << result;
    myfile.close();
  }

  return 0;
}
