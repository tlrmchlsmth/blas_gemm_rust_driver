#![feature(libc)]

//#![allow(unused_imports)]
extern crate libc;
extern crate momms;

use std::time::{Instant};
use libc::{c_double, int64_t, c_char};
use std::ffi::CString;
use momms::matrix::{Mat, Scalar, Matrix, RoCM};
use momms::util;


extern{
    fn dgemm( transa: *const c_char, transb: *const c_char,
               m: *const int64_t, n: *const int64_t, k: *const int64_t,
               alpha: *const c_double, 
               a: *const c_double, lda: *const int64_t,
               b: *const c_double, ldb: *const int64_t,
               beta: *const c_double,
               c: *mut c_double, ldc: *const int64_t );
}

pub fn blas_dgemm( a: &mut Matrix<f64>, b: &mut Matrix<f64>, c: &mut Matrix<f64> ) 
{
    unsafe{ 
        let transa = CString::new("N").unwrap();
        let transb = CString::new("N").unwrap();
        let ap = a.get_mut_buffer();
        let bp = b.get_buffer();
        let cp = c.get_buffer();

        let lda = a.get_column_stride() as int64_t;
        let ldb = b.get_column_stride() as int64_t;
        let ldc = c.get_column_stride() as int64_t;

        let m = c.height() as int64_t;
        let n = b.width() as int64_t;
        let k = a.width() as int64_t;

        let alpha: f64 = 1.0;
        let beta: f64 = 1.0;
    
        dgemm( transa.as_ptr() as *const c_char, transb.as_ptr() as *const c_char,
                &m, &n, &k,
                &alpha as *const c_double, 
                ap as *const c_double, &lda,
                bp as *const c_double, &ldb,
                &beta as *const c_double,
                cp as *mut c_double, &ldc );
    }
}

fn test_blas_dgemm ( m:usize, n: usize, k: usize, flusher: &mut Vec<f64>, n_reps: usize ) -> (f64, f64) 
{
    let mut best_time: f64 = 9999999999.0;
    let mut worst_err: f64 = 0.0;

    for _ in 0..n_reps {
        //Create matrices.
        let mut a : Matrix<f64> = Matrix::new(m, k);
        let mut b : Matrix<f64> = Matrix::new(k, n);
        let mut c : Matrix<f64> = Matrix::new(m, n);

        //Fill the matrices
        a.fill_rand(); c.fill_zero(); b.fill_rand();

        //Read a buffer so that A, B, and C are cold in cache.
        for i in flusher.iter_mut() { *i += 1.0; }
            
        //Time and run algorithm
        let start = Instant::now();
        blas_dgemm( &mut a, &mut b, &mut c);
        best_time = best_time.min(util::dur_seconds(start));
        let err = util::test_c_eq_a_b( &mut a, &mut b, &mut c);
        worst_err = worst_err.max(err);
    }
    (best_time, worst_err)
}

fn test() {
    //Initialize array to flush cache with
    let flusher_len = 2*1024*1024; //16MB
    let mut flusher : Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len { flusher.push(0.0); }

    println!("m\tn\tk\t{: <13}{: <15}", "goto", "l3b", "goto", "l3b");
    for index in 01..129 {
        let size = index*32;
        let (m, n, k) = (size, size, size);

        let n_reps = 5;
        let (mkl_time, mkl_err) = test_blas_dgemm(m, n, k, &mut flusher, n_reps);

        println!("{}\t{}\t{}\t{}{}", 
                 m, n, k,
                 format!("{: <13.5}", util::gflops(m,n,k,mkl_time)),
                 format!("{: <15.5e}", mkl_err.sqrt()));
    }

    let mut sum = 0.0;
    for a in flusher.iter() {
        sum += *a;
    }
    println!("Flush value {}", sum);
}

fn main() {
    test( );
}
