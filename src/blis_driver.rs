#![feature(libc)]

//#![allow(unused_imports)]
extern crate libc;
extern crate momms;

use std::time::{Instant};
use momms::matrix::{Mat, Matrix};
use momms::util;

//Note: momms::util::blas_dgemm is BLIS since momms links to BLIS.
fn test_blis ( m:usize, n: usize, k: usize, flusher: &mut Vec<f64>, n_reps: usize ) -> (f64, f64) 
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
        util::blas_dgemm( &mut a, &mut b, &mut c);
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

    println!("m\tn\tk\t{: <13}{: <15}", "blis", "blis");
    for index in 01..81 {
        let size = index*50;
        let (m, n, k) = (size, size, size);

        let n_reps = 5;
        let (blis_time, blis_err) = test_blis(m, n, k, &mut flusher, n_reps);

        println!("{}\t{}\t{}\t{}{}", 
                 m, n, k,
                 format!("{: <13.5}", util::gflops(m,n,k,blis_time)),
                 format!("{: <15.5e}", blis_err.sqrt()));
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
