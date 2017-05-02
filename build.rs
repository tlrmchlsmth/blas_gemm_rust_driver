fn main() -> () {
    //Link with MKL:
    println!("cargo:rustc-link-search=native=/opt/intel/mkl/lib/intel64/");
    println!("cargo:rustc-link-lib=dylib=mkl_intel_ilp64");
    println!("cargo:rustc-link-lib=dylib=mkl_intel_thread");
    println!("cargo:rustc-link-lib=dylib=mkl_core");

    println!("cargo:rustc-link-search=native=/opt/intel/lib/intel64/");
    println!("cargo:rustc-link-lib=dylib=iomp5");
}
