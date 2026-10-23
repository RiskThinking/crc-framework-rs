fn main() {
    println!(
        "cargo:rustc-env=CRC_TARGET={}",
        std::env::var("TARGET").expect("Cargo sets TARGET for build scripts")
    );
}
