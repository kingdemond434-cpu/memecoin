//! Generate the Yellowstone client from an EXACT descriptor set.
//!
//! `proto/geyser.fds` is a serialised `FileDescriptorSet` produced from the
//! descriptors the Python client already uses, so the Rust and Python sides
//! are generated from the same schema by construction rather than from two
//! hand-maintained copies that drift.
//!
//! Reconstructing `.proto` text from those descriptors was the obvious other
//! route and is strictly worse: it is a lossy re-derivation of something we
//! already hold exactly, and any mistake in it shows up as a field that
//! silently decodes as absent.

fn main() {
    #[cfg(feature = "ingress")]
    {
        let fds_path = std::path::Path::new("proto/geyser.fds");
        println!("cargo:rerun-if-changed=proto/geyser.fds");
        let bytes = std::fs::read(fds_path).expect("proto/geyser.fds is missing");
        let fds = <prost_types::FileDescriptorSet as prost::Message>::decode(&bytes[..])
            .expect("proto/geyser.fds did not decode as a FileDescriptorSet");
        tonic_prost_build::configure()
            .build_server(false)
            .build_client(true)
            .compile_fds(fds)
            .expect("failed to generate the Geyser client");
    }
}
