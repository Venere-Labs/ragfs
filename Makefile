.PHONY: fmt fmt-check clippy test deny doc ci

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --all --all-features

deny:
	cargo deny check

doc:
	cargo doc --no-deps --all-features

ci: fmt-check clippy test deny
