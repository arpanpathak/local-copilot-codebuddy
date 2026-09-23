1. Never use `unwrap()` or `expect()`, not even in tests. Use `?` and return `Result`.
2. Errors are a custom `enum` that implements `std::fmt::Display` and `std::error::Error`. No external crates.
3. Every `pub` item has a `///` doc comment.
4. No comments inside function bodies.
5. No index loops like `for i in 0..n`; use iterators.
