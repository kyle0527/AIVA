# AIVA Rust Analysis CLI Commands

## Category: ANALYSIS
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::new
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::analyze_file
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::has_clap_derive
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::extract_cli_info
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::extract_field_arg
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::extract_attr_value
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::type_to_string
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::is_option_type
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::is_bool_type
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::extract_subcommands
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::generate_cli_docs
cargo run --bin rs2mermaid -- --file .\src\main.rs --func ClapAnalyzer::to_json
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Stitcher::analyze_branches

## Category: OTHER
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Node::new
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Node::sanitize_id
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Graph::new
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Graph::add
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Graph::link
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Graph::to_mermaid
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Graph::format_node
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Graph::format_edge
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Stitcher::new
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Stitcher::add_script
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Stitcher::resolve_connections
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Stitcher::find_provider
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Stitcher::check_candidates
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Builder::new
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Builder::process_fn
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Builder::process_method
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Builder::visit_expr_if
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Builder::visit_expr_call
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Builder::visit_expr_method_call
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Builder::visit_expr_return
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Classifier::new
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Classifier::classify
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Classifier::get_result
cargo run --bin rs2mermaid -- --file .\src\main.rs --func main
cargo run --bin rs2mermaid -- --file .\src\paths_config.rs --func PathsConfig::new
cargo run --bin rs2mermaid -- --file .\src\paths_config.rs --func PathsConfig::find_project_root
cargo run --bin rs2mermaid -- --file .\src\paths_config.rs --func PathsConfig::ensure_directories
cargo run --bin rs2mermaid -- --file .\src\paths_config.rs --func PathsConfig::get_analysis_output_dir
cargo run --bin rs2mermaid -- --file .\src\paths_config.rs --func PathsConfig::get_default_output_dir
cargo run --bin rs2mermaid -- --file .\src\paths_config.rs --func PathsConfig::print_summary
cargo run --bin rs2mermaid -- --file .\src\paths_config.rs --func PathsConfig::default

## Category: RECONNAISSANCE
cargo run --bin rs2mermaid -- --file .\src\main.rs --func scan_files

## Category: REPORTING
cargo run --bin rs2mermaid -- --file .\src\main.rs --func Stitcher::generate_system_flow
cargo run --bin rs2mermaid -- --file .\src\main.rs --func generate_cli

