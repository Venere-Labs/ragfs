//! Query DSL parser.

use ragfs_core::SearchFilter;

/// Parsed query with text and filters.
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    /// Main query text
    pub text: String,
    /// Extracted filters
    pub filters: Vec<SearchFilter>,
    /// Result limit
    pub limit: usize,
}

/// Query parser for the DSL.
pub struct QueryParser {
    /// Default result limit
    default_limit: usize,
}

impl QueryParser {
    /// Create a new query parser.
    #[must_use]
    pub fn new(default_limit: usize) -> Self {
        Self { default_limit }
    }

    /// Parse a query string.
    ///
    /// Supports filters like:
    /// - `lang:rust` or `language:python`
    /// - `path:src/**`
    /// - `type:code` or `type:text`
    /// - `limit:10`
    #[must_use]
    pub fn parse(&self, query: &str) -> ParsedQuery {
        let mut text_parts = Vec::new();
        let mut filters = Vec::new();
        let mut limit = self.default_limit;

        for part in query.split_whitespace() {
            if let Some((key, value)) = part.split_once(':') {
                match key.to_lowercase().as_str() {
                    "lang" | "language" => {
                        filters.push(SearchFilter::Language(value.to_string()));
                    }
                    "path" => {
                        if value.contains('*') {
                            filters.push(SearchFilter::PathGlob(value.to_string()));
                        } else {
                            filters.push(SearchFilter::PathPrefix(value.to_string()));
                        }
                    }
                    "type" | "mime" => {
                        filters.push(SearchFilter::MimeType(value.to_string()));
                    }
                    "limit" => {
                        if let Ok(n) = value.parse() {
                            limit = n;
                        }
                    }
                    "depth" => {
                        if let Ok(n) = value.parse() {
                            filters.push(SearchFilter::MaxDepth(n));
                        }
                    }
                    _ => {
                        // Unknown filter, treat as text
                        text_parts.push(part);
                    }
                }
            } else {
                text_parts.push(part);
            }
        }

        ParsedQuery {
            text: text_parts.join(" "),
            filters,
            limit,
        }
    }
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let parser = QueryParser::default();
        let result = parser.parse("how to implement auth");

        assert_eq!(result.text, "how to implement auth");
        assert!(result.filters.is_empty());
        assert_eq!(result.limit, 10);
    }

    #[test]
    fn test_parse_with_filters() {
        let parser = QueryParser::default();
        let result = parser.parse("authentication lang:rust path:src/** limit:5");

        assert_eq!(result.text, "authentication");
        assert_eq!(result.filters.len(), 2);
        assert_eq!(result.limit, 5);
    }

    #[test]
    fn test_parse_empty_query() {
        let parser = QueryParser::default();
        let result = parser.parse("");

        assert_eq!(result.text, "");
        assert!(result.filters.is_empty());
        assert_eq!(result.limit, 10);
    }

    #[test]
    fn test_parse_language_filter() {
        let parser = QueryParser::default();
        let result = parser.parse("lang:rust");

        assert_eq!(result.text, "");
        assert_eq!(result.filters.len(), 1);
        assert!(matches!(&result.filters[0], SearchFilter::Language(l) if l == "rust"));
    }

    #[test]
    fn test_parse_path_prefix() {
        let parser = QueryParser::default();
        let result = parser.parse("path:src/lib");

        assert_eq!(result.filters.len(), 1);
        assert!(matches!(&result.filters[0], SearchFilter::PathPrefix(p) if p == "src/lib"));
    }

    #[test]
    fn test_parse_depth_filter() {
        let parser = QueryParser::default();
        let result = parser.parse("depth:2 search term");

        assert_eq!(result.text, "search term");
        assert_eq!(result.filters.len(), 1);
        assert!(matches!(&result.filters[0], SearchFilter::MaxDepth(2)));
    }

    #[test]
    fn test_parse_invalid_limit() {
        let parser = QueryParser::default();
        let result = parser.parse("limit:abc search");

        assert_eq!(result.text, "search");
        assert_eq!(result.limit, 10); // Default preserved when invalid
    }

    #[test]
    fn test_parse_unknown_filter_as_text() {
        let parser = QueryParser::default();
        let result = parser.parse("unknown:value search");

        assert_eq!(result.text, "unknown:value search");
        assert!(result.filters.is_empty());
    }
}
