use imara_diff::{Algorithm, BasicLineDiffPrinter, Diff, InternedInput, UnifiedDiffConfig};

/// Create a unified diff string between old and new text, using imara-diff.
pub fn create_patch(old: &str, new: &str) -> String {
    if old == new {
        return String::new();
    }

    let input = InternedInput::new(old, new);
    let mut diff = Diff::compute(Algorithm::Histogram, &input);
    diff.postprocess_lines(&input);

    diff.unified_diff(
        &BasicLineDiffPrinter(&input.interner),
        UnifiedDiffConfig::default(),
        &input,
    )
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_patch_identical() {
        let result = create_patch("hello", "hello");
        assert!(result.is_empty());
    }

    #[test]
    fn test_create_patch_simple_insert() {
        let result = create_patch("hello", "hello world");
        assert!(result.contains("+"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_create_patch_simple_delete() {
        let result = create_patch("hello world", "hello");
        assert!(result.contains("-"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_create_patch_multiline() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3";
        let result = create_patch(old, new);
        assert!(result.contains("-line2"));
        assert!(result.contains("+modified"));
    }
}
