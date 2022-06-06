use std::collections::HashMap;

#[derive(Debug, Clone)]
enum ReplacementFormatter {
    None,
    Formatter(String),
    Condition(Option<String>, Option<String>),
}

impl ReplacementFormatter {
    pub fn format(&self, ctx: &SnippetContext, txt: &str) -> Cow<'_, str> {
        match self {
            ReplacementFormatter::None => Cow::Borrow(txt),
            ReplacementFormatter::Formatter(fmt) => {
                if let Some(fmt) = ctx.get_formatter() {
                    fmt(txt)
                } else {
                    Cow::Borrow(txt)
                }
            }
            ReplacementFormatter::Condition(on_true, on_false) => {
                if txt.trim().len() > 0 {
                    if let Some(repl) = on_true {
                        repl.into()
                    } else {
                        Cow::Borrow(txt)
                    }
                } else {
                    if let Some(repl) = on_false {
                        repl.into()
                    } else {
                        Cow::Borrow(txt)
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
enum TransformReplacement {
    Text(String),
    Replacement(i32, ReplacementFormatter),
}

#[derive(Debug, Clone)]
struct TabstopTransform {
    regex: Regex,
    replacement: Vec<TransformReplacement>,
}

impl TabstopTransform {
    pub fn new(regex: &str, entries: Vec<()>) -> Result<Self, Error> {
        Ok(Self {
            regex: Regex::new(regex)?,
            replacement: vec![],
        })
    }

    pub fn transform(&self, ctx: &SnippetContext, input: &str) -> String {
        self.regex.replace_all(input, move |caps: &Captures| {
            let mut repl = String::new();
            for (idx, rep) in self.replacemen.iter().enumerate() {
                match rep {
                    TransformReplacement::Text(txt) => repl.push_str(txt),
                    TransformReplacement::Replacement(index, fmt) => {
                        repl.push_str(fmt.format(ctx, caps[index]))
                    }
                }
            }
            repl
        })
    }
}

#[derive(Debug, Clone)]
struct SnippetTabstop {
    index: i32,
    placeholder: Option<String>,
    transform: Option<TabstopTransform>,
    selection: Selection,
}

pub struct SnippetContext {
    formatters: HashMap<Cow<'static, str>, Box<dyn for<'a> Fn(&'a str) -> Cow<'a, str>>>,
    variables: HashMap<Cow<'static, str>, String>,
}

pub struct Snippet {
    template: String,
    tabstops: Vec<Selection>,
    context: SnippetContext,
}

impl Snippet {
    pub fn new() {}
}
