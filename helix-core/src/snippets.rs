use std::{
    borrow::Cow,
    cell::{Ref, RefCell},
    collections::HashMap,
    rc::Rc,
};

use regex::{Captures, Regex};

#[derive(Debug)]
pub enum TokenKind {
    Symbol(char),
    Ident,
    Integer,
    Text,
    TextEnding(String),
}

#[derive(Debug)]
pub enum ParseErrorKind {
    ExpectedToken(TokenKind),
    ExpectedOneOf(Vec<TokenKind>),
}

#[derive(Debug)]
pub struct ParseError<'a> {
    kind: ParseErrorKind,
    code: &'a str,
}

pub type ParseResult<'a, R> = Result<(&'a str, R), ParseError<'a>>;

impl<'a> ParseError<'a> {
    pub fn new_tok(tok: TokenKind, code: &'a str) -> Self {
        Self {
            kind: ParseErrorKind::ExpectedToken(tok),
            code,
        }
    }
}

pub fn sym<const C: char>(input: &str) -> ParseResult<()> {
    match input.chars().next() {
        Some(next) if next == C => Ok((&input[C.len_utf8()..], ())),
        _ => Err(ParseError::new_tok(TokenKind::Symbol(C), input)),
    }
}

pub fn ident(input: &str) -> ParseResult<&str> {
    let mut length = 0;
    let mut chars = input.chars();

    match chars.next() {
        Some(next) if next.is_alphabetic() || next == '_' => {
            length = length + next.len_utf8();
        }

        _ => return Err(ParseError::new_tok(TokenKind::Ident, input)),
    }

    while let Some(next) = chars.next() {
        if next.is_alphanumeric() || next == '_' {
            length = length + next.len_utf8();
        } else {
            break;
        }
    }

    Ok((&input[length..], &input[..length]))
}

pub fn integer(input: &str) -> ParseResult<i32> {
    let mut matched = String::new();
    let mut chars = input.chars();

    match chars.next() {
        Some(next) if next.is_digit(10) => matched.push(next),
        _ => return Err(ParseError::new_tok(TokenKind::Integer, input)),
    }

    while let Some(next) = chars.next() {
        if next.is_digit(10) {
            matched.push(next);
        } else {
            break;
        }
    }

    let next_index = matched.len();

    Ok((&input[next_index..], matched.parse().unwrap()))
}

pub fn seq<'a, P1, P2, R1, R2>(p1: P1, p2: P2) -> impl Fn(&'a str) -> ParseResult<(R1, R2)> + Clone
where
    P1: Fn(&'a str) -> ParseResult<'a, R1> + Clone,
    P2: Fn(&'a str) -> ParseResult<'a, R2> + Clone,
    R1: 'a,
    R2: 'a,
{
    move |input| {
        let (next_input, res1) = p1(input)?;
        let (next_input, res2) = p2(next_input)?;

        Ok((next_input, (res1, res2)))
    }
}

pub fn left<'a, P1, P2, R1, R2>(p1: P1, p2: P2) -> impl Fn(&'a str) -> ParseResult<R1> + Clone
where
    P1: Fn(&'a str) -> ParseResult<'a, R1> + Clone,
    P2: Fn(&'a str) -> ParseResult<'a, R2> + Clone,
    R1: 'a,
    R2: 'a,
{
    map(seq(p1, p2), |(l, _)| l)
}

pub fn right<'a, P1, P2, R1, R2>(p1: P1, p2: P2) -> impl Fn(&'a str) -> ParseResult<'a, R2> + Clone
where
    P1: Fn(&'a str) -> ParseResult<'a, R1> + Clone,
    P2: Fn(&'a str) -> ParseResult<'a, R2> + Clone,
    R1: 'a,
    R2: 'a,
{
    map(seq(p1, p2), |(_, r)| r)
}

pub fn any<'a, P1, P2, R: 'a>(p1: P1, p2: P2) -> impl Fn(&'a str) -> ParseResult<R> + Clone
where
    P1: Fn(&'a str) -> ParseResult<'a, R> + Clone,
    P2: Fn(&'a str) -> ParseResult<'a, R> + Clone,
{
    use ParseErrorKind::*;

    move |input| match p1(input) {
        Ok((next_input, res)) => Ok((next_input, res)),
        Err(err1) => match p2(input) {
            Ok((next_input, res)) => Ok((next_input, res)),
            Err(err2) => {
                let toks = match (err1.kind, err2.kind) {
                    (ExpectedOneOf(mut toks1), ExpectedOneOf(mut toks2)) => {
                        toks1.append(&mut toks2);
                        toks1
                    }
                    (ExpectedOneOf(mut toks), ExpectedToken(tok)) => {
                        toks.push(tok);
                        toks
                    }
                    (ExpectedToken(tok), ExpectedOneOf(mut toks)) => {
                        toks.push(tok);
                        toks
                    }
                    (ExpectedToken(tok1), ExpectedToken(tok2)) => {
                        vec![tok1, tok2]
                    }
                };

                Err(ParseError {
                    kind: ExpectedOneOf(toks),
                    code: input,
                })
            }
        },
    }
}

pub fn map<'a, P, F, A, B>(parser: P, map_fn: F) -> impl Fn(&'a str) -> ParseResult<'a, B> + Clone
where
    P: Fn(&'a str) -> ParseResult<'a, A> + Clone,
    F: Fn(A) -> B + Clone,
    A: 'a,
    B: 'a,
{
    move |input| parser(input).map(|(next_input, result)| (next_input, map_fn(result)))
}

pub fn optional<'a, P, A>(parser: P) -> impl Fn(&'a str) -> ParseResult<'a, Option<A>> + Clone
where
    P: Fn(&'a str) -> ParseResult<'a, A> + Clone,
    A: 'a,
{
    move |input| {
        Ok(match parser(input) {
            Ok((i, r)) => (i, Some(r)),
            Err(_) => (input, None),
        })
    }
}

pub fn many<'a, P, A>(
    parser: P,
    required: bool,
) -> impl Fn(&'a str) -> ParseResult<'a, Vec<A>> + Clone
where
    P: Fn(&'a str) -> ParseResult<'a, A> + Clone,
    A: 'a,
{
    move |mut input| {
        let mut result = Vec::new();

        if required {
            match parser(input) {
                Ok((next_input, first_item)) => {
                    if next_input.len() != input.len() {
                        input = next_input;
                        result.push(first_item);
                    }
                }
                Err(err) => return Err(err),
            }
        }

        while let Ok((next_input, next_item)) = parser(input) {
            if next_input.len() == input.len() {
                break;
            }

            input = next_input;
            result.push(next_item);
        }

        Ok((input, result))
    }
}

pub fn separated<'a, P, S, A, B>(
    sep: S,
    parser: P,
) -> impl Fn(&'a str) -> ParseResult<'a, Vec<A>> + Clone
where
    P: Fn(&'a str) -> ParseResult<'a, A> + Clone,
    S: Fn(&'a str) -> ParseResult<'a, B> + Clone,
    A: 'a,
    B: 'a,
{
    let parse_next = seq(sep.clone(), parser.clone());

    move |mut input| {
        let mut result = Vec::new();

        match parser(input) {
            Ok((next_input, first_item)) => {
                if next_input.len() != input.len() {
                    input = next_input;
                    result.push(first_item);
                }
            }
            Err(err) => return Err(err),
        }

        loop {
            if let Ok((next_input, (_, next_item))) = parse_next(input) {
                if next_input.len() == input.len() {
                    break;
                }

                input = next_input;
                result.push(next_item);
            } else {
                break;
            }
        }

        Ok((input, result))
    }
}

pub fn text<P>(pat: P, required: bool) -> impl Fn(&str) -> ParseResult<&str> + Clone
where
    P: AsRef<[char]> + Clone,
{
    move |input| {
        let mut prev = '\0';
        let pat = pat.as_ref();

        if let Some(index) = input.find(|x: char| {
            let val = pat.contains(&x) && prev != '\\';
            prev = x;
            val
        }) {
            if required && index == 0 {
                return Err(ParseError::new_tok(TokenKind::Text, input));
            }

            return Ok((&input[index..], &input[0..index]));
        }

        Err(ParseError::new_tok(
            TokenKind::TextEnding(pat.iter().copied().collect()),
            input,
        ))
    }
}

trait SnippetEval<'a> {
    fn eval(&self, ctx: &'a SnippetContext<'a>) -> Cow<'a, str>;
}

#[derive(Debug, Clone)]
enum TransformFormatFilter<'a> {
    Formatter(&'a str),
    Condition(Option<&'a str>, Option<&'a str>),
}

impl<'a> TransformFormatFilter<'a> {
    pub fn format<'b>(&self, ctx: &SnippetContext, txt: &'b str) -> Cow<'b, str> {
        match self {
            TransformFormatFilter::Formatter(fmt) => {
                if let Some(fmt) = ctx.get_formatter(fmt) {
                    fmt(txt)
                } else {
                    Cow::Borrowed(txt)
                }
            }
            TransformFormatFilter::Condition(on_true, on_false) => {
                if txt.trim().len() > 0 {
                    if let Some(repl) = on_true {
                        repl.to_string().into()
                    } else {
                        Cow::Borrowed(txt)
                    }
                } else {
                    if let Some(repl) = on_false {
                        repl.to_string().into()
                    } else {
                        Cow::Borrowed(txt)
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
enum TransformEntry<'a> {
    Text(&'a str),
    Format(i32, Option<TransformFormatFilter<'a>>),
}

#[derive(Debug, Clone)]
struct Transform<'a> {
    pub regex: &'a str,
    pub entries: Vec<TransformEntry<'a>>,
    pub options: &'a str,
}

impl<'a> Transform<'a> {
    pub fn transform<'b>(&self, ctx: &SnippetContext<'a>, input: &'b str) -> Cow<'b, str> {
        let regex = ctx.get_or_create_regex(
            self.regex,
            self.options.contains('i'),
            self.options.contains('m'),
        );

        let regex = if let Some(regex) = regex {
            regex
        } else {
            return Cow::Borrowed(input);
        };

        let global = if self.options.contains('g') { 0 } else { 1 };

        regex.replacen(input, global, move |caps: &Captures| {
            let mut repl = String::new();
            for rep in self.entries.iter() {
                match rep {
                    TransformEntry::Text(txt) => repl.push_str(txt),
                    TransformEntry::Format(index, fmt) => {
                        let entry = &caps[*index as usize];

                        if let Some(fmt) = fmt {
                            repl.push_str(&fmt.format(ctx, entry));
                        } else {
                            repl.push_str(entry);
                        }
                    }
                }
            }
            repl
        })
    }
}

struct SnippetContext<'a> {
    regexes: RefCell<HashMap<(&'a str, bool, bool), Option<Rc<Regex>>>>,
    formatters: HashMap<Cow<'static, str>, Box<dyn for<'f> Fn(&'f str) -> Cow<'f, str>>>,
    variables: HashMap<Cow<'static, str>, String>,
    tabstops: RefCell<HashMap<i32, String>>,
}

impl<'a> SnippetContext<'a> {
    pub fn new() -> Self {
        Self {
            regexes: RefCell::new(HashMap::new()),
            formatters: HashMap::new(),
            variables: HashMap::new(),
            tabstops: RefCell::new(HashMap::new()),
        }
    }

    pub fn add_variable(&mut self, name: Cow<'static, str>, value: String) {
        self.variables.insert(name, value);
    }
    pub fn add_formatter(
        &mut self,
        name: Cow<'static, str>,
        fmt: Box<dyn for<'f> Fn(&'f str) -> Cow<'f, str>>,
    ) {
        self.formatters.insert(name, fmt);
    }

    fn get_or_create_regex(
        &self,
        regex: &'a str,
        ignore_case: bool,
        multiline: bool,
    ) -> Option<Rc<Regex>> {
        self.regexes
            .borrow_mut()
            .entry((regex, ignore_case, multiline))
            .or_insert_with(|| Regex::new(regex).ok().map(Rc::new))
            .clone()
    }

    #[inline]
    fn get_formatter(&self, fmt: &str) -> Option<&dyn for<'f> Fn(&'f str) -> Cow<'f, str>> {
        self.formatters.get(fmt).map(|x| x as _)
    }

    #[inline]
    fn get_variable(&self, name: &str) -> Option<&str> {
        self.variables.get(name).map(String::as_str)
    }

    #[inline]
    fn get_tabstop(&self, index: i32) -> Option<Ref<str>> {
        let val = self.tabstops.borrow();
        if !val.contains_key(&index) {
            return None;
        }

        Some(Ref::map(val, |ts| ts[&index].as_str()))
    }

    #[inline]
    fn set_tabstop(&self, index: i32, ts: &str) {
        self.tabstops.borrow_mut().insert(index, ts.to_string());
    }
}

#[derive(Debug, Clone)]
enum VariableExt<'a> {
    Transform(Transform<'a>),
    Placeholder(Vec<Token<'a>>),
}

impl<'a> VariableExt<'a> {
    pub fn transform(&self, ctx: &'a SnippetContext<'a>, value: &'a str) -> Cow<'a, str> {
        match self {
            VariableExt::Transform(tr) => tr.transform(ctx, value),
            VariableExt::Placeholder(ph) => {
                if value.is_empty() {
                    ph.eval(ctx)
                } else {
                    Cow::Borrowed(value)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Variable<'a> {
    pub name: &'a str,
    pub ext: Option<VariableExt<'a>>,
}

impl<'a> Variable<'a> {
    pub fn new(name: &'a str, ext: Option<VariableExt<'a>>) -> Self {
        Self { name, ext }
    }

    pub fn new_simple(name: &'a str) -> Self {
        Self { name, ext: None }
    }
}

impl<'a> SnippetEval<'a> for Variable<'a> {
    fn eval(&self, ctx: &'a SnippetContext<'a>) -> Cow<'a, str> {
        ctx.get_variable(self.name)
            .map(|val| match &self.ext {
                Some(ext) => ext.transform(ctx, val),
                None => Cow::Borrowed(val),
            })
            .unwrap_or(Cow::Borrowed(self.name))
    }
}

#[derive(Debug, Clone)]
enum TabstopExt<'a> {
    Transform(Transform<'a>),
    Placeholder(Vec<Token<'a>>),
    Choice(Vec<&'a str>),
}

impl<'a> TabstopExt<'a> {
    pub fn eval(&self, ctx: &'a SnippetContext<'a>, index: i32) -> Cow<'a, str> {
        match self {
            TabstopExt::Transform(ts) => {
                if let Some(ts_val) = ctx.get_tabstop(index) {
                    Cow::Owned(ts.transform(ctx, &ts_val).to_string())
                } else {
                    Cow::Borrowed("")
                }
            }
            TabstopExt::Placeholder(ph) => ph.eval(ctx),
            TabstopExt::Choice(ch) => Cow::Borrowed(ch.get(0).map(|x| *x).unwrap_or("")),
        }
    }
}

#[derive(Debug, Clone)]
struct Tabstop<'a> {
    pub index: i32,
    pub ext: Option<TabstopExt<'a>>,
}

impl<'a> Tabstop<'a> {
    pub fn new(index: i32, ext: Option<TabstopExt<'a>>) -> Self {
        Self { index, ext }
    }

    pub fn new_simple(index: i32) -> Self {
        Self { index, ext: None }
    }
}

impl<'a> SnippetEval<'a> for Tabstop<'a> {
    fn eval(&self, ctx: &'a SnippetContext<'a>) -> Cow<'a, str> {
        if let Some(val) = ctx.get_tabstop(self.index) {
            Cow::Owned(val.to_string())
        } else if let Some(ext) = &self.ext {
            let ts = ext.eval(ctx, self.index);
            if !ts.is_empty() {
                ctx.set_tabstop(self.index, &ts);
            }
            ts
        } else {
            Cow::Borrowed("")
        }
    }
}

#[derive(Debug, Clone)]
enum Token<'a> {
    Tabstop(Tabstop<'a>),
    Variable(Variable<'a>),
    Text(&'a str),
}

impl<'a> SnippetEval<'a> for Token<'a> {
    fn eval(&self, ctx: &'a SnippetContext<'a>) -> Cow<'a, str> {
        match self {
            Token::Tabstop(ts) => ts.eval(ctx),
            Token::Variable(var) => var.eval(ctx),
            Token::Text(txt) => Cow::Borrowed(txt),
        }
    }
}

impl<'a> SnippetEval<'a> for Vec<Token<'a>> {
    fn eval(&self, ctx: &'a SnippetContext<'a>) -> Cow<'a, str> {
        if self.is_empty() {
            Cow::Borrowed("")
        } else if self.len() == 1 {
            self[0].eval(ctx)
        } else {
            let mut res = String::new();

            for tok in self {
                res.push_str(&tok.eval(ctx));
            }

            res.into()
        }
    }
}

///
/// any         ::= tabstop | placeholder | choice | variable | text
///
fn token<'a, P: AsRef<[char]> + Clone>(
    pat: P,
) -> impl Fn(&'a str) -> ParseResult<'a, Token<'a>> + Clone {
    any(
        any(
            map(variable(), Token::Variable),
            map(tabstop(), Token::Tabstop),
        ),
        map(text(pat, true), Token::Text),
    )
}

///
/// tabstop     ::=   '$' int
///                 | '${' int '}'
///                 | '${' int transform '}'
///                 | '${' int '|' text (',' text)* '|}'
///                 | '${' int ':' any '}'
///
fn tabstop<'a>() -> impl Fn(&'a str) -> ParseResult<'a, Tabstop<'a>> + Clone {
    let dollar = sym::<'$'>;
    let open = sym::<'{'>;
    let close = sym::<'}'>;
    let colon = sym::<':'>;
    let pipe = sym::<'|'>;
    let comma = sym::<','>;
    let txt = text([',', '|'], true);

    let choice = left(right(pipe, separated(comma, txt)), pipe);
    let placeholder = right(colon, |s| snippet('}')(s));

    let ext = any(
        any(
            map(choice, TabstopExt::Choice),
            map(placeholder, TabstopExt::Placeholder),
        ),
        map(transform(), TabstopExt::Transform),
    );

    right(
        dollar,
        any(
            map(integer, Tabstop::new_simple),
            map(
                seq(right(open, integer), left(optional(ext), close)),
                |(a, b)| Tabstop::new(a, b),
            ),
        ),
    )
}

///
/// variable    ::=   '$' var
///                 | '${' var '}'
///                 | '${' var ':' any '}'
///                 | '${' var transform '}'
///
fn variable<'a>() -> impl Fn(&'a str) -> ParseResult<'a, Variable<'a>> + Clone {
    let dollar = sym::<'$'>;
    let open = sym::<'{'>;
    let close = sym::<'}'>;
    let colon = sym::<':'>;

    let colon = right(colon, |s| snippet('}')(s));

    right(
        dollar,
        any(
            map(
                seq(
                    right(open, ident),
                    left(
                        optional(any(
                            map(colon, VariableExt::Placeholder),
                            map(transform(), VariableExt::Transform),
                        )),
                        close,
                    ),
                ),
                |(n, ext)| Variable::new(n, ext),
            ),
            map(ident, Variable::new_simple),
        ),
    )
}

///
/// transform   ::= '/' regex '/' (format | text)+ '/' options
///
fn transform<'a>() -> impl Fn(&'a str) -> ParseResult<'a, Transform<'a>> + Clone {
    let slash = sym::<'/'>;

    map(
        seq(
            seq(
                // regex
                right(slash, text(['/'], true)),
                // replacement
                right(
                    slash,
                    many(
                        any(
                            map(format(), |(i, f)| TransformEntry::Format(i, f)),
                            map(text(['$', '/'], true), TransformEntry::Text),
                        ),
                        false,
                    ),
                ),
            ),
            // options
            right(slash, text(['}'], false)),
        ),
        |((regex, entries), options)| Transform {
            regex,
            entries,
            options,
        },
    )
}

///
/// format      ::=   '$' int
///                 | '${' int '}'
///                 | '${' int ':' '/upcase' | '/downcase' | '/capitalize' | '/camelcase' | '/pascalcase' '}'
///                 | '${' int ':+' if '}'
///                 | '${' int ':?' if ':' else '}'
///                 | '${' int ':-' else '}'
///                 | '${' int ':' else '}'
///
fn format<'a>(
) -> impl Fn(&'a str) -> ParseResult<'a, (i32, Option<TransformFormatFilter<'a>>)> + Clone {
    let dollar = sym::<'$'>;
    let open = sym::<'{'>;
    let close = sym::<'}'>;

    let parse_colon = any(
        map(
            right(sym::<'/'>, text(['}'], true)),
            TransformFormatFilter::Formatter,
        ),
        any(
            any(
                map(right(sym::<'+'>, text(['}'], false)), |txt| {
                    TransformFormatFilter::Condition(Some(txt), None)
                }),
                map(
                    seq(
                        right(sym::<'?'>, text([':'], false)),
                        right(sym::<':'>, text(['}'], false)),
                    ),
                    |(l, r)| TransformFormatFilter::Condition(Some(l), Some(r)),
                ),
            ),
            any(
                map(right(sym::<'-'>, text(['}'], false)), |txt| {
                    TransformFormatFilter::Condition(None, Some(txt))
                }),
                map(text(['}'], false), |txt| {
                    TransformFormatFilter::Condition(None, Some(txt))
                }),
            ),
        ),
    );

    right(
        dollar,
        any(
            right(
                open,
                seq(
                    integer,
                    left(optional(right(sym::<':'>, parse_colon)), close),
                ),
            ),
            map(integer, |i| (i, None)),
        ),
    )
}

fn snippet<'a>(stop_on: char) -> impl Fn(&'a str) -> ParseResult<'a, Vec<Token<'a>>> + Clone {
    many(token(['$', stop_on]), false)
}

fn main() {
    let (rest, snip) = snippet('\0')(
        "test ${TEST:default} ___ ${FILE/fuck_(.*)/$1 ${1} ${1:/upcase} ${1:+Hi} ${1:?Hi:Ho} ${1:Ho} ${1:-Ho}/} test ${1:andrey} ${2|a,b,c|} ${3:default} $3  $2 dasdas \0",
    )
    .unwrap();

    println!("snip {:?}", snip);
    println!("rest {}", rest);

    let mut ctx = SnippetContext::new();
    ctx.add_formatter("upcase".into(), Box::new(|x| x.to_uppercase().into()));
    ctx.add_variable("TEST".into(), "".into());
    ctx.add_variable("FILE".into(), "fuck_you".into());

    println!("{:?}", snip.eval(&ctx));
}
