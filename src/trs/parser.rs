use super::lexicon::Lexicon;
use super::rewrite::TRS;
use polytype::{Context as TypeContext, TypeScheme};
use std::fmt;
use std::io;
use term_rewriting::{
    parse_context as parse_untyped_context, parse_rule as parse_untyped_rule,
    parse_rulecontext as parse_untyped_rulecontext, Atom, Context, Rule, RuleContext, Signature,
};
use winnow::{
    ascii::{multispace0, newline, not_line_ending},
    combinator::{alt, delimited, opt, preceded, repeat, separated_pair, terminated},
    error::ErrMode,
    prelude::*,
    token::{take_till, take_while},
};

#[derive(Debug, PartialEq)]
/// The error type for parsing operations.
pub struct ParseError;
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "failed parse")
    }
}
impl From<()> for ParseError {
    fn from(_: ()) -> ParseError {
        ParseError
    }
}
impl From<io::Error> for ParseError {
    fn from(_: io::Error) -> ParseError {
        ParseError
    }
}
impl<I, E> From<winnow::error::ParseError<I, E>> for ParseError {
    fn from(_: winnow::error::ParseError<I, E>) -> ParseError {
        ParseError
    }
}
impl ::std::error::Error for ParseError {
    fn description(&self) -> &'static str {
        "parse error"
    }
}

/// Parse a [`Lexicon`], including its `background` knowledge, and any `templates`
/// that might be used during learning, ensuring that `background` and
/// `templates` typecheck according to the types in the newly instantiated
/// [`Lexicon`]. Syntax errors and failed typechecking both lead to `Err`.
///
/// # Lexicon syntax
///
/// `input` is parsed as a `lexicon`, defined below in [augmented Backus-Naur
/// form]. The definition of `scheme` is as given in [`polytype`], while other
/// terms are as given in [`term_rewriting`]:
///
/// ```text
/// lexicon = *wsp *( *comment declaration ";" *comment ) *wsp
///
/// declaration = *wsp identifier *wsp ":" *wsp scheme *wsp
/// ```
///
/// # Background syntax
///
/// `background` is parsed as a `rule_list`, defined below in [augmented Backus-Naur form].
/// The format of the other terms is as given in [`term_rewriting`]:
///
/// ```text
/// rule_list = *wsp *( *comment rule ";" *comment ) *wsp
/// ```
///
/// # Template Syntax
///
/// `templates` is parsed as a `rulecontext_list`, defined below in [augmented Backus-Naur form].
/// The format of the other terms is as given in [`term_rewriting`]:
///
/// ```text
/// rulecontext_list = *wsp *( *comment rulecontext ";" *comment ) *wsp
/// ```
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
/// [`polytype`]: ../../../polytype/index.html
/// [augmented Backus-Naur form]: https://en.wikipedia.org/wiki/Augmented_Backus–Naur_form
pub fn parse_lexicon(
    lexicon: &str,
    background: &str,
    templates: &str,
    deterministic: bool,
    ctx: TypeContext,
) -> Result<Lexicon, ParseError> {
    _parse_lexicon(lexicon, background, templates, deterministic, ctx).map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a [`TRS`]. The format of the
/// [`TRS`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`TRS`]: struct.TRS.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_trs(mut input: &str, lex: &mut Lexicon) -> Result<TRS, ParseError> {
    let mut ctx = lex.0.read().unwrap().ctx.clone();
    _parse_trs(&mut input, lex, &mut ctx).map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a [`Context`]. The format of the
/// [`Context`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`Context`]: ../../../term_rewriting/enum.Context.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_context(
    input: &str,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> Result<Context, ParseError> {
    parse_typed_context(input, lex, ctx).map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a [`RuleContext`]. The format of the
/// [`RuleContext`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`RuleContext`]: ../../../term_rewriting/struct.RuleContext.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_rulecontext(
    mut input: &str,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> Result<RuleContext, ParseError> {
    parse_typed_rulecontext(&mut input, lex, ctx).map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a [`Rule`]. The format of the
/// [`Rule`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`Rule`]: ../../../term_rewriting/struct.Rule.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_rule(
    mut input: &str,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> Result<Rule, ParseError> {
    parse_typed_rule(&mut input, lex, ctx).map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a list of [`RuleContext`s] (e.g.
/// `templates` in [`parse_lexicon`]). The format of a [`RuleContext`] is as
/// given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`RuleContext`s]: ../../../term_rewriting/struct.RuleContext.html
/// [`RuleContext`]: ../../../term_rewriting/struct.RuleContext.html
/// [`parse_lexicon`]: fn.parse_lexicon.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_templates(mut input: &str, lex: &mut Lexicon) -> Result<Vec<RuleContext>, ParseError> {
    let mut ctx = lex.0.write().unwrap().ctx.clone();
    _parse_templates(&mut input, lex, &mut ctx).map_err(|_| ParseError)
}

#[derive(Debug, Clone, PartialEq)]
enum AtomName {
    Variable(String),
    Operator(String),
}

fn make_atom(
    name: AtomName,
    sig: &mut Signature,
    scheme: TypeScheme,
    vars: &mut Vec<TypeScheme>,
    ops: &mut Vec<TypeScheme>,
) -> Atom {
    match name {
        AtomName::Variable(s) => {
            let v = sig.new_var(Some(s));
            vars.push(scheme);
            Atom::Variable(v)
        }
        AtomName::Operator(s) => {
            let arity = scheme
                .instantiate(&mut TypeContext::default())
                .args()
                .map_or(0, |args| args.len());
            let o = sig.new_op(arity as u32, Some(s));
            ops.push(scheme);
            Atom::Operator(o)
        }
    }
}

// reserved characters include:
// - # for comments
// - _ for variables
// - : for signatures
// - ( and ) for grouping
// - = for specifying rules
// - ; for ending statements
fn parse_id<'a>(input: &mut &'a str) -> PResult<&'a str> {
    take_while(0.., |c: char| !"#_:()=;".contains(c)).parse_next(input)
}
fn parse_atom_name(input: &mut &str) -> PResult<AtomName> {
    alt((
        terminated(parse_id, "_").map(|s| AtomName::Variable(s.to_owned())),
        parse_id.map(|s| AtomName::Operator(s.to_owned())),
    ))
    .parse_next(input)
}
fn parse_comment<'a>(input: &mut &'a str) -> PResult<&'a str> {
    preceded("#", terminated(not_line_ending, newline)).parse_next(input)
}
fn parse_irrelevant(input: &mut &str) -> PResult<()> {
    multispace0.parse_next(input)?;
    repeat(0.., (parse_comment, multispace0)).parse_next(input)?;
    Ok(())
}
fn parse_declaration(input: &mut &str) -> PResult<(AtomName, TypeScheme)> {
    delimited(
        multispace0,
        separated_pair(
            parse_atom_name,
            (multispace0, ':', multispace0),
            terminated(take_till(0.., ';'), ';').parse_to(),
        ),
        multispace0,
    )
    .parse_next(input)
}
fn parse_simple_lexicon(
    input: &mut &str,
    deterministic: bool,
    ctx: TypeContext,
) -> PResult<Lexicon> {
    let atom_infos: Vec<(AtomName, TypeScheme)> = repeat(
        0..,
        delimited(parse_irrelevant, parse_declaration, parse_irrelevant),
    )
    .parse_next(input)?;

    let mut sig = Signature::default();
    let mut vars: Vec<TypeScheme> = vec![];
    let mut ops: Vec<TypeScheme> = vec![];
    for (atom_name, scheme) in atom_infos {
        make_atom(atom_name, &mut sig, scheme, &mut vars, &mut ops);
    }
    Ok(Lexicon::from_signature(
        sig,
        ops,
        vars,
        vec![],
        vec![],
        deterministic,
        ctx,
    ))
}
pub fn _parse_lexicon(
    mut lexicon: &str,
    mut background: &str,
    mut templates: &str,
    deterministic: bool,
    ctx: TypeContext,
) -> PResult<Lexicon> {
    let mut lex = parse_simple_lexicon(&mut lexicon, deterministic, ctx)?;
    let mut ctx = lex.0.read().unwrap().ctx.clone();
    let trs = _parse_trs(&mut background, &mut lex, &mut ctx)?;
    lex.0.write().unwrap().background = trs.utrs.rules;
    let templates = _parse_templates(&mut templates, &mut lex, &mut ctx)?;
    lex.0.write().unwrap().templates = templates;
    lex.0.write().unwrap().ctx = ctx;
    Ok(lex)
}

fn parse_typed_rule(input: &mut &str, lex: &mut Lexicon, ctx: &mut TypeContext) -> PResult<Rule> {
    let text = terminated(take_till(0.., ';'), opt(';')).parse_next(input)?;
    let rule = parse_untyped_rule(&mut lex.0.write().unwrap().signature, text)
        .map_err(|_| ErrMode::Backtrack(Default::default()))?;

    add_parsed_variables_to_lexicon(lex, ctx);
    if lex.infer_rule(&rule, ctx).is_ok() {
        Ok(rule)
    } else {
        Err(ErrMode::Cut(Default::default()))
    }
}

fn _parse_trs(input: &mut &str, lex: &mut Lexicon, ctx: &mut TypeContext) -> PResult<TRS> {
    multispace0.parse_next(input)?;
    let parse_typed_rule =
        |input: &mut &str| -> PResult<Rule> { parse_typed_rule(input, lex, ctx) };
    let rules = repeat(
        0..,
        delimited(parse_irrelevant, parse_typed_rule, parse_irrelevant),
    )
    .parse_next(input)?;
    TRS::new(lex, rules, ctx).map_err(|_| ErrMode::Cut(Default::default()))
}

fn parse_typed_context(input: &str, lex: &mut Lexicon, ctx: &mut TypeContext) -> PResult<Context> {
    let context = parse_untyped_context(&mut lex.0.write().unwrap().signature, input)
        .map_err(|_| ErrMode::Backtrack(Default::default()))?;
    add_parsed_variables_to_lexicon(lex, ctx);
    if lex.infer_context(&context, ctx).is_ok() {
        Ok(context)
    } else {
        Err(ErrMode::Cut(Default::default()))
    }
}
fn parse_typed_rulecontext(
    input: &mut &str,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> PResult<RuleContext> {
    let text = terminated(take_till(0.., ';'), opt(';')).parse_next(input)?;
    let rulecontext = parse_untyped_rulecontext(&mut lex.0.write().unwrap().signature, text)
        .map_err(|_| ErrMode::Backtrack(Default::default()))?;

    add_parsed_variables_to_lexicon(lex, ctx);
    if lex.infer_rulecontext(&rulecontext, ctx).is_ok() {
        Ok(rulecontext)
    } else {
        Err(ErrMode::Cut(Default::default()))
    }
}
fn _parse_templates(
    input: &mut &str,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> PResult<Vec<RuleContext>> {
    let parse_typed_rulecontext =
        |input: &mut &str| -> PResult<RuleContext> { parse_typed_rulecontext(input, lex, ctx) };
    repeat(
        0..,
        delimited(parse_irrelevant, parse_typed_rulecontext, parse_irrelevant),
    )
    .parse_next(input)
}

fn add_parsed_variables_to_lexicon(lex: &Lexicon, ctx: &mut TypeContext) {
    let n_vars = lex.0.read().unwrap().signature.variables().len();
    let n_schemes = lex.0.read().unwrap().vars.len();
    let diff = n_vars - n_schemes;
    for _ in 0..diff {
        let scheme = TypeScheme::Monotype(ctx.new_variable());
        lex.0.write().unwrap().vars.push(scheme);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comment_test() {
        let res = parse_comment(&mut "# this is a test\n");
        assert_eq!(res.unwrap(), " this is a test");
    }

    #[test]
    fn declaration_op_test() {
        let (a, s) = parse_declaration(&mut "SUCC: int -> int;").unwrap();
        assert_eq!(a, AtomName::Operator("SUCC".to_owned()));
        assert_eq!(s.to_string(), "int → int");
    }

    #[test]
    fn declaration_var_test() {
        let (a, s) = parse_declaration(&mut "x_: int;").unwrap();
        assert_eq!(a, AtomName::Variable("x".to_owned()));
        assert_eq!(s.to_string(), "int");
    }

    #[test]
    fn lexicon_test() {
        let res = parse_lexicon(
            "# COMMENT\nSUCC: int -> int;\nx_: list(int);",
            "",
            "",
            false,
            TypeContext::default(),
        );
        assert!(res.is_ok());
        assert_eq!(res.unwrap().to_string(), "Signature:\nSUCC: int → int\nx_: list(int)\n\nBackground: 0\n\nTemplates: 0\n\nDeterministic: false\n");
    }

    #[test]
    fn typed_rule_test() {
        let mut lex = parse_lexicon(
            "ZERO: int; SUCC: int -> int;",
            "",
            "",
            false,
            TypeContext::default(),
        )
        .unwrap();
        let mut ctx = lex.0.read().unwrap().ctx.clone();
        let res = parse_typed_rule(&mut "SUCC(x_) = ZERO", &mut lex, &mut ctx);

        assert_eq!(res.unwrap().display(), "SUCC(x_) = ZERO");
    }

    #[test]
    fn trs_test() {
        let mut lex = parse_lexicon(
            "ZERO: int; SUCC: int -> int; PLUS: int -> int -> int;",
            "",
            "",
            false,
            TypeContext::default(),
        )
        .unwrap();
        let mut ctx = lex.0.read().unwrap().ctx.clone();
        let res = _parse_trs(
            &mut "PLUS(ZERO x_) = ZERO; PLUS(SUCC(x_) y_) = SUCC(PLUS(x_ y_));",
            &mut lex,
            &mut ctx,
        );

        assert_eq!(
            res.unwrap().utrs.display(),
            "PLUS(ZERO x_) = ZERO;\nPLUS(SUCC(x_) y_) = SUCC(PLUS(x_ y_));"
        );
    }

    #[test]
    fn context_test() {
        let mut lex = parse_lexicon(
            "ZERO: int; SUCC: int -> int; PLUS: int -> int -> int;",
            "",
            "",
            false,
            TypeContext::default(),
        )
        .unwrap();
        let mut ctx = lex.0.read().unwrap().ctx.clone();
        let res = parse_typed_context("PLUS(x_ [!])", &mut lex, &mut ctx);

        assert_eq!(res.unwrap().display(), "PLUS(x_ [!])");
    }

    #[test]
    fn rulecontext_test() {
        let mut lex = parse_lexicon(
            "ZERO: int; SUCC: int -> int; PLUS: int -> int -> int;",
            "",
            "",
            false,
            TypeContext::default(),
        )
        .unwrap();
        let mut ctx = lex.0.read().unwrap().ctx.clone();
        let res = parse_typed_rulecontext(&mut "PLUS(x_ [!]) = ZERO", &mut lex, &mut ctx);

        assert_eq!(res.unwrap().display(), "PLUS(x_ [!]) = ZERO");
    }

    #[test]
    fn templates_test() {
        let mut lex = parse_lexicon(
            "ZERO: int; SUCC: int -> int; PLUS: int -> int -> int;",
            "",
            "",
            false,
            TypeContext::default(),
        )
        .unwrap();
        let mut ctx = lex.0.read().unwrap().ctx.clone();
        let res = _parse_templates(
            &mut "PLUS(x_ [!]) = ZERO; [!] = SUCC(ZERO);",
            &mut lex,
            &mut ctx,
        );

        let res_string = res
            .unwrap()
            .iter()
            .map(|rc| format!("{};", rc.display()))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(res_string, "PLUS(x_ [!]) = ZERO;\n[!] = SUCC(ZERO);");
    }
}
