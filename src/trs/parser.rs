use super::lexicon::Lexicon;
use super::rewrite::TRS;
use nom;
use nom::types::CompleteStr;
use nom::{Context as Nomtext, Err};
use polytype::{Context as TypeContext, TypeSchema};
use std::fmt;
use std::io;
use term_rewriting::{
    parse_context as parse_untyped_context, parse_rule as parse_untyped_rule,
    parse_rulecontext as parse_untyped_rulecontext, Atom, Context, Rule, RuleContext, Signature,
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
impl<'a> From<io::Error> for ParseError {
    fn from(_: io::Error) -> ParseError {
        ParseError
    }
}
impl<'a> From<Err<&'a str>> for ParseError {
    fn from(_: Err<&'a str>) -> ParseError {
        ParseError
    }
}
impl ::std::error::Error for ParseError {
    fn description(&self) -> &'static str {
        "parse error"
    }
}

/// Parse a `Lexicon`.
pub fn parse_lexicon(
    input: &str,
    background: &str,
    templates: &str,
    deterministic: bool,
    ctx: TypeContext,
) -> Result<Lexicon, ParseError> {
    if let Ok((CompleteStr(""), l)) = lexicon(
        CompleteStr(input),
        CompleteStr(background),
        CompleteStr(templates),
        Signature::default(),
        vec![],
        vec![],
        deterministic,
        ctx,
    ) {
        Ok(l)
    } else {
        Err(ParseError)
    }
}

/// Parse a `TRS`.
pub fn parse_trs(input: &str, lex: &mut Lexicon) -> Result<TRS, ParseError> {
    let mut ctx = lex.0.read().expect("poisoned lexicon").ctx.clone();
    if let Ok((CompleteStr(""), t)) = trs(CompleteStr(input), lex, &mut ctx) {
        Ok(t)
    } else {
        Err(ParseError)
    }
}

/// Parse a `Context`.
pub fn parse_context(
    input: &str,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> Result<Context, ParseError> {
    if let Ok((CompleteStr(""), t)) = typed_context(CompleteStr(input), lex, ctx) {
        Ok(t)
    } else {
        Err(ParseError)
    }
}

/// Parse a `RuleContext`.
pub fn parse_rulecontext(
    input: &str,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> Result<RuleContext, ParseError> {
    if let Ok((CompleteStr(""), t)) = typed_rulecontext(CompleteStr(input), lex, ctx) {
        Ok(t)
    } else {
        Err(ParseError)
    }
}

/// Parse a `Rule`.
pub fn parse_rule(
    input: &str,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> Result<Rule, ParseError> {
    if let Ok((CompleteStr(""), t)) = typed_rule(input, lex, ctx) {
        Ok(t)
    } else {
        Err(ParseError)
    }
}

/// Parse a list of `RuleContext`s.
pub fn parse_templates(input: &str, lex: &mut Lexicon) -> Result<Vec<RuleContext>, ParseError> {
    let mut ctx = lex.0.write().expect("poisoned lexicon").ctx.clone();
    if let Ok((CompleteStr(""), t)) = templates(CompleteStr(input), lex, &mut ctx) {
        Ok(t)
    } else {
        Err(ParseError)
    }
}

#[derive(Debug, Clone)]
enum AtomName {
    Variable(String),
    Operator(String),
}

fn schema_wrapper(input: CompleteStr) -> nom::IResult<CompleteStr, TypeSchema> {
    if let Ok(schema) = TypeSchema::parse(*input) {
        Ok((CompleteStr(""), schema))
    } else {
        Err(Err::Error(Nomtext::Code(input, nom::ErrorKind::Custom(0))))
    }
}

fn make_atom(
    name: AtomName,
    sig: &mut Signature,
    schema: TypeSchema,
    vars: &mut Vec<TypeSchema>,
    ops: &mut Vec<TypeSchema>,
) -> Atom {
    match name {
        AtomName::Variable(s) => {
            let v = sig.new_var(Some(s.to_string()));
            vars.push(schema);
            Atom::Variable(v)
        }
        AtomName::Operator(s) => {
            let arity = schema
                .instantiate(&mut TypeContext::default())
                .args()
                .map_or(0, |args| args.len());
            let o = sig.new_op(arity as u32, Some(s.to_string()));
            ops.push(schema);
            Atom::Operator(o)
        }
    }
}

named!(colon<CompleteStr, CompleteStr>, tag!(":"));
// reserved characters include:
// - # for comments
// - _ for variables
// - : for signatures
// - ( and ) for grouping
// - = for specifying rules
// - ; for ending statements
named!(identifier<CompleteStr, CompleteStr>, is_not!("#_:()=;"));
named!(underscore<CompleteStr, CompleteStr>, tag!("_"));
named!(atom_name<CompleteStr, AtomName>,
       alt!(map!(terminated!(identifier, underscore),
                 |s| AtomName::Variable(s.to_string())) |
            map!(identifier,
                 |s| AtomName::Operator(s.to_string()))));
named!(schema<CompleteStr, TypeSchema>,
       call!(schema_wrapper));
named!(comment<CompleteStr, CompleteStr>,
       map!(preceded!(tag!("#"), take_until_and_consume!("\n")),
            |s| CompleteStr(&s.trim())));
named_args!(declaration<'a>(sig: &mut Signature, vars: &mut Vec<TypeSchema>, ops: &mut Vec<TypeSchema>) <CompleteStr<'a>, (Atom, TypeSchema)>,
       map!(ws!(do_parse!(name: atom_name >>
                      colon >>
                      schema: schema >>
                      (name, schema))),
            |(n, s)| {
                (make_atom(n, sig, s.clone(), vars, ops), s)}
       ));
fn simple_lexicon<'a>(
    input: CompleteStr<'a>,
    mut sig: Signature,
    mut vars: Vec<TypeSchema>,
    mut ops: Vec<TypeSchema>,
    deterministic: bool,
    ctx: TypeContext,
) -> nom::IResult<CompleteStr<'a>, Lexicon> {
    map!(
        input,
        ws!(many0!(do_parse!(
            many0!(ws!(comment))
                >> dec: take_until_and_consume!(";")
                >> expr_res!(declaration(dec, &mut sig, &mut vars, &mut ops))
                >> many0!(ws!(comment))
                >> ()
        ))),
        |_| Lexicon::from_signature(sig, ops, vars, vec![], vec![], deterministic, ctx)
    )
}
#[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
fn lexicon<'a>(
    input: CompleteStr<'a>,
    bg: CompleteStr<'a>,
    temp: CompleteStr<'a>,
    sig: Signature,
    vars: Vec<TypeSchema>,
    ops: Vec<TypeSchema>,
    deterministic: bool,
    ctx: TypeContext,
) -> nom::IResult<CompleteStr<'a>, Lexicon> {
    let (remaining, mut lex) = simple_lexicon(input, sig, vars, ops, deterministic, ctx)?;
    let mut ctx = lex.0.read().expect("poisoned lexicon").ctx.clone();
    match trs(bg, &mut lex, &mut ctx)? {
        (CompleteStr(""), trs) => {
            lex.0.write().expect("poisoned lexicon").background = trs.utrs.rules;
        }
        _ => return Err(Err::Error(Nomtext::Code(bg, nom::ErrorKind::Custom(0)))),
    }
    match templates(temp, &mut lex, &mut ctx)? {
        (CompleteStr(""), templates) => {
            lex.0.write().expect("poisoned lexicon").templates = templates;
            lex.0.write().expect("poisoned lexicon").ctx = ctx;
            Ok((remaining, lex))
        }
        _ => Err(Err::Error(Nomtext::Code(bg, nom::ErrorKind::Custom(0)))),
    }
}
fn typed_rule<'a>(
    input: &'a str,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> nom::IResult<CompleteStr<'a>, Rule> {
    let result = parse_untyped_rule(
        &mut lex.0.write().expect("poisoned lexicon").signature,
        input,
    );
    if let Ok(rule) = result {
        add_parsed_variables_to_lexicon(lex, ctx);
        if lex.infer_rule(&rule, ctx).is_ok() {
            return Ok((CompleteStr(""), rule));
        }
    }
    Err(Err::Error(Nomtext::Code(
        CompleteStr(input),
        nom::ErrorKind::Custom(0),
    )))
}
named_args!(trs<'a>(lex: &mut Lexicon, ctx: &mut TypeContext) <CompleteStr<'a>, TRS>,
    ws!(do_parse!(rules: many0!(do_parse!(many0!(ws!(comment)) >>
                                          rule_text: take_until_and_consume!(";") >>
                                          rule: expr_res!(typed_rule(&rule_text, lex, ctx)) >>
                                          many0!(ws!(comment)) >>
                                          (rule.1))) >>
                  trs: expr_res!(TRS::new(lex, rules, ctx)) >>
                  (trs)))
);
fn add_parsed_variables_to_lexicon(lex: &Lexicon, ctx: &mut TypeContext) {
    let n_vars = lex
        .0
        .read()
        .expect("poisoned lexicon")
        .signature
        .variables()
        .len();
    let n_schemas = lex.0.read().expect("poisoned lexicon").vars.len();
    let diff = n_vars - n_schemas;
    for _ in 0..diff {
        let schema = TypeSchema::Monotype(ctx.new_variable());
        lex.0.write().expect("poisoned lexicon").vars.push(schema);
    }
}
fn typed_context<'a>(
    input: CompleteStr<'a>,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> nom::IResult<CompleteStr<'a>, Context> {
    let result = parse_untyped_context(
        &mut lex.0.write().expect("poisoned lexicon").signature,
        *input,
    );
    if let Ok(rule) = result {
        add_parsed_variables_to_lexicon(lex, ctx);
        if lex.infer_context(&rule, ctx).is_ok() {
            return Ok((CompleteStr(""), rule));
        }
    }
    Err(Err::Error(Nomtext::Code(input, nom::ErrorKind::Custom(0))))
}
fn typed_rulecontext<'a>(
    input: CompleteStr<'a>,
    lex: &mut Lexicon,
    ctx: &mut TypeContext,
) -> nom::IResult<CompleteStr<'a>, RuleContext> {
    let result = parse_untyped_rulecontext(
        &mut lex.0.write().expect("poisoned lexicon").signature,
        *input,
    );
    if let Ok(rule) = result {
        add_parsed_variables_to_lexicon(lex, ctx);
        if lex.infer_rulecontext(&rule, ctx).is_ok() {
            return Ok((CompleteStr(""), rule));
        }
    }
    Err(Err::Error(Nomtext::Code(input, nom::ErrorKind::Custom(0))))
}
named_args!(templates<'a>(lex: &mut Lexicon, ctx: &mut TypeContext) <CompleteStr<'a>, Vec<RuleContext>>,
            ws!(do_parse!(templates: many0!(do_parse!(many0!(ws!(comment)) >>
                                                      rc_text: take_until_and_consume!(";") >>
                                                      rc: expr_res!(typed_rulecontext(rc_text, lex, ctx)) >>
                                                      many0!(ws!(comment)) >>
                                                      (rc.1))) >>
                          (templates)))
);

#[cfg(test)]
mod tests {
    use super::*;
    use term_rewriting::Signature;

    #[test]
    fn comment_test() {
        let res = comment(CompleteStr("# this is a test\n"));
        assert_eq!(res.unwrap().1, CompleteStr("this is a test"));
    }

    #[test]
    fn declaration_op_test() {
        let mut vars = vec![];
        let mut ops = vec![];
        let mut sig = Signature::default();
        let (_, (a, s)) = declaration(
            CompleteStr("SUCC: int -> int"),
            &mut sig,
            &mut vars,
            &mut ops,
        )
        .unwrap();
        assert_eq!(a.display(), "SUCC");
        assert_eq!(s.to_string(), "int → int");
    }

    #[test]
    fn declaration_var_test() {
        let mut vars = vec![];
        let mut ops = vec![];
        let mut sig = Signature::default();
        let (_, (a, s)) =
            declaration(CompleteStr("x_: int"), &mut sig, &mut vars, &mut ops).unwrap();
        assert_eq!(a.display(), "x_");
        assert_eq!(s.to_string(), "int");
    }

    #[test]
    fn lexicon_test() {
        let res = lexicon(
            CompleteStr("# COMMENT\nSUCC: int -> int;\nx_: list(int);"),
            CompleteStr(""),
            CompleteStr(""),
            Signature::default(),
            vec![],
            vec![],
            false,
            TypeContext::default(),
        );
        assert!(res.is_ok());
    }

    #[test]
    fn typed_rule_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO: int; SUCC: int -> int;"),
            CompleteStr(""),
            CompleteStr(""),
            Signature::default(),
            vec![],
            vec![],
            false,
            TypeContext::default(),
        )
        .unwrap()
        .1;
        let mut ctx = lex.0.read().expect("poisoned lexicon").ctx.clone();
        let res = typed_rule("SUCC(x_) = ZERO", &mut lex, &mut ctx);

        assert_eq!(res.unwrap().1.display(), "SUCC(x_) = ZERO");
    }

    #[test]
    fn trs_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO: int; SUCC: int -> int; PLUS: int -> int -> int;"),
            CompleteStr(""),
            CompleteStr(""),
            Signature::default(),
            vec![],
            vec![],
            false,
            TypeContext::default(),
        )
        .unwrap()
        .1;
        let mut ctx = lex.0.read().expect("poisoned lexicon").ctx.clone();
        let res = trs(
            CompleteStr("PLUS(ZERO x_) = ZERO; PLUS(SUCC(x_) y_) = SUCC(PLUS(x_ y_));"),
            &mut lex,
            &mut ctx,
        );

        assert_eq!(
            res.unwrap().1.utrs.display(),
            "PLUS(ZERO x_) = ZERO;\nPLUS(SUCC(x_) y_) = SUCC(PLUS(x_ y_));"
        );
    }

    #[test]
    fn context_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO: int; SUCC: int -> int; PLUS: int -> int -> int;"),
            CompleteStr(""),
            CompleteStr(""),
            Signature::default(),
            vec![],
            vec![],
            false,
            TypeContext::default(),
        )
        .unwrap()
        .1;
        let mut ctx = lex.0.read().expect("poisoned lexicon").ctx.clone();
        let res = typed_context(CompleteStr("PLUS(x_ [!])"), &mut lex, &mut ctx);

        assert_eq!(res.unwrap().1.display(), "PLUS(x_ [!])");
    }

    #[test]
    fn rulecontext_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO: int; SUCC: int -> int; PLUS: int -> int -> int;"),
            CompleteStr(""),
            CompleteStr(""),
            Signature::default(),
            vec![],
            vec![],
            false,
            TypeContext::default(),
        )
        .unwrap()
        .1;
        let mut ctx = lex.0.read().expect("poisoned lexicon").ctx.clone();
        let res = typed_rulecontext(CompleteStr("PLUS(x_ [!]) = ZERO"), &mut lex, &mut ctx);

        assert_eq!(res.unwrap().1.display(), "PLUS(x_ [!]) = ZERO");
    }

    #[test]
    fn templates_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO: int; SUCC: int -> int; PLUS: int -> int -> int;"),
            CompleteStr(""),
            CompleteStr(""),
            Signature::default(),
            vec![],
            vec![],
            false,
            TypeContext::default(),
        )
        .unwrap()
        .1;
        let mut ctx = lex.0.read().expect("poisoned lexicon").ctx.clone();
        let res = templates(
            CompleteStr("PLUS(x_ [!]) = ZERO; [!] = SUCC(ZERO);"),
            &mut lex,
            &mut ctx,
        );

        let res_string = res
            .unwrap()
            .1
            .iter()
            .map(|rc| format!("{};", rc.display()))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(res_string, "PLUS(x_ [!]) = ZERO;\n[!] = SUCC(ZERO);");
    }
}
