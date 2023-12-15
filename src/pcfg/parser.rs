use polytype::Type;
use winnow::{
    ascii::multispace0,
    combinator::{alt, delimited, separated},
    prelude::*,
    token::take_while,
};

use super::{AppliedRule, Grammar, Rule};

#[derive(Clone, Debug)]
pub enum ParseError {
    InapplicableRule(Type, String),
    Other(String),
}
impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match *self {
            ParseError::InapplicableRule(ref nt, ref s) => {
                write!(f, "invalid rule {} for nonterminal {}", s, nt)
            }
            ParseError::Other(ref err) => write!(f, "could not parse: {}", err),
        }
    }
}
impl std::error::Error for ParseError {
    fn description(&self) -> &str {
        "could not parse expression"
    }
}

fn location<'a>(
    grammar: &'a Grammar,
    nt: &Type,
    name: &str,
) -> Result<(usize, &'a Rule), ParseError> {
    if let Some(rules) = grammar.rules.get(nt) {
        rules
            .iter()
            .enumerate()
            .find(|&(_, r)| r.name == name)
            .ok_or_else(|| ParseError::InapplicableRule(nt.clone(), String::from(name)))
    } else {
        Err(ParseError::InapplicableRule(nt.clone(), String::from(name)))
    }
}

#[derive(Debug)]
struct Item(String, Vec<Item>);
impl Item {
    fn into_applied(self, grammar: &Grammar, nt: Type) -> Result<AppliedRule, ParseError> {
        let (loc, r) = location(grammar, &nt, &self.0)?;
        if let Some(args) = r.production.args() {
            let inner: Result<Vec<AppliedRule>, ParseError> = self
                .1
                .into_iter()
                .zip(args)
                .map(move |(item, nt)| item.into_applied(grammar, nt.clone()))
                .collect();
            Ok(AppliedRule(nt, loc, inner?))
        } else {
            Ok(AppliedRule(nt, loc, vec![]))
        }
    }
}

fn alphanumeric_ext(c: char) -> bool {
    (c >= 0x21 as char && c <= 0x7E as char) && !(c == '(' || c == ')' || c == ',')
}

fn parse_item_name(input: &mut &str) -> PResult<String> {
    multispace0(input)?;
    let name = take_while(0.., alphanumeric_ext).parse_next(input)?;
    multispace0(input)?;
    Ok(name.to_owned())
}

fn parse_var(input: &mut &str) -> PResult<Item> {
    let name = parse_item_name.parse_next(input)?;
    Ok(Item(name, vec![]))
}

fn parse_func(input: &mut &str) -> PResult<Item> {
    let name = parse_item_name.parse_next(input)?;
    let args = delimited("(", separated(0.., parse_expr, ","), ")").parse_next(input)?;
    Ok(Item(name, args))
}

fn parse_expr(input: &mut &str) -> PResult<Item> {
    alt((parse_func, parse_var)).parse_next(input)
}

pub fn parse(grammar: &Grammar, input: &str, nt: Type) -> Result<AppliedRule, ParseError> {
    match parse_expr.parse(input) {
        Ok(item) => item.into_applied(grammar, nt),
        Err(err) => Err(ParseError::Other(err.to_string())),
    }
}
