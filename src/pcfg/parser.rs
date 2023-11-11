use nom::types::CompleteStr;
use nom::{alt, delimited, do_parse, named, separated_list, tag, take_while, ws};
use polytype::Type;
use std::{error, fmt};

use super::{AppliedRule, Grammar, Rule};

#[derive(Clone, Debug)]
pub enum ParseError {
    InapplicableRule(Type, String),
    NomError(String),
}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            ParseError::InapplicableRule(ref nt, ref s) => {
                write!(f, "invalide rule {} for nonterminal {}", s, nt)
            }
            ParseError::NomError(ref err) => write!(f, "could not parse: {}", err),
        }
    }
}
impl error::Error for ParseError {
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

/// doesn't match parentheses or comma, but matches most ascii printable characters.
fn alphanumeric_ext(c: char) -> bool {
    (c >= 0x21 as char && c <= 0x7E as char) && !(c == '(' || c == ')' || c == ',')
}

named!(var<CompleteStr, Item>,
do_parse!(
    name: ws!( take_while!(alphanumeric_ext) ) >>
    (Item(name.to_string(), vec![]))
));
named!(func<CompleteStr, Item>,
do_parse!(
    name: ws!( take_while!(alphanumeric_ext) ) >>
    args: delimited!(tag!("("), separated_list!(tag!(","), expr), tag!(")")) >>
    (Item(name.to_string(), args))
));
named!(expr<CompleteStr, Item>, alt!(func | var));

pub fn parse(grammar: &Grammar, input: &str, nt: Type) -> Result<AppliedRule, ParseError> {
    match expr(CompleteStr(input)) {
        Ok((_, item)) => item.into_applied(grammar, nt),
        Err(err) => Err(ParseError::NomError(format!("{:?}", err))),
    }
}
