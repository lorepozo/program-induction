use std::rc::Rc;
use nom::{digit, multispace};
use nom::types::CompleteStr;

use super::LispError;
use super::interp;

#[derive(Debug, Clone, PartialEq)]
pub enum Sexp {
    Bool(bool),
    Integer(i32),
    Real(f64),
    Char(char),
    Str(String),
    Ident(String),
    Pair(Box<Sexp>, Box<Sexp>),
    Null,
}
impl Sexp {
    pub fn eval(&self) -> Result<Rc<interp::Value>, LispError> {
        interp::eval(self)
    }
}

/// doesn't match parentheses or comma, but matches many printable characters.
fn alphanumeric_ext(c: char) -> bool {
    ((c >= 0x21 as char && c <= 0x7E as char) && !(c == '(' || c == ')')) || (c == 'λ')
}

named!(atom_null<CompleteStr, Sexp>, do_parse!(tag!("'()")  >> (Sexp::Null)));
named!(atom_str<CompleteStr, Sexp>,
    do_parse!(
        content: delimited!(
            tag!("\""),
            take_until_s!("\""),
            tag!("\"")
        ) >>
        (Sexp::Str(content.0.to_string()))));
named!(atom_bool<CompleteStr, Sexp>, alt!(atom_bool_true | atom_bool_false));
named!(atom_bool_true<CompleteStr, Sexp>,  do_parse!(tag!("#t") >> (Sexp::Bool(true))));
named!(atom_bool_false<CompleteStr, Sexp>, do_parse!(tag!("#f") >> (Sexp::Bool(false))));
named!(atom_int<CompleteStr, Sexp>, alt!(atom_int_neg | atom_int_pos));
named!(atom_int_pos<CompleteStr, Sexp>,
    do_parse!(
        opt!(tag!("+")) >>
        num_str: digit >>
        (Sexp::Integer(num_str.0.parse().unwrap()))));
named!(atom_int_neg<CompleteStr, Sexp>,
    do_parse!(
        tag!("-") >>
        num_str: digit >>
        (Sexp::Integer(-num_str.0.parse::<i32>().unwrap()))));
named!(atom_real<CompleteStr, Sexp>, alt!(atom_real_neg | atom_real_pos));
named!(atom_real_pos<CompleteStr, Sexp>,
    do_parse!(
        opt!(tag!("+")) >>
        x: recognize!( delimited!(digit, tag!("."), digit) ) >>
        (Sexp::Real(x.0.parse().unwrap()))));
named!(atom_real_neg<CompleteStr, Sexp>,
    do_parse!(
        tag!("-") >>
        x: recognize!( delimited!(digit, tag!("."), digit) ) >>
        (Sexp::Real(-x.0.parse::<f64>().unwrap()))));
named!(atom_char<CompleteStr, Sexp>,
    do_parse!(
        tag!("#\\")         >>
        content: take_s!(1) >>
        (Sexp::Char(content.0.chars().nth(0).unwrap()))));
named!(atom_ident<CompleteStr, Sexp>,
    do_parse!(
        ident: take_while1!(alphanumeric_ext) >>
        (Sexp::Ident(ident.0.to_string()))));
named!(atom<CompleteStr, Sexp>,
    alt!(atom_null |
         atom_str  |
         atom_bool |
         atom_real |
         atom_int  |
         atom_char |
         atom_ident));
named!(pair<CompleteStr, Sexp>,
    do_parse!(
        xs: delimited!(
            tag!("("),
            separated_nonempty_list!(multispace, expr),
            tag!(")")
        ) >>
        (pair_of_list(xs))));
named!(expr<CompleteStr, Sexp>, alt!(pair | atom));

fn pair_of_list(mut xs: Vec<Sexp>) -> Sexp {
    xs.reverse();
    pair_of_list_internal(xs)
}
fn pair_of_list_internal(mut xs: Vec<Sexp>) -> Sexp {
    match xs.len() {
        0 => Sexp::Null,
        1 => Sexp::Pair(Box::new(xs.pop().unwrap()), Box::new(Sexp::Null)),
        2 => {
            let car = Box::new(xs.pop().unwrap());
            let cadr = Box::new(xs.pop().unwrap());
            Sexp::Pair(car, Box::new(Sexp::Pair(cadr, Box::new(Sexp::Null))))
        }
        _ => {
            let car = Box::new(xs.pop().unwrap());
            Sexp::Pair(car, Box::new(pair_of_list_internal(xs)))
        }
    }
}

pub fn parse(inp: &str) -> Result<Sexp, LispError> {
    match expr(CompleteStr(inp)) {
        Ok((_, sexp)) => Ok(sexp),
        Err(e) => Err(LispError::ParseError(e.into_error_kind())),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse, Sexp};

    macro_rules! check_parse {
        ($title: ident, $inp: expr, $pat: pat) => {
            check_parse!($title, $inp, $pat => ());
        };
        ($title: ident, $inp: expr, $pat: pat => $arm: expr) => {
            #[test]
            fn $title() {
                match parse($inp).expect(&format!("sexp {}", stringify!($title))) {
                    $pat => $arm,
                    ref e => panic!(
                        "assertion failed: `{:?}` does not match `{}`",
                        e,
                        stringify!($pat)
                    ),
                }
            }
        };
    }

    check_parse!(parse_bool_1, "#t", Sexp::Bool(true));
    check_parse!(parse_bool_2, "#f", Sexp::Bool(false));
    check_parse!(parse_int_1, "42", Sexp::Integer(42));
    check_parse!(parse_int_2, "-42", Sexp::Integer(-42));
    check_parse!(parse_real_1, "42.42", Sexp::Real(f) => assert_eq!(f, 42.42));
    check_parse!(parse_real_2, "-42.42", Sexp::Real(f) => assert_eq!(f, -42.42));
    check_parse!(parse_char_1, "#\\,", Sexp::Char(','));
    check_parse!(parse_char_2, "#\\ ", Sexp::Char(' '));
    check_parse!(parse_str, "\"some string\"", Sexp::Str(s) => assert_eq!(s, "some string"));
    check_parse!(parse_ident_1, "ident", Sexp::Ident(s) => assert_eq!(s, "ident"));
    check_parse!(parse_ident_2, "λ", Sexp::Ident(s) => assert_eq!(s, "λ"));
    check_parse!(parse_null, "'()", Sexp::Null);

    check_parse!(parse_pair_1, "(foo)", Sexp::Pair(car, cdr) => {
        match *car {
            Sexp::Ident(s) => assert_eq!(s, "foo"),
            ref e => panic!("assertion failed: `{:?}` does not match Sexp::Ident(_)", e),
        }
        match *cdr {
            Sexp::Null => (),
            ref e => panic!("assertion failed: `{:?}` does not match Sexp::Null", e),
        }
    });
    check_parse!(parse_pair_2, "(foo \"bar\")", Sexp::Pair(car, cdr) => {
        match *car {
            Sexp::Ident(s) => assert_eq!(s, "foo"),
            ref e => panic!("assertion failed: `{:?}` does not match Sexp::Ident(_)", e),
        }
        match *cdr {
            Sexp::Pair(ref cadr, ref cddr) => {
                match **cadr {
                    Sexp::Str(ref s) => assert_eq!(s, "bar"),
                    ref e => panic!("assertion failed: `{:?}` does not match Sexp::Str(_)", e),
                }
                match **cddr {
                    Sexp::Null => (),
                    ref e => panic!("assertion failed: `{:?}` does not match Sexp::Null", e),
                }
            }
            ref e => panic!("assertion failed: `{:?}` does not match Sexp::Pair(_, _)", e),
        }
    });
    check_parse!(parse_pair_3, "(1 2.2 3)", Sexp::Pair(car, cdr) => {
        match *car {
            Sexp::Integer(1) => (),
            ref e => panic!("assertion failed: `{:?}` does not match Sexp::Integer(1)", e),
        }
        match *cdr {
            Sexp::Pair(ref cadr, ref cddr) => {
                match **cadr {
                    Sexp::Real(f) => assert_eq!(f, 2.2),
                    ref e => panic!("assertion failed: `{:?}` does not match Sexp::Real(2.2)", e),
                }
                match **cddr {
                    Sexp::Pair(ref caddr, ref cdddr) => {
                        match **caddr {
                            Sexp::Integer(3) => (),
                            ref e => panic!("assertion failed: `{:?}` does not match Sexp::Integer(3)", e),
                        }
                        match **cdddr {
                            Sexp::Null => (),
                            ref e => panic!("assertion failed: `{:?}` does not match Sexp::Null", e),
                        }
                    }
                    ref e => panic!("assertion failed: `{:?}` does not match Sexp::Pair(_, _)", e),
                }
            }
            ref e => panic!("assertion failed: `{:?}` does not match Sexp::Pair(_, _)", e),
        }
    });
}
