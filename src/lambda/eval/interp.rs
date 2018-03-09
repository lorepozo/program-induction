use std::collections::VecDeque;
use std::fmt;

use super::LispError;
use super::sexp::Sexp;

#[derive(Clone)]
pub enum Value {
    Bool(bool),
    Integer(i32),
    Real(f64),
    Char(char),
    Str(String),
    // notably not included: Ident(String),
    Pair(Box<Value>, Box<Value>),
    Null,

    Lambda(Vec<String>, Sexp),
    BuiltinSpecial(fn(VecDeque<Sexp>, &mut Env) -> Result<Value, LispError>),
    Builtin(fn(VecDeque<Value>, &mut Env) -> Result<Value, LispError>),
}
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Value::Bool(b) => write!(f, "Bool({:?})", b),
            Value::Integer(i) => write!(f, "Integer({:?})", i),
            Value::Real(r) => write!(f, "Real({:?})", r),
            Value::Char(ref c) => write!(f, "Char({:?})", c),
            Value::Str(ref s) => write!(f, "Str({:?})", s),
            Value::Pair(ref l, ref r) => write!(f, "Pair({:?}, {:?})", l, r),
            Value::Null => write!(f, "Null"),
            Value::Lambda(ref params, ref body) => write!(f, "Lambda({:?}, {:?})", params, body),
            Value::Builtin(_) => write!(f, "Builtin(..)"),
            Value::BuiltinSpecial(_) => write!(f, "BuiltinSpecial(..)"),
        }
    }
}
impl From<Sexp> for Value {
    fn from(sexp: Sexp) -> Value {
        match sexp {
            Sexp::Bool(v) => Value::Bool(v),
            Sexp::Integer(v) => Value::Integer(v),
            Sexp::Real(v) => Value::Real(v),
            Sexp::Char(v) => Value::Char(v),
            Sexp::Str(v) => Value::Str(v),
            Sexp::Pair(l, r) => Value::Pair(Box::new(Value::from(*l)), Box::new(Value::from(*r))),
            Sexp::Null => Value::Null,
            Sexp::Ident(_) => panic!("ident is not a valid value"),
        }
    }
}

/// the environment is a list of assignments to identifiers.
pub struct Env(Vec<(String, Value)>);
impl Env {
    fn get(&self, s: &str) -> Option<Value> {
        self.0
            .iter()
            .rev()
            .find(|&&(ref l, _)| l == s)
            .map(|&(_, ref v)| v.clone())
    }
}

pub fn eval(sexp: &Sexp) -> Result<Value, LispError> {
    eval_sexp(sexp, &mut Env(Vec::new()))
}

fn eval_sexp(sexp: &Sexp, env: &mut Env) -> Result<Value, LispError> {
    match *sexp {
        Sexp::Pair(ref car, ref cdr) => {
            let mut xs = VecDeque::new();
            read_list(cdr, env, &mut xs);
            assert_eq!(
                xs.pop_back().unwrap(),
                Sexp::Null,
                "invalid list (was not null-terminated)"
            );
            let f = eval_sexp(car, env)?;
            eval_call(f, xs, env)
        }
        Sexp::Ident(ref s) => env.get(s)
            .or_else(|| get_builtin(s))
            .ok_or_else(|| LispError::IdentNotFound(s.clone())),
        _ => Ok(Value::from(sexp.clone())),
    }
}

fn eval_call(f: Value, xs: VecDeque<Sexp>, env: &mut Env) -> Result<Value, LispError> {
    match f {
        Value::BuiltinSpecial(f) => f(xs, env),
        _ => {
            let xs = xs.iter()
                .map(|s| eval_sexp(s, env))
                .collect::<Result<VecDeque<_>, LispError>>()?;
            eval_call_nonspecial(f, xs, env)
        }
    }
}

/// like `eval_call`, but takes already-evaluated expression
fn eval_call_nonspecial(f: Value, xs: VecDeque<Value>, env: &mut Env) -> Result<Value, LispError> {
    match f {
        Value::Lambda(params, body) => {
            let original_env_len = env.0.len();
            for (s, v) in params.into_iter().zip(xs) {
                env.0.push((s, v))
            }
            let res = eval_sexp(&body, env);
            env.0.truncate(original_env_len);
            res
        }
        Value::Builtin(f) => f(xs, env),
        _ => Err(LispError::InvalidCall),
    }
}

fn read_list(sexp: &Sexp, env: &mut Env, v: &mut VecDeque<Sexp>) {
    match *sexp {
        Sexp::Pair(ref car, ref cdr) => {
            v.push_back(*car.clone());
            read_list(cdr, env, v)
        }
        _ => v.push_back(sexp.clone()),
    }
}

fn get_builtin(s: &str) -> Option<Value> {
    match s {
        "null" => Some(Value::Null),

        "if" => Some(Value::BuiltinSpecial(builtin::if_)),
        "lambda" | "λ" => Some(Value::BuiltinSpecial(builtin::lambda)),
        "and" => Some(Value::BuiltinSpecial(builtin::and)),
        "or" => Some(Value::BuiltinSpecial(builtin::or)),

        "+" => Some(Value::Builtin(builtin::add)),
        "-" => Some(Value::Builtin(builtin::sub)),
        "not" => Some(Value::Builtin(builtin::not)),
        "*" => Some(Value::Builtin(builtin::mul)),
        "/" => Some(Value::Builtin(builtin::div)),
        "quotient" => Some(Value::Builtin(builtin::quotient)),
        "remainder" => Some(Value::Builtin(builtin::remainder)),
        "modulo" => Some(Value::Builtin(builtin::modulo)),
        "cons" => Some(Value::Builtin(builtin::cons)),
        "car" => Some(Value::Builtin(builtin::car)),
        "cdr" => Some(Value::Builtin(builtin::cdr)),
        "=" | "equal?" | "eqv?" | "eq?" => Some(Value::Builtin(builtin::eq)),
        ">" => Some(Value::Builtin(builtin::gt)),
        ">=" => Some(Value::Builtin(builtin::gte)),
        "<" => Some(Value::Builtin(builtin::lt)),
        "<=" => Some(Value::Builtin(builtin::lte)),
        "!=" => Some(Value::Builtin(builtin::neq)),
        "pair?" => Some(Value::Builtin(builtin::is_pair)),
        "null?" => Some(Value::Builtin(builtin::is_null)),
        "identity" => Some(Value::Builtin(builtin::identity)),
        "list" => Some(Value::Builtin(builtin::list)),
        "list-ref" => Some(Value::Builtin(builtin::list_ref)),
        "append" => Some(Value::Builtin(builtin::append)),
        "append-map" => Some(Value::Builtin(builtin::appendmap)),
        "map" => Some(Value::Builtin(builtin::map)),
        "filter" => Some(Value::Builtin(builtin::filter)),
        "foldl" => Some(Value::Builtin(builtin::foldl)),
        "count" => Some(Value::Builtin(builtin::count)),
        "member" => Some(Value::Builtin(builtin::member)),
        "string-length" => Some(Value::Builtin(builtin::string_length)),
        "string-downcase" => Some(Value::Builtin(builtin::string_downcase)),
        "string-upcase" => Some(Value::Builtin(builtin::string_upcase)),
        "string-append" => Some(Value::Builtin(builtin::string_append)),
        "substring" => Some(Value::Builtin(builtin::substring)),
        "string-split" => Some(Value::Builtin(builtin::string_split)),
        "string-join" => Some(Value::Builtin(builtin::string_join)),
        _ => None,
    }
}

mod builtin {
    use std::f64::EPSILON;
    use std::collections::VecDeque;
    use std::iter;
    use itertools::Itertools;

    use super::super::LispError;
    use super::super::sexp::Sexp;
    use super::{eval_call_nonspecial, eval_sexp, read_list, Env, Value};

    #[cfg_attr(feature = "cargo-clippy", allow(needless_pass_by_value))]
    pub fn if_(args: VecDeque<Sexp>, env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 3 {
            Err(LispError::BadArity("if"))
        } else {
            let cond = eval_sexp(&args[0], env)?;
            match cond {
                Value::Bool(b) if b => eval_sexp(&args[1], env),
                Value::Bool(b) if !b => eval_sexp(&args[2], env),
                _ => Err(LispError::Runtime("if took non-boolean condition")),
            }
        }
    }

    pub fn lambda(mut args: VecDeque<Sexp>, env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("lambda"))
        } else {
            let mut params_sexp = VecDeque::new();
            read_list(&args[0], env, &mut params_sexp);
            let mut params = Vec::with_capacity(params_sexp.len());
            for sexp in params_sexp {
                match sexp {
                    Sexp::Ident(s) => params.push(s),
                    Sexp::Null => break,
                    x => return Err(LispError::InvalidLambdaParameter(x)),
                }
            }
            let body = args.pop_back().unwrap();
            Ok(Value::Lambda(params, body))
        }
    }

    pub fn and(args: VecDeque<Sexp>, env: &mut Env) -> Result<Value, LispError> {
        args.into_iter().fold(Ok(Value::Bool(true)), |acc, x| {
            acc.and_then(|acc| {
                let x = eval_sexp(&x, env)?;
                match (acc, x) {
                    (Value::Bool(false), _) => Ok(Value::Bool(false)),
                    (Value::Bool(true), Value::Bool(b)) => Ok(Value::Bool(b)),
                    (_, x) => Err(LispError::ExpectedBool("and", x)),
                }
            })
        })
    }

    pub fn or(args: VecDeque<Sexp>, env: &mut Env) -> Result<Value, LispError> {
        args.into_iter().fold(Ok(Value::Bool(false)), |acc, x| {
            acc.and_then(|acc| {
                let x = eval_sexp(&x, env)?;
                match (acc, x) {
                    (Value::Bool(true), _) => Ok(Value::Bool(true)),
                    (Value::Bool(false), Value::Bool(b)) => Ok(Value::Bool(b)),
                    (_, x) => Err(LispError::ExpectedBool("or", x)),
                }
            })
        })
    }

    pub fn add(args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        args.into_iter().fold(Ok(Value::Integer(0)), |acc, x| {
            acc.and_then(|acc| match (acc, x) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l + r)),
                (Value::Integer(l), Value::Real(r)) => Ok(Value::Real(f64::from(l) + r)),
                (Value::Real(l), Value::Integer(r)) => Ok(Value::Real(l + f64::from(r))),
                (Value::Real(l), Value::Real(r)) => Ok(Value::Real(l + r)),
                (_, x) => Err(LispError::ExpectedNumber("add", x)),
            })
        })
    }

    pub fn sub(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(init) = args.pop_front() {
            if args.is_empty() {
                // negate
                match init {
                    Value::Integer(x) => Ok(Value::Integer(-x)),
                    Value::Real(x) => Ok(Value::Real(-x)),
                    x => Err(LispError::ExpectedNumber("sub", x)),
                }
            } else {
                match init {
                    Value::Integer(_) | Value::Real(_) => (),
                    x => return Err(LispError::ExpectedNumber("sub", x)),
                };
                args.into_iter().fold(Ok(init), |acc, x| {
                    acc.and_then(|acc| match (acc, x) {
                        (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l - r)),
                        (Value::Integer(l), Value::Real(r)) => Ok(Value::Real(f64::from(l) - r)),
                        (Value::Real(l), Value::Integer(r)) => Ok(Value::Real(l - f64::from(r))),
                        (Value::Real(l), Value::Real(r)) => Ok(Value::Real(l - r)),
                        (_, x) => Err(LispError::ExpectedNumber("sub", x)),
                    })
                })
            }
        } else {
            Ok(Value::Integer(0))
        }
    }

    pub fn not(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(v) = args.pop_front() {
            match v {
                Value::Bool(false) => Ok(Value::Bool(true)),
                _ => Ok(Value::Bool(false)),
            }
        } else {
            Err(LispError::BadArity("not"))
        }
    }

    pub fn mul(args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        args.into_iter().fold(Ok(Value::Integer(1)), |acc, x| {
            acc.and_then(|acc| match (acc, x) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l * r)),
                (Value::Integer(l), Value::Real(r)) => Ok(Value::Real(f64::from(l) * r)),
                (Value::Real(l), Value::Integer(r)) => Ok(Value::Real(l * f64::from(r))),
                (Value::Real(l), Value::Real(r)) => Ok(Value::Real(l * r)),
                (_, x) => Err(LispError::ExpectedNumber("mul", x)),
            })
        })
    }

    pub fn div(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(zv) = args.pop_front() {
            match zv {
                Value::Integer(_) | Value::Real(_) => (),
                x => return Err(LispError::ExpectedNumber("div", x)),
            }
            if args.is_empty() {
                // always return float for 1/z
                match zv {
                    Value::Integer(z) => Ok(Value::Real(1f64 / f64::from(z))),
                    Value::Real(z) => Ok(Value::Real(1f64 / z)),
                    _ => unreachable!(),
                }
            } else {
                args.into_iter().fold(Ok(zv), |acc, x| {
                    acc.and_then(|acc| match (acc, x) {
                        (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l / r)),
                        (Value::Integer(l), Value::Real(r)) => Ok(Value::Real(f64::from(l) / r)),
                        (Value::Real(l), Value::Integer(r)) => Ok(Value::Real(l / f64::from(r))),
                        (Value::Real(l), Value::Real(r)) => Ok(Value::Real(l / r)),
                        (_, x) => Err(LispError::ExpectedNumber("div", x)),
                    })
                })
            }
        } else {
            Err(LispError::BadArity("div"))
        }
    }

    pub fn quotient(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("quotient"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (n, m) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l / r)),
                (Value::Integer(l), Value::Real(r)) => Ok(Value::Real(f64::from(l / r as i32))),
                (Value::Real(l), Value::Integer(r)) => Ok(Value::Real(f64::from(l as i32 / r))),
                (Value::Real(l), Value::Real(r)) => Ok(Value::Real(f64::from(l as i32 / r as i32))),
                (Value::Integer(_), x) | (Value::Real(_), x) => {
                    Err(LispError::ExpectedNumber("quotient", x))
                }
                (x, _) => Err(LispError::ExpectedNumber("quotient", x)),
            }
        }
    }

    pub fn remainder(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("remainder"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (n, m) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l % r)),
                (Value::Integer(l), Value::Real(r)) => Ok(Value::Real(f64::from(l) % r)),
                (Value::Real(l), Value::Integer(r)) => Ok(Value::Real(l % f64::from(r))),
                (Value::Real(l), Value::Real(r)) => Ok(Value::Real(l % r)),
                (Value::Integer(_), x) | (Value::Real(_), x) => {
                    Err(LispError::ExpectedNumber("remainder", x))
                }
                (x, _) => Err(LispError::ExpectedNumber("remainder", x)),
            }
        }
    }

    pub fn modulo(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("modulo"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (n, m) {
                (Value::Integer(l), Value::Integer(r)) => if (l < 0) ^ (r < 0) {
                    Ok(Value::Integer((l % r) + r))
                } else {
                    Ok(Value::Integer(l % r))
                },
                (Value::Integer(l), Value::Real(r)) => if (l < 0) ^ (r < 0.0) {
                    Ok(Value::Real((f64::from(l) % r) + r))
                } else {
                    Ok(Value::Real(f64::from(l) % r))
                },
                (Value::Real(l), Value::Integer(r)) => if (l < 0.0) ^ (r < 0) {
                    Ok(Value::Real((l % f64::from(r)) + f64::from(r)))
                } else {
                    Ok(Value::Real(l % f64::from(r)))
                },
                (Value::Real(l), Value::Real(r)) => if (l < 0.0) ^ (r < 0.0) {
                    Ok(Value::Real((l % r) + r))
                } else {
                    Ok(Value::Real(l % r))
                },
                (Value::Integer(_), x) | (Value::Real(_), x) => {
                    Err(LispError::ExpectedNumber("modulo", x))
                }
                (x, _) => Err(LispError::ExpectedNumber("modulo", x)),
            }
        }
    }

    pub fn cons(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("cons"))
        } else {
            let car = args.pop_front().unwrap();
            let cdr = args.pop_front().unwrap();
            Ok(Value::Pair(Box::new(car), Box::new(cdr)))
        }
    }

    pub fn car(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(pair) = args.pop_front() {
            match pair {
                Value::Pair(car, _) => Ok(*car),
                _ => Err(LispError::Runtime("attempted car of non-pair")),
            }
        } else {
            Err(LispError::BadArity("car"))
        }
    }

    pub fn cdr(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(pair) = args.pop_front() {
            match pair {
                Value::Pair(_, cdr) => Ok(*cdr),
                _ => Err(LispError::Runtime("attempted cdr of non-pair")),
            }
        } else {
            Err(LispError::BadArity("cdr"))
        }
    }

    #[cfg_attr(feature = "cargo-clippy", allow(needless_pass_by_value))]
    pub fn eq(args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("eq"))
        } else {
            Ok(Value::Bool(eq_values(&args[0], &args[1])))
        }
    }

    fn eq_values(left: &Value, right: &Value) -> bool {
        match (left, right) {
            // bool
            (&Value::Bool(l), &Value::Bool(r)) => l == r,
            // numbers
            (&Value::Integer(l), &Value::Integer(r)) => l == r,
            (&Value::Integer(l), &Value::Real(r)) => f64::from(l) == r,
            (&Value::Real(l), &Value::Integer(r)) => l == f64::from(r),
            (&Value::Real(l), &Value::Real(r)) => l == r,
            // char
            (&Value::Char(l), &Value::Char(r)) => l == r,
            // str
            (&Value::Str(ref l), &Value::Str(ref r)) => l == r,
            // pair
            (&Value::Pair(ref l1, ref l2), &Value::Pair(ref r1, ref r2)) => {
                eq_values(&*l1, &*r1) && eq_values(&*l2, &*r2)
            }
            // null
            (&Value::Null, &Value::Null) => true,
            // functions and others
            _ => false,
        }
    }

    pub fn gt(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity(">"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (n, m) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Bool(l > r)),
                (Value::Integer(l), Value::Real(r)) => Ok(Value::Bool(f64::from(l) > r)),
                (Value::Real(l), Value::Integer(r)) => Ok(Value::Bool(l > f64::from(r))),
                (Value::Real(l), Value::Real(r)) => Ok(Value::Bool(l > r)),
                (Value::Integer(_), x) | (Value::Real(_), x) => {
                    Err(LispError::ExpectedNumber(">", x))
                }
                (x, _) => Err(LispError::ExpectedNumber(">", x)),
            }
        }
    }

    pub fn gte(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity(">="))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (n, m) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Bool(l >= r)),
                (Value::Integer(l), Value::Real(r)) => Ok(Value::Bool(f64::from(l) >= r)),
                (Value::Real(l), Value::Integer(r)) => Ok(Value::Bool(l >= f64::from(r))),
                (Value::Real(l), Value::Real(r)) => Ok(Value::Bool(l >= r)),
                (Value::Integer(_), x) | (Value::Real(_), x) => {
                    Err(LispError::ExpectedNumber(">=", x))
                }
                (x, _) => Err(LispError::ExpectedNumber(">=", x)),
            }
        }
    }

    pub fn lt(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("<"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (n, m) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Bool(l < r)),
                (Value::Integer(l), Value::Real(r)) => Ok(Value::Bool(f64::from(l) < r)),
                (Value::Real(l), Value::Integer(r)) => Ok(Value::Bool(l < f64::from(r))),
                (Value::Real(l), Value::Real(r)) => Ok(Value::Bool(l < r)),
                (Value::Integer(_), x) | (Value::Real(_), x) => {
                    Err(LispError::ExpectedNumber("<", x))
                }
                (x, _) => Err(LispError::ExpectedNumber("<", x)),
            }
        }
    }

    pub fn lte(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("<="))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (n, m) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Bool(l <= r)),
                (Value::Integer(l), Value::Real(r)) => Ok(Value::Bool(f64::from(l) <= r)),
                (Value::Real(l), Value::Integer(r)) => Ok(Value::Bool(l <= f64::from(r))),
                (Value::Real(l), Value::Real(r)) => Ok(Value::Bool(l <= r)),
                (Value::Integer(_), x) | (Value::Real(_), x) => {
                    Err(LispError::ExpectedNumber("<=", x))
                }
                (x, _) => Err(LispError::ExpectedNumber("<=", x)),
            }
        }
    }

    pub fn neq(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("!="))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (n, m) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Bool(l != r)),
                (Value::Integer(l), Value::Real(r)) => {
                    Ok(Value::Bool((f64::from(l) - r).abs() > EPSILON))
                }
                (Value::Real(l), Value::Integer(r)) => {
                    Ok(Value::Bool((l - f64::from(r)).abs() > EPSILON))
                }
                (Value::Real(l), Value::Real(r)) => Ok(Value::Bool((l - r).abs() > EPSILON)),
                (Value::Integer(_), x) | (Value::Real(_), x) => {
                    Err(LispError::ExpectedNumber("!=", x))
                }
                (x, _) => Err(LispError::ExpectedNumber("!=", x)),
            }
        }
    }

    pub fn is_pair(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(x) = args.pop_front() {
            match x {
                Value::Pair(_, _) => Ok(Value::Bool(true)),
                _ => Ok(Value::Bool(false)),
            }
        } else {
            Err(LispError::BadArity("pair?"))
        }
    }

    pub fn is_null(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(x) = args.pop_front() {
            match x {
                Value::Null => Ok(Value::Bool(true)),
                _ => Ok(Value::Bool(false)),
            }
        } else {
            Err(LispError::BadArity("null?"))
        }
    }

    pub fn identity(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        args.pop_front().ok_or(LispError::BadArity("identity"))
    }

    pub fn list(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(car) = args.pop_back() {
            let car = Box::new(car);
            let mut l = Value::Pair(car, Box::new(Value::Null));
            for item in args.into_iter().rev() {
                let cdr = Box::new(l);
                l = Value::Pair(Box::new(item), cdr);
            }
            Ok(l)
        } else {
            Ok(Value::Null)
        }
    }

    pub fn list_ref(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("list-ref"))
        } else {
            let lst = args.pop_front().unwrap();
            let idx = args.pop_front().unwrap();
            match (lst, idx) {
                (Value::Pair(l, r), Value::Integer(i)) => if i >= 0 {
                    let lst = list_of_pair(Value::Pair(l, r));
                    lst.into_iter()
                        .nth(i as usize)
                        .ok_or(LispError::Runtime("list-ref index out of bound"))
                } else {
                    Err(LispError::Runtime("list-ref expected nonnegative integer"))
                },
                _ => Err(LispError::Runtime("list-ref expected pair and integer")),
            }
        }
    }

    fn list_of_pair(mut v: Value) -> Vec<Value> {
        let mut lst = Vec::new();
        while let Value::Pair(car, cdr) = v {
            lst.push(*car);
            v = *cdr;
        }
        lst.push(v);
        lst
    }

    pub fn append(mut args: VecDeque<Value>, env: &mut Env) -> Result<Value, LispError> {
        match args.len() {
            0 => Ok(Value::Null),
            1 => Ok(args.pop_front().unwrap()),
            _ => {
                let mut vs = VecDeque::new();
                for arg in args {
                    match arg {
                        Value::Null => (),
                        Value::Pair(car, cdr) => {
                            let mut lst: VecDeque<_> = list_of_pair(Value::Pair(car, cdr)).into();
                            lst.pop_back(); // no null
                            vs.append(&mut lst);
                        }
                        x => return Err(LispError::ExpectedList("append", x)),
                    }
                }
                list(vs, env)
            }
        }
    }

    pub fn appendmap(mut args: VecDeque<Value>, env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("append-map"))
        } else {
            let f = args.pop_front().unwrap();
            let xs = args.pop_front().unwrap();
            match xs {
                Value::Pair(_, _) => {
                    let mut lst = list_of_pair(xs);
                    lst.pop(); // no null
                    let items = lst.into_iter()
                        .zip(iter::repeat(f))
                        .map(|(v, f)| eval_call_nonspecial(f, vec![v].into(), env))
                        .collect::<Result<VecDeque<_>, LispError>>()?;
                    let mut vs = VecDeque::new();
                    for item in items {
                        match item {
                            Value::Null => (),
                            Value::Pair(car, cdr) => {
                                let mut lst: VecDeque<_> =
                                    list_of_pair(Value::Pair(car, cdr)).into();
                                lst.pop_back(); // no null
                                vs.append(&mut lst);
                            }
                            x => return Err(LispError::ExpectedList("append-map", x)),
                        }
                    }
                    list(vs, env)
                }
                Value::Null => Ok(Value::Null),
                x => Err(LispError::ExpectedList("append-map", x)),
            }
        }
    }

    pub fn map(mut args: VecDeque<Value>, env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("map"))
        } else {
            let f = args.pop_front().unwrap();
            let xs = args.pop_front().unwrap();
            match xs {
                Value::Pair(_, _) => {
                    let mut lst = list_of_pair(xs);
                    lst.pop(); // no null
                    let items = lst.into_iter()
                        .zip(iter::repeat(f))
                        .map(|(v, f)| eval_call_nonspecial(f, vec![v].into(), env))
                        .collect::<Result<VecDeque<_>, LispError>>()?;
                    list(items, env)
                }
                Value::Null => Ok(Value::Null),
                x => Err(LispError::ExpectedList("map", x)),
            }
        }
    }

    pub fn filter(mut args: VecDeque<Value>, env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("filter"))
        } else {
            let f = args.pop_front().unwrap();
            let xs = args.pop_front().unwrap();
            match xs {
                Value::Pair(_, _) => {
                    let mut lst = list_of_pair(xs);
                    lst.pop(); // no null
                    let evaluated = lst.into_iter()
                        .zip(iter::repeat(f))
                        .map(|(v, f)| {
                            eval_call_nonspecial(f, vec![v.clone()].into(), env).map(|x| (x, v))
                        })
                        .collect::<Result<Vec<_>, LispError>>()?;
                    let items = evaluated
                        .into_iter()
                        .filter_map(|(x, v)| match x {
                            Value::Bool(true) => Some(v),
                            _ => None,
                        })
                        .collect();
                    list(items, env)
                }
                Value::Null => Ok(Value::Null),
                x => Err(LispError::ExpectedList("map", x)),
            }
        }
    }

    pub fn foldl(mut args: VecDeque<Value>, env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 3 {
            Err(LispError::BadArity("foldl"))
        } else {
            let f = args.pop_front().unwrap();
            let init = args.pop_front().unwrap();
            let lst = args.pop_front().unwrap();
            match lst {
                Value::Pair(car, cdr) => {
                    let mut cur = init;
                    let mut lst = list_of_pair(Value::Pair(car, cdr));
                    lst.pop(); // no null
                    for x in lst {
                        cur = eval_call_nonspecial(f.clone(), vec![x, cur].into(), env)?
                    }
                    Ok(cur)
                }
                Value::Null => Ok(init),
                x => Err(LispError::ExpectedList("foldl", x)),
            }
        }
    }

    pub fn count(mut args: VecDeque<Value>, env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("count"))
        } else {
            let f = args.pop_front().unwrap();
            let lst = args.pop_front().unwrap();
            match lst {
                Value::Pair(car, cdr) => {
                    let mut count = 0;
                    let mut lst = list_of_pair(Value::Pair(car, cdr));
                    lst.pop(); // no null
                    for x in lst {
                        match eval_call_nonspecial(f.clone(), vec![x].into(), env)? {
                            Value::Bool(true) => count += 1,
                            Value::Bool(false) => (),
                            x => return Err(LispError::ExpectedBool("count", x)),
                        }
                    }
                    Ok(Value::Integer(count))
                }
                Value::Null => Ok(Value::Integer(0)),
                x => Err(LispError::ExpectedList("count", x)),
            }
        }
    }

    pub fn member(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("member"))
        } else {
            let needle = args.pop_front().unwrap();
            let mut haystack = args.pop_front().unwrap();
            while let Value::Pair(car, cdr) = haystack {
                if eq_values(&needle, &*car) {
                    return Ok(Value::Pair(car, cdr));
                }
                haystack = *cdr
            }
            Ok(Value::Bool(false))
        }
    }

    pub fn string_length(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(s) = args.pop_front() {
            match s {
                Value::Str(s) => Ok(Value::Integer(s.chars().count() as i32)),
                x => Err(LispError::ExpectedStr("string-length", x)),
            }
        } else {
            Err(LispError::BadArity("string-length"))
        }
    }

    pub fn string_downcase(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(s) = args.pop_front() {
            match s {
                Value::Str(s) => Ok(Value::Str(s.to_lowercase())),
                x => Err(LispError::ExpectedStr("string-downcase", x)),
            }
        } else {
            Err(LispError::BadArity("string-downcase"))
        }
    }

    pub fn string_upcase(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(s) = args.pop_front() {
            match s {
                Value::Str(s) => Ok(Value::Str(s.to_uppercase())),
                x => Err(LispError::ExpectedStr("stirng-upcase", x)),
            }
        } else {
            Err(LispError::BadArity("string-upcase"))
        }
    }

    pub fn string_append(args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        let strs = args.into_iter()
            .map(|v| match v {
                Value::Str(s) => Ok(s),
                x => Err(LispError::ExpectedStr("string-append", x)),
            })
            .collect::<Result<Vec<_>, LispError>>()?;
        Ok(Value::Str(strs.into_iter().join("")))
    }

    pub fn substring(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(s) = args.pop_front() {
            match s {
                Value::Str(s) => {
                    if let Some(startv) = args.pop_front() {
                        match startv {
                            Value::Integer(start) if start >= 0 => {
                                if let Some(endv) = args.pop_front() {
                                    match endv {
                                        Value::Integer(end) if end >= start => Ok(Value::Str(
                                            s.chars()
                                                .skip(start as usize)
                                                .take((end - start) as usize)
                                                .collect(),
                                        )),
                                        Value::Integer(_) => Err(LispError::Runtime(
                                            "list-ref end was less that start",
                                        )),
                                        x => Err(LispError::ExpectedInteger("substring", x)),
                                    }
                                } else {
                                    Ok(Value::Str(s.chars().skip(start as usize).collect()))
                                }
                            }
                            Value::Integer(_) => {
                                Err(LispError::Runtime("list-ref expected nonnegative integer"))
                            }
                            x => Err(LispError::ExpectedInteger("substring", x)),
                        }
                    } else {
                        Err(LispError::BadArity("string-join"))
                    }
                }
                x => Err(LispError::ExpectedStr("substring", x)),
            }
        } else {
            Err(LispError::BadArity("string-join"))
        }
    }

    pub fn string_split(mut args: VecDeque<Value>, env: &mut Env) -> Result<Value, LispError> {
        if let Some(xs) = args.pop_front() {
            match xs {
                Value::Str(s) => {
                    if let Some(sepv) = args.pop_front() {
                        match sepv {
                            Value::Str(sep) => {
                                let vals =
                                    s.split(&sep).map(|s| Value::Str(String::from(s))).collect();
                                list(vals, env)
                            }
                            x => Err(LispError::ExpectedStr("string-split", x)),
                        }
                    } else {
                        let vals = s.split_whitespace()
                            .map(|s| Value::Str(String::from(s)))
                            .collect();
                        list(vals, env)
                    }
                }
                x => Err(LispError::ExpectedStr("string-split", x)),
            }
        } else {
            Err(LispError::BadArity("string-join"))
        }
    }

    pub fn string_join(mut args: VecDeque<Value>, _env: &mut Env) -> Result<Value, LispError> {
        if let Some(xs) = args.pop_front() {
            match xs {
                Value::Pair(_, _) => {
                    let mut lst = list_of_pair(xs);
                    lst.pop(); // no null
                    let strs = lst.into_iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s),
                            x => Err(LispError::ExpectedStr("string-join", x)),
                        })
                        .collect::<Result<Vec<String>, LispError>>()?;
                    if let Some(sepv) = args.pop_front() {
                        match sepv {
                            Value::Str(sep) => Ok(Value::Str(strs.into_iter().join(&sep))),
                            x => Err(LispError::ExpectedStr("string-join", x)),
                        }
                    } else {
                        Ok(Value::Str(strs.into_iter().join("")))
                    }
                }
                Value::Null => Ok(Value::Str(String::new())),
                x => Err(LispError::ExpectedList("string-join", x)),
            }
        } else {
            Err(LispError::BadArity("string-join"))
        }
    }
}
